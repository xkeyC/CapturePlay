//! Upscaling with NVIDIA RTX Video Super Resolution through NGX on D3D12.
//!
//! Recorded into the frame's own command list: encode the working frame as
//! PQ/BT.2020 into an `R10G10B10A2_UNORM` texture (PQ measured 1–3 dB better
//! than SDR-style encodings for this model), evaluate VSR into a second one,
//! decode into the working frame of the target size. Everything is ordered
//! on the direct queue — no CUDA interop, no CPU waits — and the result is
//! the frame that was submitted (the Video Effects SDK route returned the
//! previous one; see `upscale_bench::slow_pan_temporal_stability`).
//!
//! The feature (model) is created once on a loader thread; sizes and quality
//! are per-call parameters, so resizing never reloads it.

use std::path::PathBuf;
use std::rc::Rc;

use cp_nvidia::ngx::{self, VsrFeature, VsrQuality};
use windows::Win32::Foundation::{CloseHandle, E_FAIL, E_NOTIMPL};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R10G10B10A2_UNORM;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Result;

use crate::descriptors::StagingHeap;
use crate::gpu::Gpu;
use crate::interp_vfg::Constants;
use crate::renderer::{RecordCtx, Upscaler};
use crate::shaders::{self, ComputePipeline, dispatch_2d, set_compute_constants};
use crate::texture::{Texture, TextureDesc, flush_barriers};

type Size = (u32, u32);

/// The created feature, handed from the loader thread.
struct Loaded(VsrFeature);
// SAFETY: the NGX handle is only used under the NGX lock (see `cp_nvidia::ngx`).
unsafe impl Send for Loaded {}

type Loader = std::thread::JoinHandle<std::result::Result<Loaded, String>>;

/// RGB10A2 surfaces NGX reads and writes, for one `(src, dst)` pair.
struct Surfaces {
    src: Size,
    dst: Size,
    input: Texture,
    output: Texture,
}

pub struct NgxVsrUpscaler {
    encode: ComputePipeline,
    decode: ComputePipeline,
    quality: VsrQuality,
    feature_dirs: Vec<PathBuf>,
    data_dir: PathBuf,
    feature: Option<Loaded>,
    loading: Option<Loader>,
    failed: bool,
    surfaces: Option<Surfaces>,
}

impl std::fmt::Debug for NgxVsrUpscaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgxVsrUpscaler")
            .field("quality", &self.quality)
            .finish_non_exhaustive()
    }
}

impl NgxVsrUpscaler {
    /// Fails fast when NGX was not built in, the GPU is not NVIDIA or no
    /// `nvngx_vsr.dll` is found in `feature_dirs`.
    pub fn new(
        gpu: &Gpu,
        feature_dirs: Vec<PathBuf>,
        data_dir: PathBuf,
        quality: VsrQuality,
    ) -> Result<Self> {
        if !ngx::built() {
            return Err(windows::core::Error::new(
                E_NOTIMPL,
                ngx::NgxError::NotBuilt.to_string(),
            ));
        }
        if !gpu.is_nvidia() || !feature_dirs.iter().any(|d| ngx::runtime_present(d)) {
            return Err(windows::core::Error::new(
                E_NOTIMPL,
                "nvngx_vsr.dll not found (RTX Video SDK runtime, see runtime/README.md)",
            ));
        }
        Ok(Self {
            encode: ComputePipeline::new(&gpu.device, shaders::NGX_ENCODE)?,
            decode: ComputePipeline::new(&gpu.device, shaders::NGX_DECODE)?,
            quality,
            feature_dirs,
            data_dir,
            feature: None,
            loading: None,
            failed: false,
            surfaces: None,
        })
    }

    pub fn set_quality(&mut self, quality: VsrQuality) {
        self.quality = quality;
    }

    fn spawn_loader(&mut self, gpu: &Gpu) {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let dirs = self.feature_dirs.clone();
        let data = self.data_dir.clone();
        let handle = std::thread::Builder::new()
            .name("ngx-load".into())
            .spawn(move || load(&device, &queue, &dirs, &data));
        match handle {
            Ok(h) => self.loading = Some(h),
            Err(e) => {
                tracing::error!("cannot start NGX loader: {e}");
                self.failed = true;
            }
        }
    }

    fn ensure_surfaces(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        src: Size,
        dst: Size,
    ) -> Result<()> {
        if self
            .surfaces
            .as_ref()
            .is_some_and(|s| s.src == src && s.dst == dst)
        {
            return Ok(());
        }
        let surface = |(w, h): Size| {
            Texture::new(
                gpu,
                heap,
                TextureDesc::new(w, h, DXGI_FORMAT_R10G10B10A2_UNORM).uav(),
            )
        };
        self.surfaces = Some(Surfaces {
            src,
            dst,
            input: surface(src)?,
            output: surface(dst)?,
        });
        Ok(())
    }
}

/// Initializes NGX and creates the feature on a private command list.
fn load(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    dirs: &[PathBuf],
    data_dir: &std::path::Path,
) -> std::result::Result<Loaded, String> {
    let _ = std::fs::create_dir_all(data_dir);
    let t0 = std::time::Instant::now();
    // SAFETY: plain D3D12 object creation and submission on a thread-safe
    // queue; the list is executed and waited for before returning.
    unsafe {
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(|e| e.to_string())?;
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .map_err(|e| e.to_string())?;
        let feature =
            VsrFeature::create(device, &list, dirs, data_dir).map_err(|e| e.to_string())?;
        list.Close().map_err(|e| e.to_string())?;
        queue.ExecuteCommandLists(&[Some(
            windows::core::Interface::cast(&list)
                .map_err(|e: windows::core::Error| e.to_string())?,
        )]);
        let fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|e| e.to_string())?;
        queue.Signal(&fence, 1).map_err(|e| e.to_string())?;
        let event = CreateEventW(None, false, false, None).map_err(|e| e.to_string())?;
        if fence.GetCompletedValue() < 1 {
            fence
                .SetEventOnCompletion(1, event)
                .map_err(|e| e.to_string())?;
            WaitForSingleObject(event, INFINITE);
        }
        let _ = CloseHandle(event);
        tracing::info!(
            ms = t0.elapsed().as_millis() as u64,
            "NVIDIA RTX VSR (NGX D3D12) ready"
        );
        Ok(Loaded(feature))
    }
}

impl Upscaler for NgxVsrUpscaler {
    fn name(&self) -> &'static str {
        "NVIDIA RTX VSR"
    }

    fn factors(&self) -> &'static [f32] {
        &cp_core::scale::VSR_FACTORS
    }

    fn prepare(&mut self, gpu: &Gpu, heap: &Rc<StagingHeap>, src: Size, dst: Size) -> bool {
        if self.failed {
            return false;
        }
        if self.feature.is_none() {
            match self.loading.take() {
                None => {
                    self.spawn_loader(gpu);
                    return false;
                }
                Some(h) if !h.is_finished() => {
                    self.loading = Some(h);
                    return false;
                }
                Some(h) => match h.join() {
                    Ok(Ok(f)) => self.feature = Some(f),
                    Ok(Err(e)) => {
                        tracing::error!("RTX VSR unavailable: {e}");
                        self.failed = true;
                        return false;
                    }
                    Err(_) => {
                        tracing::error!("RTX VSR loader panicked");
                        self.failed = true;
                        return false;
                    }
                },
            }
        }
        match self.ensure_surfaces(gpu, heap, src, dst) {
            Ok(()) => true,
            Err(e) => {
                tracing::error!("RTX VSR surfaces: {e}");
                self.failed = true;
                false
            }
        }
    }

    fn upscale(&mut self, ctx: &mut RecordCtx<'_>, src: &Texture, out: &Texture) -> Result<()> {
        let (Some(Loaded(feature)), Some(s)) = (self.feature.as_ref(), self.surfaces.as_ref())
        else {
            return Err(windows::core::Error::new(E_FAIL, "RTX VSR not prepared"));
        };
        if s.src != (src.width, src.height) || s.dst != (out.width, out.height) {
            return Err(windows::core::Error::new(
                E_FAIL,
                "RTX VSR prepared for another size",
            ));
        }
        let device = &ctx.gpu.device;
        let constants = |w: u32, h: u32| Constants {
            width: w,
            height: h,
            hdr: 1,
            ..Default::default()
        };

        // 1. Working frame → PQ RGB10A2.
        let mut b = Vec::new();
        src.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        s.input
            .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.encode.bind(ctx.list);
        let srv = ctx.ring.table(device, &[src.srv.handle]);
        let uav = ctx.ring.table(device, &[s.input.uav_handle()]);
        set_compute_constants(ctx.list, &constants(src.width, src.height));
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, src.width, src.height);

        // 2. VSR.
        s.input
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        s.output
            .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        feature
            .evaluate(
                ctx.list,
                &s.input.resource,
                s.src,
                &s.output.resource,
                s.dst,
                self.quality,
            )
            .map_err(|e| windows::core::Error::new(E_FAIL, e.to_string()))?;
        // NGX leaves its own pipeline state and heaps bound.
        // SAFETY: the ring heap is the frame's shader-visible heap.
        unsafe {
            ctx.list
                .SetDescriptorHeaps(&[Some(ctx.ring.heap().clone())]);
        }

        // 3. PQ RGB10A2 → working frame of the target size.
        s.output
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        out.transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.decode.bind(ctx.list);
        let srv = ctx.ring.table(device, &[s.output.srv.handle]);
        let uav = ctx.ring.table(device, &[out.uav_handle()]);
        set_compute_constants(ctx.list, &constants(out.width, out.height));
        // SAFETY: as above.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, out.width, out.height);
        Ok(())
    }
}
