//! DLSS Neural Rendering (NGX feature 18) as a per-frame enhancer.
//!
//! Runs on every unique captured frame right after color conversion, so the
//! frames fed to interpolation and to super resolution already carry the
//! model's edit. The model is a same-resolution, display-referred network:
//!
//!   linear scRGB frame -> sRGB proxy (HDR: invertible highlight roll-off)
//!     -> NGX feature 18 (color, constant depth, zero motion)
//!     -> sRGB edit -> bounded linear residual on the original frame
//!
//! Everything is recorded into the frame's own command list on the direct
//! queue; no CUDA interop and no CPU waits. The feature is created once per
//! source size on a loader thread (~1-3 s), because creation compiles the
//! model's kernels while evaluation is the cheap per-frame step.
//!
//! Guides: capture cards provide no depth or motion vectors. A constant depth
//! and zero motion let the network work spatially; motion from an optical
//! flow stage can be wired in here later without changing the rest.

use std::path::PathBuf;
use std::rc::Rc;

use cp_nvidia::ngx::{self, NrFeature, NrFrame, NrOptions, NrRoute, NrTuning};
use windows::Win32::Foundation::{CloseHandle, E_FAIL, E_NOTIMPL};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, Result};

use crate::descriptors::StagingHeap;
use crate::gpu::Gpu;
use crate::renderer::{Enhancer, FrameInput, RecordCtx};
use crate::shaders::{self, ComputePipeline, dispatch_2d, set_compute_constants};
use crate::texture::{Texture, TextureDesc, flush_barriers};

type Size = (u32, u32);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EncodeConstants {
    width: u32,
    height: u32,
    hdr: u32,
    sdr_white: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeConstants {
    width: u32,
    height: u32,
    hdr: u32,
    sdr_white: f32,
    detail: f32,
    colour: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FillConstants {
    width: u32,
    height: u32,
    value: f32,
    _pad: u32,
}

/// The created feature, handed from the loader thread.
struct Loaded(NrFeature);
// SAFETY: NGX calls are serialized by cp_nvidia::ngx's lock.
unsafe impl Send for Loaded {}

type Loader = std::thread::JoinHandle<std::result::Result<Loaded, String>>;

struct Surfaces {
    size: Size,
    /// sRGB proxy the model reads.
    proxy: Texture,
    /// Constant depth guide (R32_FLOAT).
    depth: Texture,
    /// Zero motion guide (R16G16_FLOAT, render pixels).
    motion: Texture,
    /// The model's sRGB edit.
    edited: Texture,
    /// Composite result, copied back over the working frame.
    composed: Texture,
    guides_filled: std::cell::Cell<bool>,
}

/// How the model's edit is composited back over the linear frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NrPost {
    /// Residual gain (1.0 = the model's edit as returned).
    pub detail: f32,
    /// 0 = luminance only, 1 = the model's colour too.
    pub colour: f32,
}

impl Default for NrPost {
    fn default() -> Self {
        Self {
            detail: 1.0,
            colour: 1.0,
        }
    }
}

pub struct NgxNrEnhancer {
    encode: ComputePipeline,
    composite: ComputePipeline,
    fill_depth: ComputePipeline,
    fill_motion: ComputePipeline,
    options: NrOptions,
    route: NrRoute,
    tuning: NrTuning,
    detail: f32,
    colour: f32,
    feature_dirs: Vec<PathBuf>,
    data_dir: PathBuf,
    hdr: bool,
    sdr_white: f32,
    feature: Option<Loaded>,
    feature_size: Option<Size>,
    loading: Option<Loader>,
    failed: bool,
    surfaces: Option<Surfaces>,
}

impl std::fmt::Debug for NgxNrEnhancer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgxNrEnhancer")
            .field("options", &self.options)
            .field("hdr", &self.hdr)
            .field("ready", &self.feature.is_some())
            .finish_non_exhaustive()
    }
}

impl NgxNrEnhancer {
    /// Fails fast when NGX was not built in or the GPU is not NVIDIA. The
    /// feature itself is created lazily (and may be unavailable: feature 18 is
    /// not part of the public NGX feature table yet).
    pub fn new(
        gpu: &Gpu,
        feature_dirs: Vec<PathBuf>,
        data_dir: PathBuf,
        options: NrOptions,
        route: NrRoute,
        tuning: NrTuning,
        post: NrPost,
    ) -> Result<Self> {
        if !ngx::built() {
            return Err(windows::core::Error::new(
                E_NOTIMPL,
                ngx::NgxError::NotBuilt.to_string(),
            ));
        }
        if !gpu.is_nvidia() {
            return Err(windows::core::Error::new(
                E_NOTIMPL,
                "DLSS NR needs an NVIDIA RTX GPU",
            ));
        }
        Ok(Self {
            encode: ComputePipeline::new(&gpu.device, shaders::NR_ENCODE)?,
            composite: ComputePipeline::new(&gpu.device, shaders::NR_COMPOSITE)?,
            fill_depth: ComputePipeline::new(&gpu.device, shaders::NR_FILL_DEPTH)?,
            fill_motion: ComputePipeline::new(&gpu.device, shaders::NR_FILL_MOTION)?,
            options,
            route,
            tuning,
            detail: post.detail,
            colour: post.colour,
            feature_dirs,
            data_dir,
            hdr: false,
            sdr_white: 1.0,
            feature: None,
            feature_size: None,
            loading: None,
            failed: false,
            surfaces: None,
        })
    }

    fn spawn_loader(&mut self, gpu: &Gpu, size: Size) {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let dirs = self.feature_dirs.clone();
        let data = self.data_dir.clone();
        let options = self.options;
        let route = self.route.clone();
        self.feature_size = Some(size);
        let handle = std::thread::Builder::new()
            .name("dlss-nr-load".into())
            .spawn(move || load(&device, &queue, &dirs, &data, size, options, &route));
        match handle {
            Ok(h) => self.loading = Some(h),
            Err(e) => {
                tracing::error!("cannot start DLSS NR loader: {e}");
                self.failed = true;
            }
        }
    }

    /// Non-blocking: starts or joins the loader, rebuilds size-dependent
    /// surfaces. False means "pass the frame through for now".
    fn prepare(&mut self, gpu: &Gpu, heap: &Rc<StagingHeap>, size: Size) -> bool {
        if self.failed {
            return false;
        }
        if self.feature.is_some() && self.feature_size != Some(size) {
            // The feature latches its size at creation.
            self.feature = None;
            self.surfaces = None;
            self.feature_size = None;
        }
        if self.feature.is_none() && self.loading.is_none() {
            self.spawn_loader(gpu, size);
            return false;
        }
        if self.feature.is_none() {
            let Some(h) = self.loading.take() else {
                return false;
            };
            if !h.is_finished() {
                self.loading = Some(h);
                return false;
            }
            match h.join() {
                Ok(Ok(f)) => self.feature = Some(f),
                Ok(Err(e)) => {
                    tracing::warn!("DLSS NR unavailable: {e}");
                    self.failed = true;
                    return false;
                }
                Err(_) => {
                    tracing::error!("DLSS NR loader panicked");
                    self.failed = true;
                    return false;
                }
            }
        }
        if self.surfaces.as_ref().is_none_or(|s| s.size != size) {
            match surfaces(gpu, heap, size) {
                Ok(s) => self.surfaces = Some(s),
                Err(e) => {
                    tracing::error!("DLSS NR surfaces: {e}");
                    self.failed = true;
                    return false;
                }
            }
        }
        true
    }
}

fn surfaces(gpu: &Gpu, heap: &Rc<StagingHeap>, size: Size) -> Result<Surfaces> {
    let tex = |format| Texture::new(gpu, heap, TextureDesc::new(size.0, size.1, format).uav());
    Ok(Surfaces {
        size,
        proxy: tex(DXGI_FORMAT_R16G16B16A16_FLOAT)?,
        depth: tex(DXGI_FORMAT_R32_FLOAT)?,
        motion: tex(DXGI_FORMAT_R16G16_FLOAT)?,
        edited: tex(DXGI_FORMAT_R16G16B16A16_FLOAT)?,
        composed: tex(DXGI_FORMAT_R16G16B16A16_FLOAT)?,
        guides_filled: std::cell::Cell::new(false),
    })
}

/// Initializes NGX and creates the feature on a private command list.
fn load(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    dirs: &[PathBuf],
    data_dir: &std::path::Path,
    size: Size,
    options: NrOptions,
    route: &NrRoute,
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
        let feature = NrFeature::create(device, &list, dirs, data_dir, size, options, route)
            .map_err(|e| e.to_string())?;
        list.Close().map_err(|e| e.to_string())?;
        queue.ExecuteCommandLists(&[Some(
            Interface::cast(&list).map_err(|e: windows::core::Error| e.to_string())?,
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
            w = size.0,
            h = size.1,
            "DLSS Neural Rendering ready"
        );
        Ok(Loaded(feature))
    }
}

impl Enhancer for NgxNrEnhancer {
    fn name(&self) -> &'static str {
        "NVIDIA DLSS NR"
    }

    fn ready(&self) -> bool {
        !self.failed && self.feature.is_some() && self.surfaces.is_some()
    }

    fn set_strength(&mut self, strength: f32) {
        self.detail = strength;
    }

    fn set_nr_tuning(&mut self, tuning: NrTuning) {
        self.tuning = tuning;
    }

    fn set_content(&mut self, hdr: bool, sdr_white: f32) {
        self.hdr = hdr;
        self.sdr_white = sdr_white.max(1e-3);
    }

    fn enhance(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        frame: FrameInput<'_>,
        reset: bool,
    ) -> Result<()> {
        let size = (frame.tex.width, frame.tex.height);
        if size.0 == 0 || size.1 == 0 || !self.prepare(ctx.gpu, ctx.heap, size) {
            return Ok(());
        }
        let (Some(Loaded(feature)), Some(s)) = (self.feature.as_ref(), self.surfaces.as_ref())
        else {
            return Ok(());
        };
        let device = &ctx.gpu.device;
        let mut b = Vec::new();

        // Constant guides, once per size.
        if !s.guides_filled.get() {
            s.depth
                .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            s.motion
                .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            flush_barriers(ctx.list, &mut b);
            self.fill_depth.bind(ctx.list);
            let fill = FillConstants {
                width: size.0,
                height: size.1,
                value: 0.5,
                _pad: 0,
            };
            set_compute_constants(ctx.list, &fill);
            let uav = ctx.ring.table(device, &[s.depth.uav_handle()]);
            // SAFETY: descriptor table is in the bound heap.
            unsafe { ctx.list.SetComputeRootDescriptorTable(1, uav) };
            dispatch_2d(ctx.list, size.0, size.1);
            self.fill_motion.bind(ctx.list);
            set_compute_constants(ctx.list, &fill);
            let uav = ctx.ring.table(device, &[s.motion.uav_handle()]);
            // SAFETY: as above.
            unsafe { ctx.list.SetComputeRootDescriptorTable(1, uav) };
            dispatch_2d(ctx.list, size.0, size.1);
            s.guides_filled.set(true);
        }

        // 1. Working frame -> display-referred sRGB proxy.
        frame
            .tex
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        s.proxy
            .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.encode.bind(ctx.list);
        set_compute_constants(
            ctx.list,
            &EncodeConstants {
                width: size.0,
                height: size.1,
                hdr: u32::from(self.hdr),
                sdr_white: self.sdr_white,
            },
        );
        let srv = ctx.ring.table(device, &[frame.tex.srv.handle]);
        let uav = ctx.ring.table(device, &[s.proxy.uav_handle()]);
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, size.0, size.1);

        // 2. The model. NGX changes the list's state and bindings.
        s.proxy
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        s.edited
            .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        feature
            .evaluate(
                ctx.list,
                &NrFrame {
                    color: &s.proxy.resource,
                    depth: &s.depth.resource,
                    motion: &s.motion.resource,
                    output: &s.edited.resource,
                    reset,
                    tuning: self.tuning,
                },
            )
            .map_err(|e| windows::core::Error::new(E_FAIL, e.to_string()))?;
        // SAFETY: the ring heap is the frame's shader-visible heap.
        unsafe {
            ctx.list
                .SetDescriptorHeaps(&[Some(ctx.ring.heap().clone())]);
        }

        // 3. Bounded residual composite over the untouched linear frame.
        frame
            .tex
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        s.edited
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        s.composed
            .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.composite.bind(ctx.list);
        set_compute_constants(
            ctx.list,
            &CompositeConstants {
                width: size.0,
                height: size.1,
                hdr: u32::from(self.hdr),
                sdr_white: self.sdr_white,
                detail: self.detail,
                colour: self.colour,
                _pad: [0.0; 2],
            },
        );
        let orig = ctx.ring.table(device, &[frame.tex.srv.handle]);
        let edit = ctx.ring.table(device, &[s.edited.srv.handle]);
        let uav = ctx.ring.table(device, &[s.composed.uav_handle()]);
        // SAFETY: as above.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, orig);
            ctx.list.SetComputeRootDescriptorTable(2, edit);
            ctx.list.SetComputeRootDescriptorTable(3, uav);
        }
        dispatch_2d(ctx.list, size.0, size.1);

        // 4. Back into the caller's working frame (the API is in place).
        s.composed
            .transition(&mut b, D3D12_RESOURCE_STATE_COPY_SOURCE);
        frame.tex.transition(&mut b, D3D12_RESOURCE_STATE_COPY_DEST);
        flush_barriers(ctx.list, &mut b);
        // SAFETY: same size, format and resource type; GPU-ordered copy.
        unsafe {
            ctx.list
                .CopyResource(&frame.tex.resource, &s.composed.resource)
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Harness, image, pattern};

    /// Proxy + residual composite is an identity when the model returns its
    /// input unchanged (SDR and HDR), and a zero strength ignores the edit.
    /// This pins the color contract the rest of the frame pipeline relies on.
    #[test]
    #[ignore = "needs a D3D12 GPU"]
    fn proxy_round_trip_is_identity() {
        for (hdr, detail) in [(0u32, 1.0f32), (1, 1.0), (1, 0.0)] {
            let mut h = Harness::new();
            let (w, ht) = (256u32, 144u32);
            // SDR white is 1.0 in scRGB units, and the proxy saturates:
            // stay inside 0..1 for the SDR case or clipping is expected.
            let (amplitude, white) = if hdr == 0 { (0.45, 1.0) } else { (8.0, 2.5) };
            let src = image(w, ht, |x, y| amplitude * pattern(x, y));
            let frame = h.upload(w, ht, &src);
            let surface = |f| {
                Texture::new(&h.gpu, &h.heap, TextureDesc::new(w, ht, f).uav()).expect("surface")
            };
            let proxy = surface(DXGI_FORMAT_R16G16B16A16_FLOAT);
            let edited = surface(DXGI_FORMAT_R16G16B16A16_FLOAT);
            let composed = surface(DXGI_FORMAT_R16G16B16A16_FLOAT);
            let mut b = Vec::new();

            // Encode.
            frame.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            proxy.transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            flush_barriers(&h.list, &mut b);
            let encode = ComputePipeline::new(&h.gpu.device, shaders::NR_ENCODE).expect("encode");
            encode.bind(&h.list);
            set_compute_constants(
                &h.list,
                &EncodeConstants {
                    width: w,
                    height: ht,
                    hdr,
                    sdr_white: white,
                },
            );
            let srv = h.ring.table(&h.gpu.device, &[frame.srv.handle]);
            let uav = h.ring.table(&h.gpu.device, &[proxy.uav_handle()]);
            // SAFETY: both tables live in the bound heap.
            unsafe {
                h.list.SetComputeRootDescriptorTable(1, srv);
                h.list.SetComputeRootDescriptorTable(2, uav);
            }
            dispatch_2d(&h.list, w, ht);

            // The model hands back exactly what it was shown.
            proxy.transition(&mut b, D3D12_RESOURCE_STATE_COPY_SOURCE);
            edited.transition(&mut b, D3D12_RESOURCE_STATE_COPY_DEST);
            flush_barriers(&h.list, &mut b);
            // SAFETY: same size and format.
            unsafe { h.list.CopyResource(&edited.resource, &proxy.resource) };

            // Composite.
            frame.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            edited.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            composed.transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            flush_barriers(&h.list, &mut b);
            let composite =
                ComputePipeline::new(&h.gpu.device, shaders::NR_COMPOSITE).expect("composite");
            composite.bind(&h.list);
            set_compute_constants(
                &h.list,
                &CompositeConstants {
                    width: w,
                    height: ht,
                    hdr,
                    sdr_white: white,
                    detail,
                    colour: 1.0,
                    _pad: [0.0; 2],
                },
            );
            let orig = h.ring.table(&h.gpu.device, &[frame.srv.handle]);
            let edit = h.ring.table(&h.gpu.device, &[edited.srv.handle]);
            let out = h.ring.table(&h.gpu.device, &[composed.uav_handle()]);
            // SAFETY: as above.
            unsafe {
                h.list.SetComputeRootDescriptorTable(1, orig);
                h.list.SetComputeRootDescriptorTable(2, edit);
                h.list.SetComputeRootDescriptorTable(3, out);
            }
            dispatch_2d(&h.list, w, ht);

            let rb = h.readback(&composed);
            h.finish();
            let got = rb.read();
            let mut max = 0.0f32;
            for (a, b) in src.iter().zip(got.iter()) {
                max = max.max((a[0] - b[0]).abs());
            }
            let normalized = max / amplitude;
            assert!(
                normalized < 0.01,
                "hdr={hdr} detail={detail}: relative error {normalized:.4}"
            );
        }
    }

    /// A runtime without feature 18 must degrade to pass-through, not fail
    /// or crash: this drives the real NGX initialization and feature query.
    #[test]
    #[ignore = "needs a D3D12 GPU and initializes NGX"]
    fn unsupported_feature_passes_through() {
        let mut h = Harness::new();
        let src = image(64, 64, |x, y| 0.2 + 0.001 * x + 0.0005 * y);
        let frame = h.upload(64, 64, &src);
        let mut e = NgxNrEnhancer::new(
            &h.gpu,
            Vec::new(),
            std::env::temp_dir().join("captureplay-nr-test"),
            NrOptions::default(),
            NrRoute::Official,
            NrTuning::default(),
            NrPost::default(),
        )
        .expect("enhancer");
        for _ in 0..100 {
            {
                let mut ctx = h.ctx();
                e.enhance(&mut ctx, FrameInput { id: 1, tex: &frame }, true)
                    .expect("pass through");
            }
            if e.ready() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        // Either the driver offered feature 18 (ready) or it did not; both
        // must leave the working frame valid and the backend consistent.
        eprintln!("DLSS NR ready: {}", e.ready());
    }

    /// Drives a community nvngx_dlssnr.dll through captureplay_nvngx.dll:
    /// create, evaluate and composite on a real GPU. Requires the bridge DLL
    /// (built with the workspace) and a snippet; override with
    /// CAPTUREPLAY_NR_BRIDGE / CAPTUREPLAY_NR_SNIPPET.
    #[test]
    #[ignore = "needs a GPU, captureplay_nvngx.dll and a community nvngx_dlssnr.dll"]
    fn community_feature_runs() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("repository root");
        let bridge = std::env::var_os("CAPTUREPLAY_NR_BRIDGE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                root.join("target")
                    .join("debug")
                    .join("captureplay_nvngx.dll")
            });
        let snippet = std::env::var_os("CAPTUREPLAY_NR_SNIPPET")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                root.join("runtime")
                    .join("nvidia-nr")
                    .join("nvngx_dlssnr.dll")
            });
        assert!(bridge.is_file(), "missing bridge {}", bridge.display());
        assert!(snippet.is_file(), "missing snippet {}", snippet.display());

        let mut h = Harness::new();
        let (w, ht) = std::env::var("CAPTUREPLAY_NR_TEST_SIZE")
            .ok()
            .and_then(|s| {
                let (a, b) = s.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((640, 360));
        let src = image(w, ht, |x, y| 0.45 * pattern(x, y));
        let frame = h.upload(w, ht, &src);
        let mut e = NgxNrEnhancer::new(
            &h.gpu,
            Vec::new(),
            std::env::temp_dir().join("captureplay-nr-community-test"),
            NrOptions { preset: 2 },
            NrRoute::Community { bridge, snippet },
            NrTuning {
                style: 2,
                intensity: 1.1,
                skin_structure: 0.2,
                auto_mask: true,
                ..Default::default()
            },
            NrPost::default(),
        )
        .expect("enhancer");
        let mut ready = false;
        for _ in 0..400 {
            {
                let mut ctx = h.ctx();
                e.enhance(&mut ctx, FrameInput { id: 1, tex: &frame }, true)
                    .expect("enhance");
            }
            if e.ready() {
                ready = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(ready, "community feature did not become ready");
        {
            let mut ctx = h.ctx();
            e.enhance(&mut ctx, FrameInput { id: 2, tex: &frame }, false)
                .expect("enhance");
        }
        let rb = h.readback(&frame);
        h.finish();
        let got = rb.read();
        let changed = src
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| (a[0] - b[0]).abs() > 1e-4)
            .count();
        let quality = crate::testutil::psnr(&src, &got, w, ht, 0);
        let mean = |v: &[[f32; 4]]| v.iter().map(|p| p[0]).sum::<f32>() / v.len() as f32;
        let min = |v: &[[f32; 4]]| v.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max = |v: &[[f32; 4]]| v.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "community NR: {changed} of {} pixels changed, {quality:.2} dB vs input; src {:.3} [{:.3},{:.3}] -> out {:.3} [{:.3},{:.3}]",
            src.len(),
            mean(&src),
            min(&src),
            max(&src),
            mean(&got),
            min(&got),
            max(&got)
        );
        assert!(got.iter().any(|p| p[0] > 0.01), "output is black");
    }
}
