//! Upscaling with NVIDIA RTX Video Super Resolution (Video Effects SDK).
//!
//! Per unique source frame: encode the working frame as PQ/BT.2020 RGB10A2
//! straight into the CUDA-shared input buffer (PQ measured 1–3 dB better than
//! SDR-style encodings), run VSR, decode the upscaled result into the working
//! frame of the target size.
//!
//! VSR executes on an SDK-internal CUDA stream, so a completion fence on our
//! stream is *not* ordered after it (see `upscale_bench`). Synchronization is
//! therefore CPU-side: wait for the encode, run, `cuCtxSynchronize`, decode.
//! The upscaled frame is on the critical path to presentation anyway, so this
//! adds no latency beyond the upscale itself.

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use cp_nvidia::vsr::{self, VsrQuality, VsrSession};
use windows::Win32::Foundation::{E_FAIL, E_NOTIMPL};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

use crate::descriptors::{CpuDescriptor, StagingHeap, raw_buffer_view};
use crate::gpu::{Gpu, transition};
use crate::interp_vfg::Constants;
use crate::renderer::{RecordCtx, Upscaler};
use crate::shaders::{self, ComputePipeline, dispatch_2d, set_compute_constants};
use crate::texture::{Texture, flush_barriers};

struct State {
    src: (u32, u32),
    dst: (u32, u32),
    in_uav: CpuDescriptor,
    out_srv: CpuDescriptor,
    value: u64,
    heap: Rc<StagingHeap>,
    session: VsrSession,
}

impl Drop for State {
    fn drop(&mut self) {
        self.heap.free(self.in_uav);
        self.heap.free(self.out_srv);
    }
}

/// A session built on a loader thread. CUDA/D3D12 handles are process-wide,
/// so moving the finished session to the render thread is sound.
struct Loaded(VsrSession);
// SAFETY: see above; the session is used by one thread at a time.
unsafe impl Send for Loaded {}

type Size = (u32, u32);
type Loader = std::thread::JoinHandle<std::result::Result<Loaded, String>>;

pub struct VsrUpscaler {
    encode: ComputePipeline,
    decode: ComputePipeline,
    runtime_dir: PathBuf,
    quality: VsrQuality,
    state: Option<State>,
    /// Session being created for `(src, dst)` (model load takes ~2 s).
    loading: Option<(Size, Size, Loader)>,
    failed: Option<(Size, Size)>,
    last_ms: f64,
}

impl std::fmt::Debug for VsrUpscaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VsrUpscaler")
            .field("quality", &self.quality)
            .finish_non_exhaustive()
    }
}

impl VsrUpscaler {
    /// Fails fast when the runtime is missing or the GPU is not NVIDIA.
    pub fn new(gpu: &Gpu, runtime_dir: PathBuf, quality: VsrQuality) -> Result<Self> {
        if !gpu.is_nvidia() || !vsr::runtime_present(&runtime_dir) {
            return Err(windows::core::Error::new(
                E_NOTIMPL,
                format!("VSR runtime not found in {}", runtime_dir.display()),
            ));
        }
        Ok(Self {
            encode: ComputePipeline::new(&gpu.device, shaders::VFG_ENCODE)?,
            decode: ComputePipeline::new(&gpu.device, shaders::VFG_DECODE)?,
            runtime_dir,
            quality,
            state: None,
            loading: None,
            failed: None,
            last_ms: 0.0,
        })
    }

    fn spawn_loader(&mut self, gpu: &Gpu, src: (u32, u32), dst: (u32, u32)) {
        let (device, luid, dir, quality) = (
            gpu.device.clone(),
            gpu.adapter_luid,
            self.runtime_dir.clone(),
            self.quality,
        );
        let handle = std::thread::Builder::new()
            .name("vsr-load".into())
            .spawn(move || {
                VsrSession::new(&device, luid, &dir, src, dst, quality, 1.0)
                    .map(Loaded)
                    .map_err(|e| e.to_string())
            });
        match handle {
            Ok(h) => self.loading = Some((src, dst, h)),
            Err(e) => {
                tracing::error!("cannot start VSR loader: {e}");
                self.failed = Some((src, dst));
            }
        }
    }

    fn install(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        src: (u32, u32),
        dst: (u32, u32),
        session: VsrSession,
    ) -> Result<()> {
        let in_words = session.input().pitch / 4 * src.1;
        let out_words = session.output().pitch / 4 * dst.1;
        let in_uav = raw_buffer_view(&gpu.device, heap, session.input_resource(), in_words, true)?;
        let out_srv = raw_buffer_view(
            &gpu.device,
            heap,
            session.output_resource(),
            out_words,
            false,
        )?;
        self.state = Some(State {
            src,
            dst,
            in_uav,
            out_srv,
            value: 0,
            heap: heap.clone(),
            session,
        });
        Ok(())
    }
}

impl Upscaler for VsrUpscaler {
    fn name(&self) -> &'static str {
        "NVIDIA RTX VSR"
    }

    fn factors(&self) -> &'static [f32] {
        &cp_core::scale::VSR_FACTORS
    }

    fn last_ms(&self) -> f64 {
        self.last_ms
    }

    fn prepare(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        src: (u32, u32),
        dst: (u32, u32),
    ) -> bool {
        if self
            .state
            .as_ref()
            .is_some_and(|s| s.src == src && s.dst == dst)
        {
            return true;
        }
        if self.failed == Some((src, dst)) {
            return false;
        }
        match self.loading.take() {
            Some((ls, ld, h)) if !h.is_finished() => {
                // Still loading (possibly an older size): wait for it first.
                self.loading = Some((ls, ld, h));
                false
            }
            Some((ls, ld, h)) => {
                let result = h
                    .join()
                    .unwrap_or_else(|_| Err("VSR loader panicked".into()));
                match result {
                    Ok(Loaded(session)) if (ls, ld) == (src, dst) => {
                        // Old session (if any) is idle: the render thread is here.
                        self.state = None;
                        match self.install(gpu, heap, src, dst, session) {
                            Ok(()) => true,
                            Err(e) => {
                                tracing::error!("VSR install failed: {e}");
                                self.failed = Some((src, dst));
                                false
                            }
                        }
                    }
                    Ok(_) => {
                        // Finished for a size we no longer want.
                        self.spawn_loader(gpu, src, dst);
                        false
                    }
                    Err(e) => {
                        tracing::error!("VSR session failed: {e}");
                        if (ls, ld) == (src, dst) {
                            self.failed = Some((src, dst));
                        } else {
                            self.spawn_loader(gpu, src, dst);
                        }
                        false
                    }
                }
            }
            None => {
                self.spawn_loader(gpu, src, dst);
                false
            }
        }
    }

    fn upscale(&mut self, ctx: &mut RecordCtx<'_>, src: &Texture, out: &Texture) -> Result<()> {
        let state = self
            .state
            .as_mut()
            .filter(|s| s.src == (src.width, src.height) && s.dst == (out.width, out.height))
            .ok_or_else(|| {
                windows::core::Error::new(E_FAIL, "VSR session not prepared for this size")
            })?;
        let device = &ctx.gpu.device;
        let constants = |w: u32, h: u32, pitch: u32| Constants {
            width: w,
            height: h,
            pitch,
            hdr: 1,
            ..Default::default()
        };

        // 1. Encode into the shared input buffer.
        let input = state.session.input_resource();
        let mut b = Vec::new();
        src.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        b.push(transition(
            input,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ));
        flush_barriers(ctx.list, &mut b);
        self.encode.bind(ctx.list);
        let srv = ctx.ring.table(device, &[src.srv.handle]);
        let uav = ctx.ring.table(device, &[state.in_uav.handle]);
        set_compute_constants(
            ctx.list,
            &constants(src.width, src.height, state.session.input().pitch),
        );
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, src.width, src.height);
        b.push(transition(
            input,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        ));
        flush_barriers(ctx.list, &mut b);

        // 2. CPU-synchronized hand-over (see module docs).
        let ready = ctx.submit_and_continue()?;
        ctx.gpu.wait(ready)?;
        let t0 = Instant::now();
        state.value += 1;
        // SAFETY: CPU-side signal of our own shared fence.
        unsafe { state.session.producer().Signal(state.value)? };
        state
            .session
            .run(state.value)
            .map_err(|e| windows::core::Error::new(E_FAIL, e.to_string()))?;
        state.session.synchronize();
        self.last_ms = t0.elapsed().as_secs_f64() * 1e3;

        // 3. Decode the upscaled frame.
        let output = state.session.output_resource();
        b.push(transition(
            output,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        ));
        out.transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.decode.bind(ctx.list);
        let srv = ctx.ring.table(device, &[state.out_srv.handle]);
        let uav = ctx.ring.table(device, &[out.uav_handle()]);
        set_compute_constants(
            ctx.list,
            &constants(out.width, out.height, state.session.output().pitch),
        );
        // SAFETY: as above.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, out.width, out.height);
        b.push(transition(
            output,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        ));
        flush_barriers(ctx.list, &mut b);
        Ok(())
    }
}
