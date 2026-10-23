//! Upscaling with NVIDIA RTX Video Super Resolution (Video Effects SDK).
//!
//! Per unique source frame: encode the working frame as PQ/BT.2020 RGB10A2
//! straight into the CUDA-shared input buffer (PQ measured 1–3 dB better than
//! SDR-style encodings), run VSR, decode the upscaled result into the working
//! frame of the target size.
//!
//! VSR executes on an SDK-internal CUDA stream, so a completion fence on our
//! stream is *not* ordered after it (see `upscale_bench`). Synchronization is
//! therefore CPU-side — on a worker thread, so the render thread keeps
//! presenting on time: the render thread records the encode and hands the
//! frame over; the worker waits for the encode, runs VSR, waits for CUDA and
//! wakes the render thread, which then records the decode.

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use cp_nvidia::vsr::{self, VsrQuality, VsrSession};
use windows::Win32::Foundation::{CloseHandle, E_FAIL, E_NOTIMPL};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Result;

use crate::descriptors::{CpuDescriptor, StagingHeap, raw_buffer_view};
use crate::gpu::{Gpu, transition};
use crate::interp_vfg::Constants;
use crate::renderer::{RecordCtx, Upscaler, Wake};
use crate::shaders::{self, ComputePipeline, dispatch_2d, set_compute_constants};
use crate::texture::{Texture, flush_barriers};

type Size = (u32, u32);

/// A VSR session moved between threads. CUDA/D3D12 handles are
/// process-wide; the session is only ever used by one thread at a time
/// (loader, then worker).
struct Loaded(VsrSession);
// SAFETY: see above.
unsafe impl Send for Loaded {}

type Loader = std::thread::JoinHandle<std::result::Result<Loaded, String>>;

/// One frame handed to the worker.
struct Job {
    seq: u64,
    /// Direct-queue fence value after which the encoded input is complete.
    encoded: u64,
    wake: Wake,
}

#[derive(Default)]
struct WorkerState {
    job: Option<Job>,
    done: u64,
    last_ms: f64,
    error: Option<String>,
    quit: bool,
}

struct Shared {
    state: Mutex<WorkerState>,
    cv: Condvar,
}

/// Worker thread owning the session.
struct Worker {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    fn spawn(session: Loaded, fence: ID3D12Fence) -> Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(WorkerState::default()),
            cv: Condvar::new(),
        });
        let s = shared.clone();
        let thread = std::thread::Builder::new()
            .name("vsr".into())
            .spawn(move || worker_main(&s, session, &fence))
            .map_err(|e| windows::core::Error::new(E_FAIL, e.to_string()))?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.quit = true;
        }
        self.shared.cv.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn worker_main(shared: &Shared, session: Loaded, fence: &ID3D12Fence) {
    let Loaded(mut session) = session;
    // SAFETY: anonymous auto-reset event, closed below.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        if let Ok(mut st) = shared.state.lock() {
            st.error = Some("cannot create event".into());
        }
        shared.cv.notify_all();
        return;
    };
    loop {
        let job = {
            let Ok(mut st) = shared.state.lock() else {
                break;
            };
            while st.job.is_none() && !st.quit {
                st = match shared.cv.wait(st) {
                    Ok(g) => g,
                    Err(_) => return,
                };
            }
            if st.quit {
                break;
            }
            st.job.take()
        };
        let Some(job) = job else { continue };
        // SAFETY: valid fence and event; the session is used by this thread
        // only.
        let result = unsafe {
            if fence.GetCompletedValue() < job.encoded {
                let _ = fence.SetEventOnCompletion(job.encoded, event);
                WaitForSingleObject(event, INFINITE);
            }
            let t0 = Instant::now();
            session
                .producer()
                .Signal(job.seq)
                .map_err(|e| e.to_string())
                .and_then(|()| session.run(job.seq).map_err(|e| e.to_string()))
                .map(|()| {
                    session.synchronize();
                    t0.elapsed().as_secs_f64() * 1e3
                })
        };
        if let Ok(mut st) = shared.state.lock() {
            match result {
                Ok(ms) => st.last_ms = ms,
                Err(e) => st.error = Some(e),
            }
            st.done = job.seq;
        }
        shared.cv.notify_all();
        (job.wake)();
    }
    // SAFETY: event owned here.
    unsafe {
        let _ = CloseHandle(event);
    }
}

struct State {
    src: Size,
    dst: Size,
    in_uav: CpuDescriptor,
    out_srv: CpuDescriptor,
    input: ID3D12Resource,
    output: ID3D12Resource,
    in_pitch: u32,
    out_pitch: u32,
    /// Last submitted job.
    seq: u64,
    heap: Rc<StagingHeap>,
    worker: Worker,
}

impl Drop for State {
    fn drop(&mut self) {
        self.heap.free(self.in_uav);
        self.heap.free(self.out_srv);
    }
}

pub struct VsrUpscaler {
    encode: ComputePipeline,
    decode: ComputePipeline,
    runtime_dir: PathBuf,
    quality: VsrQuality,
    state: Option<State>,
    /// Session being created for `(src, dst)` (model load takes ~2 s).
    loading: Option<(Size, Size, Loader)>,
    failed: Option<(Size, Size)>,
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
        })
    }

    fn spawn_loader(&mut self, gpu: &Gpu, src: Size, dst: Size) {
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
        src: Size,
        dst: Size,
        session: Loaded,
    ) -> Result<()> {
        let s = &session.0;
        let (in_pitch, out_pitch) = (s.input().pitch, s.output().pitch);
        let input = s.input_resource().clone();
        let output = s.output_resource().clone();
        let in_uav = raw_buffer_view(&gpu.device, heap, &input, in_pitch / 4 * src.1, true)?;
        let out_srv =
            match raw_buffer_view(&gpu.device, heap, &output, out_pitch / 4 * dst.1, false) {
                Ok(d) => d,
                Err(e) => {
                    heap.free(in_uav);
                    return Err(e);
                }
            };
        let worker = match Worker::spawn(session, gpu.fence.clone()) {
            Ok(w) => w,
            Err(e) => {
                heap.free(in_uav);
                heap.free(out_srv);
                return Err(e);
            }
        };
        self.state = Some(State {
            src,
            dst,
            in_uav,
            out_srv,
            input,
            output,
            in_pitch,
            out_pitch,
            seq: 0,
            heap: heap.clone(),
            worker,
        });
        Ok(())
    }
}

fn constants(w: u32, h: u32, pitch: u32) -> Constants {
    Constants {
        width: w,
        height: h,
        pitch,
        hdr: 1,
        ..Default::default()
    }
}

fn poisoned() -> windows::core::Error {
    windows::core::Error::new(E_FAIL, "VSR worker poisoned")
}

impl Upscaler for VsrUpscaler {
    fn name(&self) -> &'static str {
        "NVIDIA RTX VSR"
    }

    fn factors(&self) -> &'static [f32] {
        &cp_core::scale::VSR_FACTORS
    }

    fn last_ms(&self) -> f64 {
        self.state
            .as_ref()
            .and_then(|s| s.worker.shared.state.lock().ok().map(|st| st.last_ms))
            .unwrap_or(0.0)
    }

    fn prepare(&mut self, gpu: &Gpu, heap: &Rc<StagingHeap>, src: Size, dst: Size) -> bool {
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
                    Ok(session) if (ls, ld) == (src, dst) => {
                        // Joins the old worker (idle: the renderer finishes a
                        // frame before preparing the next).
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

    fn begin(&mut self, ctx: &mut RecordCtx<'_>, src: &Texture, wake: &Wake) -> Result<()> {
        let state = self
            .state
            .as_mut()
            .filter(|s| s.src == (src.width, src.height))
            .ok_or_else(|| {
                windows::core::Error::new(E_FAIL, "VSR session not prepared for this size")
            })?;
        let device = &ctx.gpu.device;
        let mut b = Vec::new();
        src.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        b.push(transition(
            &state.input,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ));
        flush_barriers(ctx.list, &mut b);
        self.encode.bind(ctx.list);
        let srv = ctx.ring.table(device, &[src.srv.handle]);
        let uav = ctx.ring.table(device, &[state.in_uav.handle]);
        set_compute_constants(ctx.list, &constants(src.width, src.height, state.in_pitch));
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, src.width, src.height);
        b.push(transition(
            &state.input,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        ));
        flush_barriers(ctx.list, &mut b);

        // Hand over: the worker waits for this submission, not us.
        let encoded = ctx.submit_and_continue()?;
        state.seq += 1;
        state
            .worker
            .shared
            .state
            .lock()
            .map_err(|_| poisoned())?
            .job = Some(Job {
            seq: state.seq,
            encoded,
            wake: wake.clone(),
        });
        state.worker.shared.cv.notify_all();
        Ok(())
    }

    fn finish(&mut self, ctx: &mut RecordCtx<'_>, out: &Texture, block: bool) -> Result<bool> {
        let state = self
            .state
            .as_mut()
            .filter(|s| s.dst == (out.width, out.height))
            .ok_or_else(|| {
                windows::core::Error::new(E_FAIL, "VSR session not prepared for this size")
            })?;
        {
            let shared = &state.worker.shared;
            let mut st = shared.state.lock().map_err(|_| poisoned())?;
            while block && st.done < state.seq && st.error.is_none() {
                st = shared.cv.wait(st).map_err(|_| poisoned())?;
            }
            if let Some(e) = st.error.take() {
                return Err(windows::core::Error::new(E_FAIL, e));
            }
            if st.done < state.seq {
                return Ok(false);
            }
        }

        let device = &ctx.gpu.device;
        let mut b = Vec::new();
        b.push(transition(
            &state.output,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        ));
        out.transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.decode.bind(ctx.list);
        let srv = ctx.ring.table(device, &[state.out_srv.handle]);
        let uav = ctx.ring.table(device, &[out.uav_handle()]);
        set_compute_constants(ctx.list, &constants(out.width, out.height, state.out_pitch));
        // SAFETY: as above.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, out.width, out.height);
        b.push(transition(
            &state.output,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        ));
        flush_barriers(ctx.list, &mut b);
        Ok(true)
    }
}
