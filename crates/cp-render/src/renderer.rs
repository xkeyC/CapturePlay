//! Render orchestration: ingest captured frames, pace, interpolate, present.
//!
//! One render thread owns a [`Renderer`]. Each [`Renderer::step`]:
//! 1. converts every newly captured *unique* frame (duplicates from 30-in-60
//!    cadences are released without touching the GPU),
//! 2. asks the [`Pacer`] what is due,
//! 3. builds that slot (real frame or interpolation) at the source
//!    resolution, then upscales exactly that frame when an upscaler is active
//!    (after interpolation: upscaling never sits on the ingest path, and
//!    interpolation runs on the smaller frames),
//! 4. waits on the swapchain's waitable object as late as possible, draws
//!    output + UI and presents — on a variable-refresh display exactly at the
//!    slot's target time, on a fixed-refresh one just before its vblank.

use std::rc::Rc;
use std::sync::Arc;

use cp_core::VideoMode;
use cp_core::cadence::{CadenceConfig, CadenceTracker};
use cp_core::pacing::{DelayEstimator, FrameRef, Pacer, PacingConfig, Planned, Refresh, Slot};
use cp_core::retime::Retimer;
use cp_core::settings::RefreshMode;
use cp_core::time::HostTime;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;
use windows::core::Result;

use crate::arena::UploadArena;
use crate::clock::PreciseSleep;
use crate::convert::{ConvertPass, DecodeParams};
use crate::descriptors::{GpuDescriptorRing, StagingHeap};
use crate::egui_pass::{EguiFrame, EguiPass};
use crate::gpu::{Gpu, transition};
use crate::output::{OutputParams, OutputPass};
use crate::present_timing::{PresentTiming, RefreshKind};
use crate::presenter::{PresentResult, Presenter};
use crate::texture::{Texture, TextureDesc};
use crate::upload::UploadRing;

const FRAMES_IN_FLIGHT: usize = 2;
const WORKING_FORMAT: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT =
    DXGI_FORMAT_R16G16B16A16_FLOAT;
/// Minimum shrink of an excess content → display offset per unique frame
/// (the excess otherwise decays by 1/32 per frame: transients such as a
/// model load clear within a second or two, small wobbles stay smooth).
const OFFSET_DECAY_NS: i64 = 50_000;
/// Unique source frames kept on the GPU (pair being interpolated + spares).
const FRAME_POOL: usize = 4;

/// User-facing renderer options (subset of the app settings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    pub vsync: bool,
    pub catmull_rom: bool,
    /// Mastering peak assumed for HDR content (nits).
    pub hdr_content_peak_nits: f32,
    pub sdr_white_override: Option<f32>,
    pub peak_override: Option<f32>,
    pub pacing: PacingConfig,
    pub refresh: RefreshMode,
    /// Reconstruct even content timing (see [`cp_core::retime`]).
    pub retime: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            vsync: true,
            catmull_rom: true,
            hdr_content_peak_nits: 1000.0,
            sdr_white_override: None,
            peak_override: None,
            pacing: PacingConfig::default(),
            refresh: RefreshMode::Auto,
            retime: true,
        }
    }
}

/// Interpolation backend plugged into the renderer.
pub trait Interpolator {
    fn name(&self) -> &'static str;
    /// Records/executes everything needed to produce `out` = frame at `t`
    /// between `prev` and `next` (unique frame ids + textures). Motion data
    /// may be cached per pair: it only changes when the ids change.
    fn interpolate(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        prev: FrameInput<'_>,
        next: FrameInput<'_>,
        t: f32,
        out: &Texture,
    ) -> Result<()>;

    /// Content description for backends that work in an encoded space:
    /// whether the source is HDR, and SDR white in scRGB units.
    fn set_content(&mut self, _hdr: bool, _sdr_white: f32) {}
}

/// Per-frame enhancement backend (extension point).
///
/// Runs once per *unique* source frame, right after color conversion and
/// before interpolation, so interpolated frames are built from enhanced
/// frames. It works in place on the linear scRGB working frame (FP16, BT.709
/// primaries, 1.0 = 80 nits) and may use `ctx.submit_and_continue` when the
/// runtime needs a fresh command list. Its GPU time adds directly to latency.
///
/// The shipped implementation is NgxNrEnhancer (NVIDIA DLSS Neural
/// Rendering through NGX); anything else can be added as another variant.
pub trait Enhancer {
    fn name(&self) -> &'static str;

    /// Whether the backend is processing frames. False while a model loads
    /// or after it failed to load: frames pass through unchanged.
    fn ready(&self) -> bool {
        true
    }

    /// Content description for backends that work in a display-referred
    /// space: whether the source is HDR, and SDR white in scRGB units.
    fn set_content(&mut self, _hdr: bool, _sdr_white: f32) {}

    /// Residual gain; changing it must not reload a model.
    fn set_strength(&mut self, _strength: f32) {}

    /// Live DLSS NR model tuning; ignored by other enhancers.
    fn set_nr_tuning(&mut self, _tuning: cp_nvidia::ngx::NrTuning) {}

    /// `reset` is true for the first frame after a source change or a
    /// discontinuity (temporal history must be discarded).
    fn enhance(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        frame: FrameInput<'_>,
        reset: bool,
    ) -> Result<()>;
}

/// Wakes the render thread (from capture or worker threads).
pub type Wake = Arc<dyn Fn() + Send + Sync>;

/// Upscaling backend: source-resolution working frame → larger working
/// frame, once per presented frame (after interpolation). When none is
/// active, the output pass scales.
pub trait Upscaler {
    fn name(&self) -> &'static str;
    /// Supported scale factors, ascending.
    fn factors(&self) -> &'static [f32];

    /// Readies the backend for `src → dst` without blocking (model loads
    /// happen in the background). Until it returns `true` the renderer keeps
    /// using output-pass scaling.
    fn prepare(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        src: (u32, u32),
        dst: (u32, u32),
    ) -> bool;
    /// Records the upscale of `src` (source size) into `out` (target size)
    /// on the frame's command list; GPU-ordered with the rest of the frame.
    fn upscale(&mut self, ctx: &mut RecordCtx<'_>, src: &Texture, out: &Texture) -> Result<()>;
}

/// Display-driven target changes (window resize) apply after this long, so
/// a window drag does not rebuild the upscaler on every step.
const UPSCALE_DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(300);

/// Debounced choice of the upscaled working size.
#[derive(Debug, Default)]
struct UpscaleTarget {
    src: Option<(u32, u32)>,
    current: Option<(u32, u32)>,
    pending: Option<(Option<(u32, u32)>, std::time::Instant)>,
}

impl UpscaleTarget {
    /// Returns the target to use now and whether it changed.
    fn update(
        &mut self,
        src: (u32, u32),
        desired: Option<(u32, u32)>,
    ) -> (Option<(u32, u32)>, bool) {
        let source_changed = self.src != Some(src);
        if desired == self.current && !source_changed {
            self.pending = None;
            return (self.current, false);
        }
        let now = std::time::Instant::now();
        let apply = source_changed
            || match self.pending {
                Some((p, since)) if p == desired => now - since >= UPSCALE_DEBOUNCE,
                _ => {
                    self.pending = Some((desired, now));
                    false
                }
            };
        if apply {
            let changed = self.current != desired;
            self.src = Some(src);
            self.current = desired;
            self.pending = None;
            (desired, changed)
        } else {
            (self.current, false)
        }
    }
}

/// A unique source frame handed to an interpolator or enhancer.
#[derive(Debug, Clone, Copy)]
pub struct FrameInput<'a> {
    pub id: u64,
    pub tex: &'a Texture,
}

/// Per-frame recording context handed to passes and interpolators.
#[allow(missing_debug_implementations)]
pub struct RecordCtx<'a> {
    pub gpu: &'a Gpu,
    pub heap: &'a Rc<StagingHeap>,
    pub ring: &'a mut GpuDescriptorRing,
    pub list: &'a ID3D12GraphicsCommandList,
    pub(crate) allocator: &'a ID3D12CommandAllocator,
}

impl RecordCtx<'_> {
    /// Submits what has been recorded so far and reopens the list. Returns
    /// the fence value signaled after the submitted work, so external
    /// engines (NvOF) can wait on it.
    pub fn submit_and_continue(&mut self) -> Result<u64> {
        // SAFETY: the list is open and owned by this frame; reopening with the
        // same allocator while the previous recording executes is allowed.
        unsafe {
            self.list.Close()?;
            self.gpu
                .queue
                .ExecuteCommandLists(&[Some(self.list.cast()?)]);
            let v = self.gpu.signal()?;
            self.list.Reset(self.allocator, None)?;
            self.list
                .SetDescriptorHeaps(&[Some(self.ring.heap().clone())]);
            Ok(v)
        }
    }

    /// Makes the direct queue wait for `fence >= value` before running
    /// subsequently submitted work.
    pub fn queue_wait(&self, fence: &ID3D12Fence, value: u64) -> Result<()> {
        // SAFETY: valid queue and fence.
        unsafe { self.gpu.queue.Wait(fence, value) }
    }
}

use windows::core::Interface;

/// Inspection aid: stop on an interpolated frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Freeze {
    Off,
    /// Freeze as soon as an interpolated frame was presented.
    Pending,
    Frozen,
}

struct FrameCtx {
    allocator: ID3D12CommandAllocator,
    arena: UploadArena,
    fence: u64,
    /// egui textures to free once this frame completed.
    pending_free: Vec<egui::TextureId>,
}

/// A converted unique frame on its way to the pacer.
#[derive(Clone, Copy)]
struct NewFrame {
    id: u64,
    pi: usize,
    info: cp_core::cadence::CadenceInfo,
    /// Capture time (host clock).
    observed_ns: i64,
    signal_ns: i64,
    arrival: HostTime,
}

struct PooledFrame {
    id: Option<u64>,
    tex: Texture,
}

/// Counters for the overlay.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub captured: u64,
    pub unique: u64,
    pub presented: u64,
    pub interpolated: u64,
    pub content_fps: f64,
    pub mode: Option<VideoMode>,
    pub color: Option<cp_core::ColorDesc>,
    /// Capture arrival → present call, last frame (ms).
    pub latency_ms: f64,
    /// GPU time to produce one interpolated frame (ms, smoothed), including
    /// optical-flow / CUDA work the queue waited for.
    pub interp_gpu_ms: f64,
    /// GPU time of the enhancement stage per unique frame (ms, smoothed).
    pub enhance_gpu_ms: f64,
    /// Active upscaling `(source, target)` size.
    pub upscale: Option<((u32, u32), (u32, u32))>,
    /// Time of the last upscale (ms).
    pub upscale_ms: f64,
    pub multiplier: u32,
    /// Spread of actual vs. scheduled display times (ms, p95).
    pub present_jitter_ms: Option<f64>,
    /// Detected display behaviour.
    pub refresh_kind: RefreshKind,
    /// The schedule assumes a variable-refresh display.
    pub vrr_timing: bool,
}

pub struct Renderer {
    // Field order = drop order: GPU users first, the device last.
    frames: Vec<FrameCtx>,
    timer: crate::gpu_timer::GpuTimer,
    enhance_timer: crate::gpu_timer::GpuTimer,
    upscale_timer: crate::gpu_timer::GpuTimer,
    frame_index: usize,
    list: ID3D12GraphicsCommandList,
    convert: ConvertPass,
    output: OutputPass,
    egui: EguiPass,
    pool: Vec<PooledFrame>,
    interp_out: Option<Texture>,
    interpolator: Option<Box<dyn Interpolator>>,
    freeze: Freeze,
    enhancer: Option<Box<dyn Enhancer>>,
    upscaler: Option<Box<dyn Upscaler>>,
    /// Upscaled copy of the frame being presented, and which frame it is.
    upscaled: Option<Texture>,
    upscaled_for: Option<(Slot, (u32, u32))>,
    upscale_target: UpscaleTarget,
    /// Signaled with every wake: new captures or a finished upscale.
    ingest_event: windows::Win32::Foundation::HANDLE,
    /// Next enhanced frame must reset temporal history.
    enhance_reset: bool,
    shown: Option<Slot>,
    upload: Arc<UploadRing>,
    presenter: Presenter,
    ring: GpuDescriptorRing,
    heap: Rc<StagingHeap>,
    cadence: CadenceTracker,
    pacer: Pacer,
    retimer: Retimer,
    /// Capture time → ready-to-show budget.
    delay: DelayEstimator,
    /// Content time of the previous unique frame.
    last_content_ns: Option<i64>,
    /// Smoothed content → display time offset.
    offset_ns: Option<i64>,
    timing: PresentTiming,
    sleep: PreciseSleep,
    config: RenderConfig,
    stats: RenderStats,
    /// Arrival time of each pooled frame id, for latency stats.
    arrivals: Vec<(u64, HostTime)>,
    next_id: u64,
    gpu: Gpu,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("gpu", &self.gpu)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl Renderer {
    /// `wake` is invoked from the capture thread when a frame is ready.
    pub fn new(
        hwnd: HWND,
        size: (u32, u32),
        config: RenderConfig,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self> {
        let debug = std::env::var_os("CAPTUREPLAY_D3D_DEBUG").is_some();
        let gpu = Gpu::new(debug)?;
        let heap = Rc::new(StagingHeap::new(
            &gpu.device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            1024,
        )?);
        let ring = GpuDescriptorRing::new(&gpu.device, FRAMES_IN_FLIGHT as u32, 1024)?;
        let frames = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                // SAFETY: valid device.
                let allocator = unsafe {
                    gpu.device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?
                };
                Ok(FrameCtx {
                    allocator,
                    arena: UploadArena::new(&gpu, 16 << 20)?,
                    fence: 0,
                    pending_free: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: valid device and allocator; the list starts closed.
        let list: ID3D12GraphicsCommandList = unsafe {
            let l: ID3D12GraphicsCommandList = gpu.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &frames[0].allocator,
                None,
            )?;
            l.Close()?;
            l
        };
        let presenter = Presenter::new(&gpu, hwnd, size.0, size.1)?;
        // SAFETY: anonymous auto-reset event, closed on drop.
        let ingest_event =
            unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None)? };
        let wake: Wake = {
            let ev = ingest_event.0 as usize;
            Arc::new(move || {
                // SAFETY: the event outlives every wake source (the renderer
                // joins workers and stops capture before dropping it).
                unsafe {
                    let _ = windows::Win32::System::Threading::SetEvent(
                        windows::Win32::Foundation::HANDLE(ev as *mut _),
                    );
                }
                wake();
            })
        };
        let upload = {
            let w = wake.clone();
            UploadRing::new(gpu.device.clone(), gpu.fence.clone(), move || w())
        };
        Ok(Self {
            timer: crate::gpu_timer::GpuTimer::new(&gpu, FRAMES_IN_FLIGHT)?,
            enhance_timer: crate::gpu_timer::GpuTimer::new(&gpu, FRAMES_IN_FLIGHT)?,
            upscale_timer: crate::gpu_timer::GpuTimer::new(&gpu, FRAMES_IN_FLIGHT)?,
            convert: ConvertPass::new(&gpu)?,
            output: OutputPass::new(&gpu)?,
            egui: EguiPass::new(&gpu)?,
            frames,
            frame_index: 0,
            list,
            pool: Vec::new(),
            interp_out: None,
            interpolator: None,
            freeze: Freeze::Off,
            enhancer: None,
            upscaler: None,
            upscaled: None,
            upscaled_for: None,
            upscale_target: UpscaleTarget::default(),
            ingest_event,
            enhance_reset: true,
            shown: None,
            upload,
            presenter,
            ring,
            heap,
            cadence: CadenceTracker::new(CadenceConfig::default()),
            pacer: Pacer::new(config.pacing),
            retimer: Retimer::default(),
            delay: DelayEstimator::default(),
            last_content_ns: None,
            offset_ns: None,
            timing: PresentTiming::default(),
            sleep: PreciseSleep::new()?,
            config,
            stats: RenderStats::default(),
            arrivals: Vec::new(),
            next_id: 1,
            gpu,
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn upload_ring(&self) -> Arc<UploadRing> {
        self.upload.clone()
    }

    pub fn stats(&self) -> RenderStats {
        self.stats
    }

    pub fn presenter(&self) -> &Presenter {
        &self.presenter
    }

    pub fn display(&self) -> &crate::display::DisplayInfo {
        self.presenter.display.info()
    }

    pub fn set_config(&mut self, config: RenderConfig) {
        if config.pacing != self.config.pacing {
            self.pacer.set_config(config.pacing);
        }
        if config.refresh != self.config.refresh {
            self.timing.reset();
            self.pacer.reset();
        }
        if config.retime != self.config.retime {
            self.reset_timeline();
        }
        self.config = config;
    }

    pub fn set_interpolator(&mut self, interp: Option<Box<dyn Interpolator>>) {
        // The old backend's resources may still be referenced by in-flight work.
        let _ = self.gpu.wait_idle();
        self.interpolator = interp;
    }

    pub fn interpolator_name(&self) -> Option<&'static str> {
        self.interpolator.as_ref().map(|i| i.name())
    }

    pub fn set_enhancer(&mut self, enhancer: Option<Box<dyn Enhancer>>) {
        let _ = self.gpu.wait_idle();
        self.enhancer = enhancer;
        self.enhance_reset = true;
    }

    pub fn set_upscaler(&mut self, upscaler: Option<Box<dyn Upscaler>>) {
        let _ = self.gpu.wait_idle();
        self.upscaler = upscaler;
        self.upscaled_for = None;
        self.upscale_target = UpscaleTarget::default();
        self.stats.upscale = None;
    }

    pub fn upscaler_name(&self) -> Option<&'static str> {
        self.upscaler.as_ref().map(|u| u.name())
    }

    /// Upscaled size for a presented frame of `src` size, once the backend
    /// is ready for it (model loads happen in the background; until then the
    /// output pass scales).
    fn upscale_size(&mut self, src: (u32, u32)) -> Option<(u32, u32)> {
        let desired = self
            .upscaler
            .as_ref()
            .and_then(|u| cp_core::scale::upscale_target(src, self.presenter.size(), u.factors()));
        let (target, changed) = self.upscale_target.update(src, desired);
        if changed {
            tracing::info!(?src, ?target, "upscale target changed");
        }
        let ready = match (target, self.upscaler.as_mut()) {
            (Some(dst), Some(up)) => up.prepare(&self.gpu, &self.heap, src, dst),
            _ => false,
        };
        target.filter(|_| ready)
    }

    pub fn enhancer_name(&self) -> Option<&'static str> {
        self.enhancer.as_ref().map(|e| e.name())
    }

    /// Updates the enhancement strength without rebuilding the backend.
    pub fn set_enhance_strength(&mut self, strength: f32) {
        if let Some(e) = self.enhancer.as_mut() {
            e.set_strength(strength);
        }
    }

    /// Updates DLSS NR model tuning without rebuilding the backend.
    pub fn set_nr_tuning(&mut self, tuning: cp_nvidia::ngx::NrTuning) {
        if let Some(e) = self.enhancer.as_mut() {
            e.set_nr_tuning(tuning);
        }
    }

    /// Whether the active enhancer currently processes frames.
    pub fn enhancer_ready(&self) -> bool {
        self.enhancer.as_ref().is_none_or(|e| e.ready())
    }

    /// Upscales the frame about to be presented into `upscaled` (recorded on
    /// the frame's list). Returns `false` (output-pass scaling) when no
    /// upscale applies or it failed.
    fn upscale_shown(&mut self, slot: Slot, src_is_interp: bool) -> Result<bool> {
        let src_size = {
            let src = if src_is_interp {
                self.interp_out.as_ref()
            } else {
                self.slot_texture(slot)
            };
            match src {
                Some(t) => (t.width, t.height),
                None => return Ok(false),
            }
        };
        let Some(dst) = self.upscale_size(src_size) else {
            self.stats.upscale = None;
            return Ok(false);
        };
        // Re-presenting the same frame (UI repaint, freeze): reuse.
        if self.upscaled_for == Some((slot, dst)) {
            return Ok(true);
        }
        if self
            .upscaled
            .as_ref()
            .is_none_or(|t| (t.width, t.height) != dst)
        {
            self.upscaled = Some(Texture::new(
                &self.gpu,
                &self.heap,
                TextureDesc::new(dst.0, dst.1, WORKING_FORMAT).uav(),
            )?);
        }
        let Some(mut up) = self.upscaler.take() else {
            return Ok(false);
        };
        let src = if src_is_interp {
            self.interp_out.as_ref()
        } else {
            slot_texture(&self.pool, slot)
        };
        let (Some(src), Some(out)) = (src, self.upscaled.as_ref()) else {
            self.upscaler = Some(up);
            return Ok(false);
        };
        let f = &self.frames[self.frame_index];
        let mut ctx = RecordCtx {
            gpu: &self.gpu,
            heap: &self.heap,
            ring: &mut self.ring,
            list: &self.list,
            allocator: &f.allocator,
        };
        self.upscale_timer.begin(ctx.list, self.frame_index);
        let result = up.upscale(&mut ctx, src, out);
        self.upscale_timer.end(ctx.list, self.frame_index);
        match result {
            Ok(()) => {
                self.stats.upscale = Some((src_size, dst));
                self.upscaled_for = Some((slot, dst));
                self.upscaler = Some(up);
                Ok(true)
            }
            Err(e) => {
                tracing::error!("upscaler {} failed, disabling it: {e}", up.name());
                self.upscale_target = UpscaleTarget::default();
                self.stats.upscale = None;
                self.upscaled_for = None;
                Ok(false)
            }
        }
    }

    fn slot_texture(&self, slot: Slot) -> Option<&Texture> {
        slot_texture(&self.pool, slot)
    }

    /// Enhances a finished working frame and hands it to the pacer.
    fn schedule_frame(&mut self, f: &NewFrame, refresh: Refresh) {
        let NewFrame {
            id,
            pi,
            info,
            observed_ns,
            signal_ns,
            arrival,
        } = *f;
        // Long gaps (menus, stalls) break temporal history too.
        let gap = info
            .interval_ns
            .is_some_and(|iv| info.content_interval_ns.is_some_and(|c| iv > c * 4));
        let reset = std::mem::take(&mut self.enhance_reset) || gap;
        self.run_enhancer(pi, id, reset);
        self.stats.unique += 1;
        if let Some(iv) = info.content_interval_ns {
            self.stats.content_fps = 1e9 / iv as f64;
        }
        self.arrivals.push((id, arrival));
        if self.arrivals.len() > 16 {
            self.arrivals.remove(0);
        }
        // Schedule on the (reconstructed) capture clock: the first slot
        // appears one processing budget after the content time, so arrival
        // and processing jitter never reach the screen.
        let content_ns = if self.config.retime {
            self.retimer.push(observed_ns, signal_ns)
        } else {
            observed_ns
        };
        let now = HostTime::now().0;
        let ready = now;
        self.delay.push(ready - content_ns);
        let budget = self.delay.estimate().unwrap_or(ready - content_ns);
        // Content → display offset. Rises at once when frames need longer
        // (never show late), decays slowly: a jumpy offset would make the
        // gap between consecutive pairs uneven.
        let needed = budget + self.lead_ns();
        let offset = match self.offset_ns {
            Some(o) if o > needed => o - ((o - needed) / 32).max(OFFSET_DECAY_NS).min(o - needed),
            _ => needed,
        };
        self.offset_ns = Some(offset);
        let base = (content_ns + offset).max(now + self.lead_ns());
        self.pacer.on_unique_frame(
            FrameRef {
                id,
                time_ns: content_ns,
            },
            base,
            refresh,
            info.content_interval_ns,
        );
        if let Some(prev) = self.last_content_ns.replace(content_ns) {
            self.stats.multiplier = self.pacer.multiplier(content_ns - prev, refresh);
        }
    }

    fn run_enhancer(&mut self, pool_index: usize, id: u64, reset: bool) {
        let Some(mut e) = self.enhancer.take() else {
            return;
        };
        let hdr = self.stats.color.is_some_and(|c| c.is_hdr());
        e.set_content(
            hdr,
            self.decode_params().sdr_white_nits / cp_core::color::SCRGB_UNIT_NITS,
        );
        self.enhance_timer.begin(&self.list, self.frame_index);
        let f = &self.frames[self.frame_index];
        let mut ctx = RecordCtx {
            gpu: &self.gpu,
            heap: &self.heap,
            ring: &mut self.ring,
            list: &self.list,
            allocator: &f.allocator,
        };
        let r = e.enhance(
            &mut ctx,
            FrameInput {
                id,
                tex: &self.pool[pool_index].tex,
            },
            reset,
        );
        self.enhance_timer.end(&self.list, self.frame_index);
        match r {
            Ok(()) => self.enhancer = Some(e),
            Err(err) => tracing::error!("enhancer {} failed, disabling it: {err}", e.name()),
        }
    }

    fn reset_timeline(&mut self) {
        self.pacer.reset();
        self.retimer.reset();
        self.delay.reset();
        self.last_content_ns = None;
        self.offset_ns = None;
    }

    /// Forget all frames (source changed).
    pub fn reset_source(&mut self) {
        self.cadence.reset();
        self.reset_timeline();
        self.shown = None;
        self.enhance_reset = true;
        for p in &mut self.pool {
            p.id = None;
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.presenter.resize(&self.gpu, width, height)
    }

    pub fn invalidate_display(&mut self) {
        self.presenter.display.invalidate();
        // Fullscreen/monitor changes can switch VRR and the flip model.
        self.timing.reset();
    }

    /// Next time (ns) the render loop should wake for a scheduled slot:
    /// early enough to compose it before its present time.
    pub fn next_deadline(&self) -> Option<i64> {
        let refresh = self.refresh();
        self.pacer
            .next_target()
            .map(|t| t - self.lead_ns() - slack_ns(refresh))
    }

    /// Display timing model used for scheduling.
    fn refresh(&self) -> Refresh {
        let nominal = self.presenter.display.info().refresh_period_ns();
        let fixed = match self.config.refresh {
            RefreshMode::Auto => self.timing.kind() == RefreshKind::Fixed,
            RefreshMode::Variable => false,
            RefreshMode::Fixed => true,
        };
        match (fixed, self.timing.vblank_ns()) {
            (true, Some(phase_ns)) => Refresh::Fixed {
                period_ns: self.timing.period_ns(nominal),
                phase_ns,
            },
            _ => Refresh::Variable {
                min_period_ns: nominal,
            },
        }
    }

    /// Time from starting to compose a slot until it is ready to present.
    fn lead_ns(&self) -> i64 {
        let gpu = (self.stats.interp_gpu_ms * 1e6) as i64;
        // Enhancement runs on ingest, before a frame can be scheduled.
        let enhance = if self.enhancer.is_some() && self.stats.enhance_gpu_ms > 0.0 {
            (self.stats.enhance_gpu_ms * 1e6) as i64
        } else {
            0
        };
        // The upscale of the presented frame runs after interpolation, on
        // the same queue.
        let upscale = if self.stats.upscale.is_some() {
            (self.stats.upscale_ms * 1e6) as i64
        } else {
            0
        };
        (gpu + enhance + upscale + 1_500_000).clamp(2_000_000, 14_000_000)
    }

    fn decode_params(&self) -> DecodeParams {
        let d = self.presenter.display.info();
        DecodeParams {
            sdr_white_nits: self.config.sdr_white_override.unwrap_or(if d.hdr {
                d.sdr_white_nits
            } else {
                80.0
            }),
            hlg_peak_nits: self
                .config
                .peak_override
                .unwrap_or(d.max_nits)
                .clamp(400.0, 4000.0),
        }
    }

    fn output_params(&self) -> OutputParams {
        let d = self.presenter.display.info();
        let sdr_white = self.decode_params().sdr_white_nits;
        let content_peak = match self.stats.color {
            Some(c) if c.is_hdr() => self.config.hdr_content_peak_nits,
            _ => sdr_white,
        };
        OutputParams {
            display_hdr: d.hdr,
            display_peak_nits: self.config.peak_override.unwrap_or(d.max_nits),
            sdr_white_nits: sdr_white,
            content_peak_nits: content_peak,
            catmull_rom: self.config.catmull_rom,
        }
    }

    /// UI white level in output units.
    pub fn ui_white_scale(&self) -> f32 {
        let d = self.presenter.display.info();
        if d.hdr {
            self.decode_params().sdr_white_nits / cp_core::color::SCRGB_UNIT_NITS
        } else {
            1.0
        }
    }

    fn pool_texture(&mut self, (width, height): (u32, u32)) -> Result<usize> {
        let retained: Vec<u64> = self.pacer.retained_ids().chain(self.shown_ids()).collect();
        let size_ok = |t: &Texture| t.width == width && t.height == height;
        // Drop textures of another size unless something still shows them.
        self.pool
            .retain(|p| size_ok(&p.tex) || p.id.is_some_and(|id| retained.contains(&id)));
        if let Some(i) = self
            .pool
            .iter()
            .position(|p| size_ok(&p.tex) && p.id.is_none_or(|id| !retained.contains(&id)))
        {
            return Ok(i);
        }
        if self.pool.len() < FRAME_POOL + 2 {
            let tex = Texture::new(
                &self.gpu,
                &self.heap,
                TextureDesc::new(width, height, WORKING_FORMAT).uav(),
            )?;
            self.pool.push(PooledFrame { id: None, tex });
            return Ok(self.pool.len() - 1);
        }
        // Everything retained (should not happen): reuse the oldest frame of
        // the right size, or replace the oldest of any size.
        let oldest = self
            .pool
            .iter()
            .enumerate()
            .filter(|(_, p)| size_ok(&p.tex))
            .min_by_key(|(_, p)| p.id.unwrap_or(0))
            .map(|(i, _)| i);
        match oldest {
            Some(i) => Ok(i),
            None => {
                let i = self
                    .pool
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| p.id.unwrap_or(0))
                    .map_or(0, |(i, _)| i);
                self.pool[i] = PooledFrame {
                    id: None,
                    tex: Texture::new(
                        &self.gpu,
                        &self.heap,
                        TextureDesc::new(width, height, WORKING_FORMAT).uav(),
                    )?,
                };
                Ok(i)
            }
        }
    }

    fn shown_ids(&self) -> Vec<u64> {
        match self.shown {
            Some(Slot::Real { id }) => vec![id],
            Some(Slot::Interp { prev, next, .. }) => vec![prev, next],
            None => vec![],
        }
    }

    fn begin_frame(&mut self) -> Result<()> {
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        let f = &mut self.frames[self.frame_index];
        self.gpu.wait(f.fence)?;
        self.egui
            .free_textures(&std::mem::take(&mut f.pending_free));
        if let Some(ms) = self.upscale_timer.read(self.frame_index) {
            let s = &mut self.stats.upscale_ms;
            *s = if *s == 0.0 { ms } else { *s * 0.9 + ms * 0.1 };
        }
        if let Some(ms) = self.enhance_timer.read(self.frame_index) {
            let s = &mut self.stats.enhance_gpu_ms;
            *s = if *s == 0.0 { ms } else { *s * 0.9 + ms * 0.1 };
        }
        if let Some(ms) = self.timer.read(self.frame_index) {
            let s = &mut self.stats.interp_gpu_ms;
            *s = if *s == 0.0 { ms } else { *s * 0.9 + ms * 0.1 };
        }
        f.arena.reset();
        self.ring.begin_segment(self.frame_index as u32);
        // SAFETY: the GPU finished this allocator's previous work (fence).
        unsafe {
            f.allocator.Reset()?;
            self.list.Reset(&f.allocator, None)?;
            self.list
                .SetDescriptorHeaps(&[Some(self.ring.heap().clone())]);
        }
        Ok(())
    }

    /// One iteration of the render loop. `ui` is drawn if given; `force`
    /// re-presents the current content even when nothing new is due (UI
    /// changed, window resized).
    pub fn step(&mut self, ui: Option<EguiFrame<'_>>, force: bool) -> Result<PresentResult> {
        if self.presenter.display.update(self.presenter.hwnd()) {
            tracing::info!(display = ?self.presenter.display.info(), "output mapping updated");
            self.timing.reset();
        }
        let nominal = self.presenter.display.info().refresh_period_ns();
        if let Some(s) = self.presenter.frame_stats() {
            let before = self.timing.kind();
            self.timing.on_stats(s, nominal);
            if self.timing.kind() != before {
                tracing::info!(kind = ?self.timing.kind(), "display refresh detected");
            }
        }
        self.stats.present_jitter_ms = self.timing.jitter_ms();
        self.stats.refresh_kind = self.timing.kind();
        let refresh = self.refresh();
        self.stats.vrr_timing = matches!(refresh, Refresh::Variable { .. });

        // Frozen: drop new frames, keep the picture.
        if self.freeze == Freeze::Frozen {
            self.begin_frame()?;
            for f in self.upload.take_ready() {
                self.upload.release(f.slot, 0);
            }
            if !force {
                self.finish()?;
                return Ok(PresentResult::Ok);
            }
            return self.compose_and_present(self.shown, ui, None);
        }

        // 1. Ingest new frames / a finished upscale.
        self.ingest(refresh)?;
        self.begin_frame()?;

        // 2. What is due? Anything whose present time falls within the
        // compose lead (plus a little wake-up tolerance).
        let horizon = HostTime::now().0 + self.lead_ns() + slack_ns(refresh) + 500_000;
        let due: Option<Planned> = self.pacer.take_due(horizon);
        let slot = due
            .map(|p| p.slot)
            .or(if force { self.shown } else { None });

        if slot.is_some() || force {
            self.compose_and_present(slot, ui, due.map(|p| p.target_ns))
        } else {
            self.finish()?;
            Ok(PresentResult::Ok)
        }
    }

    /// Converts newly captured frames and collects a finished upscale in a
    /// submission of its own. Cheap (upscaling runs on the backend's
    /// worker), so it also runs while waiting to present a composed slot:
    /// a frame arriving then must not wait for that present.
    fn ingest(&mut self, refresh: Refresh) -> Result<()> {
        let ready_frames = self.upload.take_ready();
        if ready_frames.is_empty() {
            return Ok(());
        }
        self.begin_frame()?;
        let mut converted_slots = Vec::new();
        for f in ready_frames {
            self.stats.captured += 1;
            self.stats.mode = Some(f.meta.mode);
            self.stats.color = Some(f.meta.color);
            let signal_ns = f.meta.mode.fps.frame_duration_hns() * 100;
            let info = self
                .cadence
                .push(f.meta.signature.clone(), f.meta.device_time.0, signal_ns);
            if !info.unique {
                self.upload.release(f.slot, 0);
                continue;
            }
            let Some(buffer) = self.upload.buffer(f.slot) else {
                self.upload.release(f.slot, 0);
                continue;
            };
            let src = (f.meta.mode.width, f.meta.mode.height);
            let pi = self.pool_texture(src)?;
            let id = self.next_id;
            self.next_id += 1;
            self.pool[pi].id = Some(id);
            let params = self.decode_params();
            self.convert.record(
                &self.gpu,
                &self.heap,
                &mut self.ring,
                &self.list,
                &buffer,
                &f.layout,
                &f.meta.mode,
                &f.meta.color,
                params,
                &self.pool[pi].tex,
            )?;
            converted_slots.push(f.slot);
            let frame = NewFrame {
                id,
                pi,
                info,
                observed_ns: f.meta.device_time.0,
                signal_ns,
                arrival: f.meta.arrival,
            };
            self.schedule_frame(&frame, refresh);
        }
        self.finish()?;
        let fence = self.frames[self.frame_index].fence;
        for s in converted_slots {
            self.upload.release(s, fence);
        }
        Ok(())
    }

    /// Builds `slot` (or black when `None`), draws the UI and presents — at
    /// `target_ns` when the slot is scheduled, otherwise right away.
    fn compose_and_present(
        &mut self,
        slot: Option<Slot>,
        ui: Option<EguiFrame<'_>>,
        target_ns: Option<i64>,
    ) -> Result<PresentResult> {
        // Resolve the slot into a texture.
        let src_is_interp = match slot {
            None | Some(Slot::Real { .. }) => false,
            Some(Slot::Interp { prev, next, t }) => {
                self.timer.begin(&self.list, self.frame_index);
                let ok = self.run_interpolation(prev, next, t)?;
                self.timer.end(&self.list, self.frame_index);
                if !ok {
                    // No backend: fall back to the newer real frame.
                    return self.compose_and_present(Some(Slot::Real { id: next }), ui, target_ns);
                }
                true
            }
        };
        let arrival = match slot {
            Some(Slot::Real { id }) => self
                .arrivals
                .iter()
                .find(|(i, _)| *i == id)
                .map(|&(_, a)| a),
            _ => None,
        };
        if slot.is_some() {
            self.shown = slot;
        }
        if src_is_interp && self.freeze == Freeze::Pending {
            self.freeze = Freeze::Frozen;
        }
        let upscaled = match slot {
            Some(s) => self.upscale_shown(s, src_is_interp)?,
            None => false,
        };

        // Wait for the swapchain as late as possible (lowest latency).
        self.presenter.wait_for_frame(100);
        let (bb, rtv) = {
            let (r, h) = self.presenter.back_buffer();
            (r.clone(), h)
        };
        let size = self.presenter.size();
        // SAFETY: back buffer transitions within this frame's list.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                &bb,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )])
        };

        let params = self.output_params();
        let src = if upscaled {
            self.upscaled.as_ref()
        } else if src_is_interp {
            self.interp_out.as_ref()
        } else {
            slot.and_then(|s| slot_texture(&self.pool, s))
        };
        self.output.record(
            &self.gpu,
            &mut self.ring,
            &self.list,
            src,
            rtv,
            size,
            &params,
        );
        if let Some(ui) = ui {
            let f = &mut self.frames[self.frame_index];
            self.egui.record(
                &self.gpu,
                &self.heap,
                &mut self.ring,
                &mut f.arena,
                &self.list,
                rtv,
                size,
                &ui,
            )?;
            f.pending_free
                .extend(ui.textures_delta.free.iter().copied());
        }
        // SAFETY: as above.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                &bb,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            )])
        };
        self.finish()?;
        if let Some(t) = target_ns {
            let deadline = t - slack_ns(self.refresh());
            while self.sleep.until_or(deadline, Some(self.ingest_event)) {
                let refresh = self.refresh();
                self.ingest(refresh)?;
            }
        }
        let r = self.presenter.present(self.config.vsync);
        if let Some(t) = target_ns {
            self.timing
                .on_present(self.presenter.last_present_count(), t);
        }
        if let Some(a) = arrival {
            self.stats.latency_ms = (HostTime::now() - a) as f64 / 1e6;
        }
        self.stats.presented += 1;
        if src_is_interp {
            self.stats.interpolated += 1;
        }
        Ok(r)
    }

    /// Runs the interpolation backend into `interp_out`.
    fn run_interpolation(&mut self, prev: u64, next: u64, t: f32) -> Result<bool> {
        let (Some(pi), Some(ni)) = (
            self.pool.iter().position(|p| p.id == Some(prev)),
            self.pool.iter().position(|p| p.id == Some(next)),
        ) else {
            return Ok(false);
        };
        let (w, h) = (self.pool[ni].tex.width, self.pool[ni].tex.height);
        if self
            .interp_out
            .as_ref()
            .is_none_or(|t| t.width != w || t.height != h)
        {
            self.interp_out = Some(Texture::new(
                &self.gpu,
                &self.heap,
                TextureDesc::new(w, h, WORKING_FORMAT).uav(),
            )?);
        }
        let Some(mut interp) = self.interpolator.take() else {
            return Ok(false);
        };
        let hdr = self.stats.color.is_some_and(|c| c.is_hdr());
        interp.set_content(
            hdr,
            self.decode_params().sdr_white_nits / cp_core::color::SCRGB_UNIT_NITS,
        );
        let out = self
            .interp_out
            .as_ref()
            .ok_or_else(windows::core::Error::empty)?;
        let f = &self.frames[self.frame_index];
        let mut ctx = RecordCtx {
            gpu: &self.gpu,
            heap: &self.heap,
            ring: &mut self.ring,
            list: &self.list,
            allocator: &f.allocator,
        };
        let r = interp.interpolate(
            &mut ctx,
            FrameInput {
                id: prev,
                tex: &self.pool[pi].tex,
            },
            FrameInput {
                id: next,
                tex: &self.pool[ni].tex,
            },
            t,
            out,
        );
        match r {
            Ok(()) => {
                self.interpolator = Some(interp);
                Ok(true)
            }
            Err(e) => {
                tracing::error!(
                    "interpolation backend {} failed, disabling it: {e}",
                    interp.name()
                );
                Ok(false)
            }
        }
    }

    /// Freezes on the next interpolated frame (`true`) or resumes.
    pub fn set_freeze(&mut self, on: bool) {
        self.freeze = if on { Freeze::Pending } else { Freeze::Off };
    }

    pub fn frozen(&self) -> bool {
        self.freeze != Freeze::Off
    }

    /// Closes and submits the list, recording the fence for this frame.
    fn finish(&mut self) -> Result<()> {
        // SAFETY: the list is open and fully recorded.
        unsafe {
            self.list.Close()?;
            self.gpu
                .queue
                .ExecuteCommandLists(&[Some(self.list.cast()?)]);
        }
        self.frames[self.frame_index].fence = self.gpu.signal()?;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let _ = self.gpu.wait_idle();
        // Stop upscale workers before the event they signal goes away.
        self.upscaler = None;
        // SAFETY: event owned by the renderer.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.ingest_event);
        }
    }
}

/// How long before a fixed-refresh vblank to present (variable refresh:
/// exactly on time).
fn slack_ns(refresh: Refresh) -> i64 {
    match refresh {
        Refresh::Variable { .. } => 0,
        Refresh::Fixed { period_ns, .. } => (period_ns / 4).min(1_000_000),
    }
}

/// Pool texture holding a real slot's frame.
fn slot_texture(pool: &[PooledFrame], slot: Slot) -> Option<&Texture> {
    match slot {
        Slot::Real { id } => pool.iter().find(|p| p.id == Some(id)).map(|p| &p.tex),
        Slot::Interp { .. } => None,
    }
}
