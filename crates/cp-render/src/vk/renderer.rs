//! Render orchestration on Vulkan: ingest captured frames, pace,
//! interpolate, present.
//!
//! This mirrors the D3D12 renderer (`crate::renderer`) step for step; see
//! its module docs for the flow. The differences are Vulkan mechanics:
//! per-frame command pools, the swapchain image acquired as late as
//! possible, textures retired until their frame's timeline value completed,
//! and the display expressed in scRGB-encoded nits (see the `vk` module
//! docs).

use std::sync::Arc;

use ash::vk;
use cp_core::cadence::{CadenceConfig, CadenceTracker};
use cp_core::pacing::{DelayEstimator, FrameRef, Pacer, Planned, Refresh, Slot};
use cp_core::retime::Retimer;
use cp_core::settings::RefreshMode;
use cp_core::time::HostTime;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::arena::Arena;
use super::clock::{Event, PreciseSleep};
use super::convert::ConvertPass;
use super::device::Gpu;
use super::display::{ColorManagement, DisplayTracker};
use super::egui_pass::EguiPass;
use super::output::OutputPass;
use super::pipeline::Layouts;
use super::presenter::{OutputEncoding, Presenter};
use super::texture::{Texture, WORKING_USAGE, layout_barrier};
use super::timer::GpuTimer;
use super::upload::VkUploadMemory;
use super::{Result, VkError};
use crate::common::{
    DecodeParams, EguiFrame, OutputParams, PresentResult, RenderConfig, RenderStats, UpscaleTarget,
    slack_ns,
};
use crate::present_timing::{PresentTiming, RefreshKind};
use crate::upload::UploadRing;

const FRAMES_IN_FLIGHT: usize = 2;
const WORKING_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Minimum shrink of an excess content → display offset per unique frame
/// (see the D3D12 renderer).
const OFFSET_DECAY_NS: i64 = 50_000;
/// Unique source frames kept on the GPU (pair being interpolated + spares).
const FRAME_POOL: usize = 4;
/// Per-frame arena: constants, egui geometry and texture uploads.
const ARENA_BYTES: u64 = 16 << 20;
/// Reference white a compositor assumes for Windows-scRGB content
/// (BT.2408, `wp_color_manager_v1.create_windows_scrgb`).
pub const SCRGB_REFERENCE_NITS: f32 = 203.0;

/// Interpolation backend plugged into the renderer.
pub trait Interpolator {
    fn name(&self) -> &'static str;
    /// Records everything needed to produce `out` = frame at `t` between
    /// `prev` and `next` on the frame's command buffer.
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

/// Upscaling backend: source-resolution working frame → larger working
/// frame, once per presented frame (after interpolation).
pub trait Upscaler {
    fn name(&self) -> &'static str;
    /// Supported scale factors, ascending.
    fn factors(&self) -> &'static [f32];
    /// Readies the backend for `src → dst` without blocking. Until it
    /// returns `true` the renderer keeps using output-pass scaling.
    fn prepare(&mut self, gpu: &Gpu, src: (u32, u32), dst: (u32, u32)) -> bool;
    /// Records the upscale of `src` into `out` on the frame's command buffer.
    fn upscale(&mut self, ctx: &mut RecordCtx<'_>, src: &Texture, out: &Texture) -> Result<()>;
}

/// A unique source frame handed to an interpolator.
#[derive(Debug, Clone, Copy)]
pub struct FrameInput<'a> {
    pub id: u64,
    pub tex: &'a Texture,
}

/// Per-frame recording context handed to backends.
#[allow(missing_debug_implementations)]
pub struct RecordCtx<'a> {
    pub gpu: &'a Gpu,
    pub layouts: &'a Layouts,
    pub arena: &'a mut Arena,
    pub cmd: vk::CommandBuffer,
    /// Textures replaced while recording: destroyed once this frame's work
    /// completed.
    pub retire: &'a mut Vec<Texture>,
}

/// Inspection aid: stop on an interpolated frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Freeze {
    Off,
    Pending,
    Frozen,
}

struct FrameCtx {
    device: ash::Device,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    arena: Arena,
    /// Timeline value of this context's last submission.
    value: u64,
    /// Signaled by the swapchain acquire of this context's present.
    acquired: vk::Semaphore,
    /// egui textures to free once this frame completed.
    pending_free: Vec<egui::TextureId>,
    /// Textures replaced while this frame was recorded; destroyed once it
    /// completed (later than any earlier use: the timeline is monotonic).
    retire: Vec<Texture>,
}

impl Drop for FrameCtx {
    fn drop(&mut self) {
        // SAFETY: the renderer waited for the GPU before dropping frames.
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_semaphore(self.acquired, None);
        }
    }
}

/// A converted unique frame on its way to the pacer.
#[derive(Clone, Copy)]
struct NewFrame {
    id: u64,
    info: cp_core::cadence::CadenceInfo,
    observed_ns: i64,
    signal_ns: i64,
    arrival: HostTime,
}

struct PooledFrame {
    id: Option<u64>,
    tex: Texture,
}

pub struct Renderer {
    // Field order = drop order: GPU users first, the device last.
    frames: Vec<FrameCtx>,
    timer: GpuTimer,
    upscale_timer: GpuTimer,
    frame_index: usize,
    convert: ConvertPass,
    output: OutputPass,
    egui: EguiPass,
    pool: Vec<PooledFrame>,
    interp_out: Option<Texture>,
    interpolator: Option<Box<dyn Interpolator>>,
    freeze: Freeze,
    upscaler: Option<Box<dyn Upscaler>>,
    upscaled: Option<Texture>,
    upscaled_for: Option<(Slot, (u32, u32))>,
    upscale_target: UpscaleTarget,
    /// Signaled with every wake: new captures or a finished upscale.
    ingest_event: Arc<Event>,
    shown: Option<Slot>,
    upload: Arc<UploadRing<VkUploadMemory>>,
    presenter: Presenter,
    display: DisplayTracker,
    layouts: Layouts,
    cadence: CadenceTracker,
    pacer: Pacer,
    retimer: Retimer,
    delay: DelayEstimator,
    last_content_ns: Option<i64>,
    offset_ns: Option<i64>,
    timing: PresentTiming,
    sleep: PreciseSleep,
    config: RenderConfig,
    stats: RenderStats,
    arrivals: Vec<(u64, HostTime)>,
    next_id: u64,
    /// Host time of the latest present (unscheduled presents are spaced by
    /// a refresh period, like the D3D12 waitable swapchain does).
    last_present_ns: i64,
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

fn io(e: std::io::Error) -> VkError {
    VkError::Other(e.to_string())
}

impl Renderer {
    /// `wake` is invoked from the capture thread when a frame is ready.
    pub fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        size: (u32, u32),
        config: RenderConfig,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self> {
        let debug = std::env::var_os("CAPTUREPLAY_VK_DEBUG").is_some();
        let (gpu, surface) = Gpu::new(display, window, debug)?;
        let cm = match (display, window) {
            (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w)) => {
                // SAFETY: the window (and with it the display) outlives the
                // renderer: the render thread is joined before the window
                // is dropped.
                unsafe { ColorManagement::new(d.display, w.surface) }
            }
            _ => None,
        };
        let display_tracker = DisplayTracker::new(cm);
        let presenter = Presenter::new(&gpu, surface, size.0, size.1, config.vsync)?;
        let layouts = Layouts::new(&gpu)?;
        let frames = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                // SAFETY: plain object creation on a valid device.
                unsafe {
                    let pool = gpu.device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                            .queue_family_index(gpu.queue_family),
                        None,
                    )?;
                    let cmd = gpu.device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )?[0];
                    let acquired = gpu
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                    Ok(FrameCtx {
                        device: gpu.device.clone(),
                        pool,
                        cmd,
                        arena: Arena::new(&gpu.device, &gpu.memory_properties, ARENA_BYTES)?,
                        value: 0,
                        acquired,
                        pending_free: Vec::new(),
                        retire: Vec::new(),
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let ingest_event = Arc::new(Event::new().map_err(io)?);
        let upload = {
            let ev = ingest_event.clone();
            let memory = VkUploadMemory {
                device: gpu.device.clone(),
                memory_properties: gpu.memory_properties,
                timeline: gpu.timeline,
            };
            UploadRing::new(memory, move || {
                ev.set();
                wake();
            })
        };
        Ok(Self {
            timer: GpuTimer::new(&gpu, FRAMES_IN_FLIGHT)?,
            upscale_timer: GpuTimer::new(&gpu, FRAMES_IN_FLIGHT)?,
            convert: ConvertPass::new(&gpu, &layouts)?,
            output: OutputPass::new(&gpu, &layouts, presenter.format)?,
            egui: EguiPass::new(&gpu, &layouts, presenter.format)?,
            frames,
            frame_index: 0,
            pool: Vec::new(),
            interp_out: None,
            interpolator: None,
            freeze: Freeze::Off,
            upscaler: None,
            upscaled: None,
            upscaled_for: None,
            upscale_target: UpscaleTarget::default(),
            ingest_event,
            shown: None,
            upload,
            presenter,
            display: display_tracker,
            layouts,
            cadence: CadenceTracker::new(CadenceConfig::default()),
            pacer: Pacer::new(config.pacing),
            retimer: Retimer::default(),
            delay: DelayEstimator::default(),
            last_content_ns: None,
            offset_ns: None,
            timing: PresentTiming::default(),
            sleep: PreciseSleep::new().map_err(io)?,
            config,
            stats: RenderStats::default(),
            arrivals: Vec::new(),
            next_id: 1,
            last_present_ns: 0,
            gpu,
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Pipeline layouts, for backends that build their own passes.
    pub fn layouts(&self) -> &Layouts {
        &self.layouts
    }

    pub fn upload_ring(&self) -> Arc<UploadRing<VkUploadMemory>> {
        self.upload.clone()
    }

    pub fn stats(&self) -> RenderStats {
        self.stats
    }

    pub fn presenter(&self) -> &Presenter {
        &self.presenter
    }

    pub fn display(&self) -> &crate::display_info::DisplayInfo {
        self.display.info()
    }

    /// Connector name and refresh rate of the output under the window (the
    /// window system knows these; the compositor reports the luminances).
    pub fn set_monitor(&mut self, name: String, refresh_hz: f64) {
        self.display.set_monitor(name, refresh_hz);
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

    /// The enhancement (DLSS NR) stage has no backend on Vulkan.
    pub fn enhancer_name(&self) -> Option<&'static str> {
        None
    }

    /// See [`Self::enhancer_name`]: nothing loads, so nothing is pending.
    pub fn enhancer_ready(&self) -> bool {
        true
    }

    /// Upscaled size for a presented frame of `src` size, once the backend
    /// is ready for it.
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
            (Some(dst), Some(up)) => up.prepare(&self.gpu, src, dst),
            _ => false,
        };
        target.filter(|_| ready)
    }

    fn cmd(&self) -> vk::CommandBuffer {
        self.frames[self.frame_index].cmd
    }

    /// A working-format texture created on the current command buffer.
    fn working_texture(&self, (w, h): (u32, u32)) -> Result<Texture> {
        Texture::new(&self.gpu, self.cmd(), w, h, WORKING_FORMAT, WORKING_USAGE)
    }

    /// Upscales the frame about to be presented into `upscaled`. Returns
    /// `false` (output-pass scaling) when no upscale applies or it failed.
    fn upscale_shown(&mut self, slot: Slot, src_is_interp: bool) -> Result<bool> {
        let src_size = {
            let src = if src_is_interp {
                self.interp_out.as_ref()
            } else {
                slot_texture(&self.pool, slot)
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
        if self.upscaled_for == Some((slot, dst)) {
            return Ok(true);
        }
        if self
            .upscaled
            .as_ref()
            .is_none_or(|t| (t.width, t.height) != dst)
        {
            let t = self.working_texture(dst)?;
            let old = self.upscaled.replace(t);
            self.frames[self.frame_index].retire.extend(old);
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
        let f = &mut self.frames[self.frame_index];
        self.upscale_timer.begin(f.cmd, self.frame_index);
        let mut ctx = RecordCtx {
            gpu: &self.gpu,
            layouts: &self.layouts,
            arena: &mut f.arena,
            cmd: f.cmd,
            retire: &mut f.retire,
        };
        let result = up.upscale(&mut ctx, src, out);
        self.upscale_timer.end(f.cmd, self.frame_index);
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

    /// Hands a converted working frame to the pacer.
    fn schedule_frame(&mut self, f: &NewFrame, refresh: Refresh) {
        let NewFrame {
            id,
            info,
            observed_ns,
            signal_ns,
            arrival,
        } = *f;
        self.stats.unique += 1;
        if let Some(iv) = info.content_interval_ns {
            self.stats.content_fps = 1e9 / iv as f64;
        }
        self.arrivals.push((id, arrival));
        if self.arrivals.len() > 16 {
            self.arrivals.remove(0);
        }
        let content_ns = if self.config.retime {
            self.retimer.push(observed_ns, signal_ns)
        } else {
            observed_ns
        };
        let now = HostTime::now().0;
        let ready = now;
        self.delay.push(ready - content_ns);
        let budget = self.delay.estimate().unwrap_or(ready - content_ns);
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
        for p in &mut self.pool {
            p.id = None;
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.presenter.resize(width, height);
        Ok(())
    }

    pub fn invalidate_display(&mut self) {
        // Fullscreen/monitor changes can switch VRR and composition.
        self.timing.reset();
    }

    /// Next time (ns) the render loop should wake for a scheduled slot.
    pub fn next_deadline(&self) -> Option<i64> {
        let refresh = self.refresh();
        self.pacer
            .next_target()
            .map(|t| t - self.lead_ns() - slack_ns(refresh))
    }

    /// Refresh period: as the presentation engine reports it, else from the
    /// window system's mode.
    fn nominal_period_ns(&self) -> i64 {
        self.presenter
            .refresh_info()
            .map_or_else(|| self.display.info().refresh_period_ns(), |(p, _)| p)
    }

    /// Display timing model used for scheduling.
    fn refresh(&self) -> Refresh {
        let nominal = self.nominal_period_ns();
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
        let upscale = if self.stats.upscale.is_some() {
            (self.stats.upscale_ms * 1e6) as i64
        } else {
            0
        };
        (gpu + upscale + 1_500_000).clamp(2_000_000, 14_000_000)
    }

    /// The swapchain carries scRGB and the compositor color-manages it.
    fn hdr_output(&self) -> bool {
        self.presenter.encoding == OutputEncoding::Scrgb && self.display.color_managed()
    }

    /// Real display nits → scRGB-encoded nits (see the `vk` module docs).
    fn encode_scale(&self) -> f32 {
        if self.hdr_output() {
            SCRGB_REFERENCE_NITS / self.display.info().sdr_white_nits.max(1.0)
        } else {
            1.0
        }
    }

    fn decode_params(&self) -> DecodeParams {
        let d = self.display.info();
        let k = self.encode_scale();
        let default_white = if self.hdr_output() {
            d.sdr_white_nits * k
        } else {
            80.0
        };
        DecodeParams {
            sdr_white_nits: self
                .config
                .sdr_white_override
                .map_or(default_white, |w| w * k),
            hlg_peak_nits: (self.config.peak_override.unwrap_or(d.max_nits) * k)
                .clamp(400.0, 4000.0),
        }
    }

    fn output_params(&self) -> OutputParams {
        let d = self.display.info();
        let sdr_white = self.decode_params().sdr_white_nits;
        let content_peak = match self.stats.color {
            Some(c) if c.is_hdr() => self.config.hdr_content_peak_nits,
            _ => sdr_white,
        };
        OutputParams {
            display_hdr: self.hdr_output(),
            display_peak_nits: self.config.peak_override.unwrap_or(d.max_nits)
                * self.encode_scale(),
            sdr_white_nits: sdr_white,
            content_peak_nits: content_peak,
            catmull_rom: self.config.catmull_rom,
        }
    }

    /// UI white level in output units.
    pub fn ui_white_scale(&self) -> f32 {
        if self.hdr_output() {
            self.decode_params().sdr_white_nits / cp_core::color::SCRGB_UNIT_NITS
        } else {
            1.0
        }
    }

    fn pool_texture(&mut self, size: (u32, u32)) -> Result<usize> {
        let retained: Vec<u64> = self.pacer.retained_ids().chain(self.shown_ids()).collect();
        let size_ok = |t: &Texture| (t.width, t.height) == size;
        // Drop textures of another size unless something still shows them.
        let (keep, drop): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pool)
            .into_iter()
            .partition(|p| size_ok(&p.tex) || p.id.is_some_and(|id| retained.contains(&id)));
        self.pool = keep;
        self.frames[self.frame_index]
            .retire
            .extend(drop.into_iter().map(|p| p.tex));
        if let Some(i) = self
            .pool
            .iter()
            .position(|p| size_ok(&p.tex) && p.id.is_none_or(|id| !retained.contains(&id)))
        {
            return Ok(i);
        }
        if self.pool.len() < FRAME_POOL + 2 {
            let tex = self.working_texture(size)?;
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
                let tex = self.working_texture(size)?;
                let old = std::mem::replace(&mut self.pool[i], PooledFrame { id: None, tex });
                self.frames[self.frame_index].retire.push(old.tex);
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
        let i = self.frame_index;
        self.gpu.wait(self.frames[i].value)?;
        let f = &mut self.frames[i];
        f.retire.clear();
        self.egui
            .free_textures(&std::mem::take(&mut f.pending_free));
        if let Some(ms) = self.upscale_timer.read(i) {
            let s = &mut self.stats.upscale_ms;
            *s = if *s == 0.0 { ms } else { *s * 0.9 + ms * 0.1 };
        }
        if let Some(ms) = self.timer.read(i) {
            let s = &mut self.stats.interp_gpu_ms;
            *s = if *s == 0.0 { ms } else { *s * 0.9 + ms * 0.1 };
        }
        f.arena.reset();
        // SAFETY: the GPU finished this pool's previous work (timeline wait).
        unsafe {
            self.gpu
                .device
                .reset_command_pool(f.pool, vk::CommandPoolResetFlags::empty())?;
            self.gpu.device.begin_command_buffer(
                f.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(())
    }

    /// One iteration of the render loop. `ui` is drawn if given; `force`
    /// re-presents the current content even when nothing new is due.
    pub fn step(&mut self, ui: Option<EguiFrame<'_>>, force: bool) -> Result<PresentResult> {
        if self.display.update() {
            self.timing.reset();
        }
        let nominal = self.nominal_period_ns();
        let before = self.timing.kind();
        for (id, shown) in self.presenter.poll_timing(&self.gpu) {
            self.timing.on_shown(id as u32, shown, nominal);
        }
        if let Some(ns) = self.presenter.present_to_display_ns.take() {
            let ms = ns as f64 / 1e6;
            let s = self.stats.present_to_display_ms.get_or_insert(ms);
            *s = *s * 0.9 + ms * 0.1;
        }
        if let Some((period, variable)) = self.presenter.refresh_info() {
            self.timing.set_reported(period, variable);
        }
        if self.timing.kind() != before {
            tracing::info!(kind = ?self.timing.kind(), "display refresh detected");
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
                self.finish(None)?;
                return Ok(PresentResult::Ok);
            }
            return self.compose_and_present(self.shown, ui, None);
        }

        self.ingest(refresh)?;
        self.begin_frame()?;

        let horizon = HostTime::now().0 + self.lead_ns() + slack_ns(refresh) + 500_000;
        let due: Option<Planned> = self.pacer.take_due(horizon);
        let slot = due
            .map(|p| p.slot)
            .or(if force { self.shown } else { None });

        if slot.is_some() || force {
            self.compose_and_present(slot, ui, due.map(|p| p.target_ns))
        } else {
            self.finish(None)?;
            Ok(PresentResult::Ok)
        }
    }

    /// Converts newly captured frames in a submission of their own (also
    /// while waiting to present a composed slot).
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
            let fr = &mut self.frames[self.frame_index];
            self.convert.record(
                &self.gpu,
                &self.layouts,
                &mut fr.arena,
                fr.cmd,
                &buffer,
                &f.layout,
                &f.meta.mode,
                &f.meta.color,
                params,
                &self.pool[pi].tex,
                &mut fr.retire,
            )?;
            converted_slots.push(f.slot);
            let frame = NewFrame {
                id,
                info,
                observed_ns: f.meta.device_time.0,
                signal_ns,
                arrival: f.meta.arrival,
            };
            self.schedule_frame(&frame, refresh);
        }
        self.finish(None)?;
        let value = self.frames[self.frame_index].value;
        for s in converted_slots {
            self.upload.release(s, value);
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
        let src_is_interp = match slot {
            None | Some(Slot::Real { .. }) => false,
            Some(Slot::Interp { prev, next, t }) => {
                self.timer.begin(self.cmd(), self.frame_index);
                let ok = self.run_interpolation(prev, next, t)?;
                self.timer.end(self.cmd(), self.frame_index);
                if !ok {
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

        // egui texture updates are recorded even when no image is available,
        // so they are never lost.
        if let Some(ui) = &ui {
            let f = &mut self.frames[self.frame_index];
            self.egui
                .upload(&self.gpu, &mut f.arena, f.cmd, ui, &mut f.retire)?;
            f.pending_free
                .extend(ui.textures_delta.free.iter().copied());
        }

        // Acquire the swapchain image as late as possible (lowest latency).
        let acquired = self.frames[self.frame_index].acquired;
        if self.presenter.acquire(&self.gpu, acquired, 100)?.is_none() {
            self.finish(None)?;
            return Ok(PresentResult::Occluded);
        }
        let Some((image, view, render_done)) = self.presenter.back_buffer() else {
            self.finish(None)?;
            return Ok(PresentResult::Occluded);
        };
        let size = self.presenter.size();
        let cmd = self.cmd();
        let d = &self.gpu.device;
        layout_barrier(
            d,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            })];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size.0,
                    height: size.1,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        // SAFETY: recording into this frame's open command buffer; the view
        // belongs to the acquired image.
        unsafe { d.cmd_begin_rendering(cmd, &rendering) };

        let params = self.output_params();
        let src = if upscaled {
            self.upscaled.as_ref()
        } else if src_is_interp {
            self.interp_out.as_ref()
        } else {
            slot.and_then(|s| slot_texture(&self.pool, s))
        };
        let f = &mut self.frames[self.frame_index];
        self.output.record(
            &self.gpu,
            &self.layouts,
            &mut f.arena,
            cmd,
            src,
            size,
            &params,
        )?;
        if let Some(ui) = &ui {
            self.egui
                .draw(&self.gpu, &self.layouts, &mut f.arena, cmd, size, ui)?;
        }
        // SAFETY: closes the rendering begun above.
        unsafe { self.gpu.device.cmd_end_rendering(cmd) };
        layout_barrier(
            &self.gpu.device,
            cmd,
            image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        self.finish(Some((acquired, render_done)))?;
        // Timed presents carry the target to the compositor; otherwise
        // present at the target from here (as the D3D12 backend does).
        // Unscheduled presents (UI repaints) are spaced by a refresh: MAILBOX
        // would accept them as fast as they come.
        let deadline = match target_ns {
            Some(t) if !self.presenter.timed() => Some(t - slack_ns(self.refresh())),
            Some(_) => None,
            None => Some(self.last_present_ns + self.nominal_period_ns()),
        };
        if let Some(deadline) = deadline {
            while self.sleep.until_or(deadline, Some(&self.ingest_event)) {
                let refresh = self.refresh();
                self.ingest(refresh)?;
            }
        }
        let r = self
            .presenter
            .present(&self.gpu, self.config.vsync, target_ns);
        self.last_present_ns = HostTime::now().0;
        if let Some(t) = target_ns {
            self.timing
                .on_present(self.presenter.last_present_id() as u32, t);
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
        let size = (self.pool[ni].tex.width, self.pool[ni].tex.height);
        if self
            .interp_out
            .as_ref()
            .is_none_or(|t| (t.width, t.height) != size)
        {
            let t = self.working_texture(size)?;
            let old = self.interp_out.replace(t);
            self.frames[self.frame_index].retire.extend(old);
        }
        let Some(mut interp) = self.interpolator.take() else {
            return Ok(false);
        };
        let hdr = self.stats.color.is_some_and(|c| c.is_hdr());
        interp.set_content(
            hdr,
            self.decode_params().sdr_white_nits / cp_core::color::SCRGB_UNIT_NITS,
        );
        let Some(out) = self.interp_out.as_ref() else {
            self.interpolator = Some(interp);
            return Ok(false);
        };
        let f = &mut self.frames[self.frame_index];
        let mut ctx = RecordCtx {
            gpu: &self.gpu,
            layouts: &self.layouts,
            arena: &mut f.arena,
            cmd: f.cmd,
            retire: &mut f.retire,
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

    /// Ends and submits the frame's command buffer; with `present`, waits
    /// for the acquired image and signals its render-done semaphore.
    fn finish(&mut self, present: Option<(vk::Semaphore, vk::Semaphore)>) -> Result<()> {
        let cmd = self.cmd();
        // SAFETY: the buffer is open and fully recorded.
        unsafe { self.gpu.device.end_command_buffer(cmd)? };
        let value = match present {
            Some((acquired, done)) => self.gpu.submit(
                cmd,
                &[(acquired, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)],
                &[done],
            )?,
            None => self.gpu.submit(cmd, &[], &[])?,
        };
        self.frames[self.frame_index].value = value;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let _ = self.gpu.wait_idle();
        self.upscaler = None;
        self.interpolator = None;
        self.presenter.destroy(&self.gpu);
    }
}

/// Pool texture holding a real slot's frame.
fn slot_texture(pool: &[PooledFrame], slot: Slot) -> Option<&Texture> {
    match slot {
        Slot::Real { id } => pool.iter().find(|p| p.id == Some(id)).map(|p| &p.tex),
        Slot::Interp { .. } => None,
    }
}
