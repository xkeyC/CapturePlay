//! Render orchestration: ingest captured frames, pace, interpolate, present.
//!
//! One render thread owns a [`Renderer`]. Each [`Renderer::step`]:
//! 1. converts every newly captured *unique* frame (duplicates from 30-in-60
//!    cadences are released without touching the GPU),
//! 2. asks the [`Pacer`] what is due,
//! 3. builds that slot (real frame or interpolation),
//! 4. waits on the swapchain's waitable object as late as possible, draws
//!    output + UI and presents.

use std::rc::Rc;
use std::sync::Arc;

use cp_core::VideoMode;
use cp_core::cadence::{CadenceConfig, CadenceTracker};
use cp_core::pacing::{FrameRef, Pacer, PacingConfig, Planned, Slot};
use cp_core::time::HostTime;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;
use windows::core::Result;

use crate::arena::UploadArena;
use crate::convert::{ConvertPass, DecodeParams};
use crate::descriptors::{GpuDescriptorRing, StagingHeap};
use crate::egui_pass::{EguiFrame, EguiPass};
use crate::gpu::{Gpu, transition};
use crate::output::{OutputParams, OutputPass};
use crate::presenter::{PresentResult, Presenter};
use crate::texture::{Texture, TextureDesc};
use crate::upload::UploadRing;

const FRAMES_IN_FLIGHT: usize = 2;
const WORKING_FORMAT: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT =
    DXGI_FORMAT_R16G16B16A16_FLOAT;
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
/// No implementation ships yet: this is where an officially published SDK
/// (e.g. a public DLSS neural rendering SDK) plugs in.
pub trait Enhancer {
    fn name(&self) -> &'static str;
    /// `reset` is true for the first frame after a source change or a
    /// discontinuity (temporal history must be discarded).
    fn enhance(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        frame: FrameInput<'_>,
        reset: bool,
    ) -> Result<()>;
}

/// Upscaling backend: source-resolution working frame → larger working
/// frame, once per unique frame (before interpolation, which then runs at
/// the upscaled size). When none is active, the output pass scales.
pub trait Upscaler {
    fn name(&self) -> &'static str;
    /// Supported scale factors, ascending.
    fn factors(&self) -> &'static [f32];
    /// Duration of the last upscale in ms.
    fn last_ms(&self) -> f64;
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
    /// Active upscaling `(source, target)` size.
    pub upscale: Option<((u32, u32), (u32, u32))>,
    /// Time of the last upscale (ms).
    pub upscale_ms: f64,
    pub multiplier: u32,
}

pub struct Renderer {
    // Field order = drop order: GPU users first, the device last.
    frames: Vec<FrameCtx>,
    timer: crate::gpu_timer::GpuTimer,
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
    /// Source-resolution frame fed to the upscaler.
    upscale_src: Option<Texture>,
    upscale_target: UpscaleTarget,
    /// Working size actually used for the last frame.
    upscale_effective: Option<(u32, u32)>,
    /// Next enhanced frame must reset temporal history.
    enhance_reset: bool,
    shown: Option<Slot>,
    upload: Arc<UploadRing>,
    presenter: Presenter,
    ring: GpuDescriptorRing,
    heap: Rc<StagingHeap>,
    cadence: CadenceTracker,
    pacer: Pacer,
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
        let upload = UploadRing::new(gpu.device.clone(), gpu.fence.clone(), wake);
        Ok(Self {
            timer: crate::gpu_timer::GpuTimer::new(&gpu, FRAMES_IN_FLIGHT)?,
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
            upscale_src: None,
            upscale_target: UpscaleTarget::default(),
            upscale_effective: None,
            enhance_reset: true,
            shown: None,
            upload,
            presenter,
            ring,
            heap,
            cadence: CadenceTracker::new(CadenceConfig::default()),
            pacer: Pacer::new(config.pacing),
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
        self.upscale_target = UpscaleTarget::default();
        self.stats.upscale = None;
    }

    pub fn upscaler_name(&self) -> Option<&'static str> {
        self.upscaler.as_ref().map(|u| u.name())
    }

    /// Working (possibly upscaled) size for a source frame.
    fn working_size(&mut self, src: (u32, u32)) -> Option<(u32, u32)> {
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
        let effective = target.filter(|_| ready);
        if effective != self.upscale_effective {
            // Never interpolate between frames of different sizes.
            self.pacer.reset();
            self.upscale_effective = effective;
        }
        effective
    }

    pub fn enhancer_name(&self) -> Option<&'static str> {
        self.enhancer.as_ref().map(|e| e.name())
    }

    /// Upscales `upscale_src` into pool texture `pi`. On failure the backend
    /// is dropped and `false` returned.
    fn run_upscaler(&mut self, pi: usize) -> bool {
        let (Some(mut up), Some(src)) = (self.upscaler.take(), self.upscale_src.as_ref()) else {
            return false;
        };
        let f = &self.frames[self.frame_index];
        let mut ctx = RecordCtx {
            gpu: &self.gpu,
            heap: &self.heap,
            ring: &mut self.ring,
            list: &self.list,
            allocator: &f.allocator,
        };
        match up.upscale(&mut ctx, src, &self.pool[pi].tex) {
            Ok(()) => {
                let ms = up.last_ms();
                let s = &mut self.stats.upscale_ms;
                *s = if *s == 0.0 { ms } else { *s * 0.9 + ms * 0.1 };
                self.upscaler = Some(up);
                true
            }
            Err(e) => {
                tracing::error!("upscaler {} failed, disabling it: {e}", up.name());
                self.upscale_target = UpscaleTarget::default();
                self.stats.upscale = None;
                false
            }
        }
    }

    fn run_enhancer(&mut self, pool_index: usize, id: u64, reset: bool) {
        let Some(mut e) = self.enhancer.take() else {
            return;
        };
        let f = &self.frames[self.frame_index];
        let mut ctx = RecordCtx {
            gpu: &self.gpu,
            heap: &self.heap,
            ring: &mut self.ring,
            list: &self.list,
            allocator: &f.allocator,
        };
        match e.enhance(
            &mut ctx,
            FrameInput {
                id,
                tex: &self.pool[pool_index].tex,
            },
            reset,
        ) {
            Ok(()) => self.enhancer = Some(e),
            Err(err) => tracing::error!("enhancer {} failed, disabling it: {err}", e.name()),
        }
    }

    /// Forget all frames (source changed).
    pub fn reset_source(&mut self) {
        self.cadence.reset();
        self.pacer.reset();
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
    }

    /// Next time (ns) the render loop should wake for a scheduled slot.
    pub fn next_deadline(&self) -> Option<i64> {
        self.pacer.next_target()
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
        }
        self.begin_frame()?;
        let now = HostTime::now();
        let period = self.presenter.display.info().refresh_period_ns();

        // 1. Ingest (frozen: drop new frames, keep the picture).
        let mut converted_slots = Vec::new();
        if self.freeze == Freeze::Frozen {
            for f in self.upload.take_ready() {
                self.upload.release(f.slot, 0);
            }
            if !force {
                self.finish()?;
                return Ok(PresentResult::Ok);
            }
            return self.compose_and_present(self.shown, ui);
        }
        for f in self.upload.take_ready() {
            self.stats.captured += 1;
            self.stats.mode = Some(f.meta.mode);
            self.stats.color = Some(f.meta.color);
            let signal_iv = f.meta.mode.fps.frame_duration_hns() * 100;
            let info = self
                .cadence
                .push(f.meta.signature.clone(), f.meta.device_time.0, signal_iv);
            if !info.unique {
                self.upload.release(f.slot, 0);
                continue;
            }
            let Some(buffer) = self.upload.buffer(f.slot) else {
                self.upload.release(f.slot, 0);
                continue;
            };
            let src = (f.meta.mode.width, f.meta.mode.height);
            let target = self.working_size(src);
            let pi = self.pool_texture(target.unwrap_or(src))?;
            let id = self.next_id;
            self.next_id += 1;
            self.pool[pi].id = Some(id);
            let params = self.decode_params();
            if target.is_some()
                && self
                    .upscale_src
                    .as_ref()
                    .is_none_or(|t| (t.width, t.height) != src)
            {
                self.upscale_src = Some(Texture::new(
                    &self.gpu,
                    &self.heap,
                    TextureDesc::new(src.0, src.1, WORKING_FORMAT).uav(),
                )?);
            }
            let convert_dst = match (target, self.upscale_src.as_ref()) {
                (Some(_), Some(t)) => t,
                _ => &self.pool[pi].tex,
            };
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
                convert_dst,
            )?;
            converted_slots.push(f.slot);
            if let Some(dst) = target {
                if !self.run_upscaler(pi) {
                    // Frame is lost; later frames take the output-pass path.
                    self.pool[pi].id = None;
                    continue;
                }
                self.stats.upscale = Some((src, dst));
            } else {
                self.stats.upscale = None;
            }
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
            self.arrivals.push((id, f.meta.arrival));
            if self.arrivals.len() > 16 {
                self.arrivals.remove(0);
            }
            self.pacer.on_unique_frame(
                FrameRef {
                    id,
                    time_ns: f.meta.device_time.0,
                },
                now.0,
                period,
                info.content_interval_ns,
            );
            if let Some(iv) = info.interval_ns {
                self.stats.multiplier = self.pacer.multiplier(iv, period);
            }
        }

        // 2. What is due? Present slightly ahead of the target: the
        // waitable below aligns us to the next vblank anyway.
        let due: Option<Planned> = self.pacer.take_due(now.0 + period / 2);
        let slot = due
            .map(|p| p.slot)
            .or(if force { self.shown } else { None });

        let result = if slot.is_some() || force {
            self.compose_and_present(slot, ui)?
        } else {
            // Nothing to show: still submit conversions.
            self.finish()?;
            PresentResult::Ok
        };
        let fence = self.frames[self.frame_index].fence;
        for s in converted_slots {
            self.upload.release(s, fence);
        }
        Ok(result)
    }

    /// Builds `slot` (or black when `None`), draws the UI and presents.
    fn compose_and_present(
        &mut self,
        slot: Option<Slot>,
        ui: Option<EguiFrame<'_>>,
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
                    return self.compose_and_present(Some(Slot::Real { id: next }), ui);
                }
                true
            }
        };
        if let Some(Slot::Real { id }) = slot
            && let Some(&(_, arrival)) = self.arrivals.iter().find(|(i, _)| *i == id)
        {
            self.stats.latency_ms = (HostTime::now() - arrival) as f64 / 1e6;
        }
        if slot.is_some() {
            self.shown = slot;
        }
        if src_is_interp && self.freeze == Freeze::Pending {
            self.freeze = Freeze::Frozen;
        }

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
        let src = if src_is_interp {
            self.interp_out.as_ref()
        } else {
            match slot {
                Some(Slot::Real { id }) => {
                    self.pool.iter().find(|p| p.id == Some(id)).map(|p| &p.tex)
                }
                _ => None,
            }
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
        let r = self.presenter.present(self.config.vsync);
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
    }
}
