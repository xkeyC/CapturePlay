//! The render thread: owns the renderer, the capture session and the UI.
//!
//! It sleeps on one auto-reset event that is signaled by (a) the capture
//! thread when a frame is ready, (b) the window thread when a message is
//! queued. Timeouts come from the pacer's next deadline, so the thread wakes
//! exactly when there is something to present.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use cp_capture::{CaptureConfig, CaptureEvent, CaptureSession, DeviceInfo, MfThread};
use cp_core::pacing::PacingConfig;
use cp_core::settings::{InterpolationBackend, ScalingFilter, Settings};
use cp_core::time::HostTime;
use cp_core::{ColorDesc, VideoMode};
use cp_render::{EguiFrame, PresentResult, RenderConfig, Renderer};
use crossbeam_channel::{Receiver, Sender};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::System::Threading::{
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateEventW, CreateWaitableTimerExW, INFINITE,
    SetEvent, SetWaitableTimer, TIMER_ALL_ACCESS, WaitForMultipleObjects,
};
use winit::event_loop::EventLoopProxy;

use crate::app::{ToMain, ToRender};
use crate::config;
use crate::ui::{self, Action, BackendStatus, Rates, UiView};

/// Auto-reset Win32 event used to wake the render thread, plus a
/// high-resolution waitable timer for sub-millisecond deadlines (plain
/// wait timeouts are quantized to the 15.6 ms system tick).
#[derive(Debug)]
pub struct WakeEvent {
    event: HANDLE,
    timer: HANDLE,
}
// SAFETY: event handles may be signaled/waited from any thread.
unsafe impl Send for WakeEvent {}
// SAFETY: as above.
unsafe impl Sync for WakeEvent {}

impl WakeEvent {
    fn new() -> Result<Self> {
        // SAFETY: anonymous auto-reset event and high-resolution timer.
        unsafe {
            Ok(Self {
                event: CreateEventW(None, false, false, None)?,
                timer: CreateWaitableTimerExW(
                    None,
                    None,
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_ALL_ACCESS.0,
                )?,
            })
        }
    }

    pub fn set(&self) {
        // SAFETY: valid handle.
        unsafe {
            let _ = SetEvent(self.event);
        }
    }

    /// Waits until signaled or `timeout` elapsed (sub-ms accurate).
    fn wait(&self, timeout: Duration) {
        if timeout.is_zero() {
            return;
        }
        // Relative due time in 100 ns units (negative = relative).
        let due = -((timeout.as_nanos() / 100).max(1) as i64);
        // SAFETY: valid handles; no completion routine.
        unsafe {
            if SetWaitableTimer(self.timer, &due, 0, None, None, false).is_ok() {
                WaitForMultipleObjects(&[self.event, self.timer], false, INFINITE);
            } else {
                WaitForMultipleObjects(&[self.event], false, timeout.as_millis() as u32);
            }
        }
    }
}

impl Drop for WakeEvent {
    fn drop(&mut self) {
        // SAFETY: handles owned by self.
        unsafe {
            let _ = CloseHandle(self.event);
            let _ = CloseHandle(self.timer);
        }
    }
}

#[derive(Debug)]
pub struct RenderHandle {
    tx: Sender<ToRender>,
    wake: Arc<WakeEvent>,
    thread: Option<JoinHandle<()>>,
}

impl RenderHandle {
    pub fn send(&self, msg: ToRender) {
        let _ = self.tx.send(msg);
        self.wake.set();
    }

    pub fn shutdown(mut self) {
        self.send(ToRender::Quit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub fn spawn(
    hwnd: isize,
    size: (u32, u32),
    ctx: egui::Context,
    settings: Settings,
    proxy: EventLoopProxy<ToMain>,
) -> Result<RenderHandle> {
    let (tx, rx) = crossbeam_channel::unbounded();
    let wake = Arc::new(WakeEvent::new()?);
    let wake2 = wake.clone();
    let thread = std::thread::Builder::new()
        .name("render".into())
        .spawn(move || {
            if let Err(e) = run(hwnd, size, ctx, settings, rx, wake2, proxy.clone()) {
                tracing::error!("render thread failed: {e:#}");
                let _ = proxy.send_event(ToMain::Exit);
            }
        })?;
    Ok(RenderHandle {
        tx,
        wake,
        thread: Some(thread),
    })
}

/// Where the NVIDIA Video Effects (VFG) runtime DLLs are looked up.
fn vfg_runtime_dir(s: &Settings) -> std::path::PathBuf {
    if let Some(d) = &s.interpolation.vfg_runtime_dir {
        return d.into();
    }
    if let Some(d) = std::env::var_os("CAPTUREPLAY_VFG_RUNTIME") {
        return d.into();
    }
    // `runtime/nvidia-vfg` next to the exe or in any parent (finds the
    // repository's `runtime/` from `target/release`).
    let exe = std::env::current_exe().unwrap_or_default();
    exe.ancestors()
        .skip(1)
        .map(|d| d.join("runtime").join("nvidia-vfg"))
        .find(|d| cp_nvidia::vfg::runtime_present(d))
        .unwrap_or_else(|| exe.with_file_name("runtime").join("nvidia-vfg"))
}

fn render_config(s: &Settings) -> RenderConfig {
    RenderConfig {
        vsync: s.display.vsync,
        catmull_rom: s.display.scaling == ScalingFilter::CatmullRom,
        hdr_content_peak_nits: 1000.0,
        sdr_white_override: s.display.sdr_white_nits,
        peak_override: s.display.peak_nits_override,
        pacing: PacingConfig {
            interpolate: s.interpolation.backend != InterpolationBackend::Off,
            max_multiplier: s.interpolation.max_multiplier.max(1),
            output_fps_cap: s.interpolation.output_fps_cap.map(f64::from),
            ..PacingConfig::default()
        },
    }
}

/// Converts counters into per-second rates once per second.
#[derive(Debug)]
struct RateMeter {
    at: Instant,
    last: (u64, u64, u64, u64),
    rates: Rates,
}

impl RateMeter {
    fn update(&mut self, s: &cp_render::RenderStats) {
        let dt = self.at.elapsed().as_secs_f64();
        if dt < 1.0 {
            return;
        }
        let cur = (s.captured, s.unique, s.presented, s.interpolated);
        let d = |a: u64, b: u64| (a - b) as f64 / dt;
        self.rates = Rates {
            capture_fps: d(cur.0, self.last.0),
            unique_fps: d(cur.1, self.last.1),
            present_fps: d(cur.2, self.last.2),
            interp_fps: d(cur.3, self.last.3),
        };
        self.last = cur;
        self.at = Instant::now();
        tracing::info!(
            capture = format!("{:.1}", self.rates.capture_fps),
            unique = format!("{:.1}", self.rates.unique_fps),
            present = format!("{:.1}", self.rates.present_fps),
            interp = format!("{:.1}", self.rates.interp_fps),
            content = format!("{:.1}", s.content_fps),
            latency_ms = format!("{:.1}", s.latency_ms),
            interp_gpu_ms = format!("{:.2}", s.interp_gpu_ms),
            "stats"
        );
    }
}

struct State {
    settings: Settings,
    renderer: Renderer,
    wake: Arc<WakeEvent>,
    devices: Vec<DeviceInfo>,
    capture: Option<CaptureSession>,
    capture_tx: Sender<CaptureEvent>,
    capture_rx: Receiver<CaptureEvent>,
    modes: Vec<VideoMode>,
    current_mode: Option<VideoMode>,
    color: Option<ColorDesc>,
    capture_error: Option<String>,
    retry_at: Option<Instant>,
    proxy: EventLoopProxy<ToMain>,
    interp_backend: Option<(InterpolationBackend, u32)>,
    audio: Option<cp_audio::AudioPassthrough>,
    audio_endpoint: Option<String>,
}

impl State {
    fn pick_device(&self) -> Option<DeviceInfo> {
        let wanted = self.settings.capture.device.as_ref();
        self.devices
            .iter()
            .find(|d| Some(&d.symbolic_link) == wanted)
            .or_else(|| self.devices.iter().find(|d| !d.name.contains("Virtual")))
            .cloned()
    }

    fn start_capture(&mut self) {
        self.stop_capture();
        if self.devices.is_empty() {
            self.devices = cp_capture::list_devices().unwrap_or_default();
        }
        let Some(device) = self.pick_device() else {
            self.capture_error = None;
            self.retry_at = Some(Instant::now() + Duration::from_secs(3));
            return;
        };
        tracing::info!(device = %device.name, "starting capture");
        self.start_audio(&device.name);
        let cfg = CaptureConfig {
            device,
            request: self.settings.capture.mode,
            color: self.settings.color,
            max_buffers: 2,
        };
        let tx = self.capture_tx.clone();
        let wake = self.wake.clone();
        match CaptureSession::start(cfg, self.renderer.upload_ring(), move |e| {
            let _ = tx.send(e);
            wake.set();
        }) {
            Ok(s) => {
                self.capture = Some(s);
                self.capture_error = None;
            }
            Err(e) => self.capture_error = Some(e.to_string()),
        }
    }

    fn start_audio(&mut self, video_name: &str) {
        self.stop_audio();
        if !self.settings.audio.enabled {
            return;
        }
        let inputs = cp_audio::endpoints::list(true).unwrap_or_default();
        let input = match &self.settings.audio.input {
            Some(id) => inputs.iter().find(|e| &e.id == id),
            None => cp_audio::match_capture_endpoint(video_name, &inputs),
        };
        let Some(input) = input else {
            tracing::warn!(
                video = video_name,
                "no matching audio endpoint for capture device"
            );
            return;
        };
        tracing::info!(endpoint = %input.name, "starting audio passthrough");
        self.audio_endpoint = Some(input.name.clone());
        let a = &self.settings.audio;
        match cp_audio::AudioPassthrough::start(cp_audio::AudioConfig {
            input_id: input.id.clone(),
            output_id: a.output.clone(),
            target_latency_ms: a.target_latency_ms,
            av_offset_ms: a.av_offset_ms,
            volume: a.volume,
        }) {
            Ok(p) => self.audio = Some(p),
            Err(e) => tracing::error!("audio start failed: {e}"),
        }
    }

    fn stop_audio(&mut self) {
        if let Some(a) = self.audio.take() {
            a.stop();
        }
        self.audio_endpoint = None;
    }

    fn stop_capture(&mut self) {
        if let Some(c) = self.capture.take() {
            c.stop();
        }
        self.stop_audio();
        self.renderer.reset_source();
    }

    /// (Re)creates the interpolation backend when the selection changed.
    fn apply_interpolator(&mut self) {
        let want = (
            self.settings.interpolation.backend,
            self.settings.interpolation.vfg_quality,
        );
        if self.interp_backend == Some(want) {
            return;
        }
        self.interp_backend = Some(want);
        let want = want.0;
        let interp: Option<Box<dyn cp_render::Interpolator>> = match want {
            InterpolationBackend::Off => None,
            InterpolationBackend::NvidiaVfg => {
                let dir = vfg_runtime_dir(&self.settings);
                match cp_render::VfgInterpolator::new(
                    self.renderer.gpu(),
                    dir,
                    self.settings.interpolation.vfg_quality,
                ) {
                    Ok(i) => Some(Box::new(i)),
                    Err(e) => {
                        tracing::warn!("VFG interpolation unavailable: {e}");
                        None
                    }
                }
            }
        };
        self.renderer.set_interpolator(interp);
    }

    fn apply_settings(&mut self) {
        self.apply_interpolator();
        match (&self.audio, self.settings.audio.enabled) {
            (Some(a), true) => {
                a.set_volume(self.settings.audio.volume);
                a.set_offset_ms(self.settings.audio.av_offset_ms);
            }
            (Some(_), false) => self.stop_audio(),
            (None, true) => {
                if let Some(d) = self.pick_device() {
                    self.start_audio(&d.name);
                }
            }
            (None, false) => {}
        }
        self.renderer.set_config(render_config(&self.settings));
        if let Some(c) = &self.capture {
            c.set_color_settings(self.settings.color);
        }
        config::save(&self.settings);
    }

    fn handle_capture_event(&mut self, e: CaptureEvent) {
        match e {
            CaptureEvent::Started { mode, color, modes } => {
                if self.current_mode != Some(mode) {
                    self.renderer.reset_source();
                }
                self.current_mode = Some(mode);
                self.color = Some(color);
                self.modes = modes;
                self.capture_error = None;
            }
            CaptureEvent::Stopped { error } => {
                if let Some(err) = error {
                    self.capture_error = Some(err);
                    // Device unplugged / signal lost: retry periodically.
                    self.retry_at = Some(Instant::now() + Duration::from_secs(2));
                    self.devices.clear();
                }
                self.capture = None;
            }
        }
    }

    fn handle_action(&mut self, a: Action) {
        match a {
            Action::SelectDevice(link) => {
                self.settings.capture.device = Some(link);
                self.settings.capture.mode = Default::default();
                config::save(&self.settings);
                self.start_capture();
            }
            Action::SelectMode(req) => {
                self.settings.capture.mode = req;
                config::save(&self.settings);
                self.start_capture();
            }
            Action::RestartCapture => {
                self.devices.clear();
                self.start_capture();
            }
            Action::SettingsChanged => self.apply_settings(),
            Action::SetFreeze(on) => self.renderer.set_freeze(on),
            Action::ToggleFullscreen => {
                let on = !self.settings.display.start_fullscreen;
                self.settings.display.start_fullscreen = on;
                let _ = self.proxy.send_event(ToMain::SetFullscreen(on));
            }
        }
    }
}

fn merge_input(acc: &mut Option<egui::RawInput>, new: egui::RawInput) {
    match acc {
        Some(a) => {
            let mut events = std::mem::take(&mut a.events);
            events.extend(new.events);
            *a = egui::RawInput { events, ..new };
        }
        None => *acc = Some(new),
    }
}

#[allow(clippy::too_many_lines)]
fn run(
    hwnd: isize,
    size: (u32, u32),
    ctx: egui::Context,
    settings: Settings,
    rx: Receiver<ToRender>,
    wake: Arc<WakeEvent>,
    proxy: EventLoopProxy<ToMain>,
) -> Result<()> {
    let _mf = MfThread::init().context("Media Foundation init")?;
    let hwnd = HWND(hwnd as *mut _);
    let wake_capture = wake.clone();
    let renderer = Renderer::new(hwnd, size, render_config(&settings), move || {
        wake_capture.set()
    })
    .context("renderer init")?;
    ui::install_fonts(&ctx);
    let (capture_tx, capture_rx) = crossbeam_channel::unbounded();
    let mut st = State {
        settings,
        renderer,
        wake: wake.clone(),
        devices: cp_capture::list_devices().unwrap_or_default(),
        capture: None,
        capture_tx,
        capture_rx,
        modes: Vec::new(),
        current_mode: None,
        color: None,
        capture_error: None,
        retry_at: None,
        proxy,
        interp_backend: None,
        audio: None,
        audio_endpoint: None,
    };
    st.apply_interpolator();
    st.start_capture();

    let mut overlay = true;
    let mut pending_input: Option<egui::RawInput> = None;
    // Viewport info of the last real input, reused for timer-driven repaints.
    // Time is always stamped here so it is monotonic across both sources.
    let mut input_template = egui::RawInput::default();
    let ui_clock = Instant::now();
    let mut primitives: Vec<egui::ClippedPrimitive> = Vec::new();
    let mut ppp = 1.0f32;
    // egui texture updates not yet uploaded (kept across occluded frames).
    let mut pending_delta = egui::TexturesDelta::default();
    let mut repaint_at = Instant::now();
    let mut meter = RateMeter {
        at: Instant::now(),
        last: (0, 0, 0, 0),
        rates: Rates::default(),
    };
    let mut occluded = false;

    loop {
        // Sleep until something can happen.
        let now = Instant::now();
        let mut timeout = Duration::from_millis(500);
        if overlay {
            timeout = timeout
                .min(repaint_at.saturating_duration_since(now))
                .min(Duration::from_millis(250));
        }
        if let Some(deadline) = st.renderer.next_deadline() {
            let half = st.renderer.display().refresh_period_ns() / 2;
            let ns = (deadline - half - HostTime::now().0).max(0);
            timeout = timeout.min(Duration::from_nanos(ns as u64));
        }
        if let Some(r) = st.retry_at {
            timeout = timeout.min(r.saturating_duration_since(now));
        }
        wake.wait(timeout);

        // Window thread messages.
        let mut force = false;
        for msg in rx.try_iter() {
            match msg {
                ToRender::Quit => {
                    pending_delta.clear();
                    st.stop_capture();
                    config::save(&st.settings);
                    return Ok(());
                }
                ToRender::Resized(w, h) => {
                    occluded = w == 0 || h == 0;
                    if !occluded {
                        st.renderer.resize(w, h)?;
                    }
                    force = true;
                }
                ToRender::DisplayChanged => {
                    st.renderer.invalidate_display();
                    force = true;
                }
                ToRender::Input(i) => merge_input(&mut pending_input, i),
                ToRender::ToggleFreeze => {
                    let f = st.renderer.frozen();
                    st.renderer.set_freeze(!f);
                    force = true;
                }
                ToRender::ToggleOverlay => {
                    overlay = !overlay;
                    force = true;
                }
                ToRender::FullscreenChanged(on) => {
                    st.settings.display.start_fullscreen = on;
                    st.renderer.invalidate_display();
                }
            }
        }
        for e in st.capture_rx.clone().try_iter() {
            st.handle_capture_event(e);
        }
        if st.retry_at.is_some_and(|r| Instant::now() >= r) {
            st.retry_at = None;
            if st.capture.is_none() {
                st.start_capture();
            }
        }

        // UI.
        meter.update(&st.renderer.stats());
        let run_ui = overlay && (pending_input.is_some() || Instant::now() >= repaint_at);
        if run_ui {
            let mut input = match pending_input.take() {
                Some(i) => {
                    input_template = egui::RawInput {
                        events: Vec::new(),
                        ..i.clone()
                    };
                    i
                }
                None => input_template.clone(),
            };
            input.time = Some(ui_clock.elapsed().as_secs_f64());
            let stats = st.renderer.stats();
            let display = st.renderer.display().clone();
            let adapter = st.renderer.gpu().adapter_name.clone();
            let interp = match (
                st.settings.interpolation.backend,
                st.renderer.interpolator_name(),
            ) {
                (InterpolationBackend::Off, _) => BackendStatus::Off,
                (_, Some(n)) => BackendStatus::Running(n),
                (_, None) => BackendStatus::Unavailable,
            };
            let audio = st.audio.as_ref().map(|a| {
                let sh = a.shared();
                ui::AudioView {
                    endpoint: st.audio_endpoint.clone().unwrap_or_default(),
                    rate: sh.sample_rate.load(std::sync::atomic::Ordering::Relaxed),
                    channels: sh.channels.load(std::sync::atomic::Ordering::Relaxed),
                    fill_ms: sh.fill_ms.load(std::sync::atomic::Ordering::Relaxed),
                    underruns: sh.underruns.load(std::sync::atomic::Ordering::Relaxed),
                }
            });
            let mut settings = st.settings.clone();
            let mut actions = Vec::new();
            let output = ctx.run_pass(input, |ctx| {
                let view = UiView {
                    devices: &st.devices,
                    modes: &st.modes,
                    current_mode: st.current_mode,
                    color: st.color,
                    capture_error: st.capture_error.as_deref(),
                    display: &display,
                    stats: &stats,
                    rates: &meter.rates,
                    adapter: &adapter,
                    interp,
                    frozen: st.renderer.frozen(),
                    audio: audio.as_ref(),
                };
                actions = ui::draw(ctx, &mut settings, &view);
            });
            st.settings = settings;
            for a in actions {
                st.handle_action(a);
            }
            ppp = output.pixels_per_point;
            primitives = ctx.tessellate(output.shapes, ppp);
            pending_delta.append(output.textures_delta);
            let _ = st
                .proxy
                .send_event(ToMain::Platform(Box::new(output.platform_output)));
            let delay = output
                .viewport_output
                .get(&egui::ViewportId::ROOT)
                .map_or(Duration::from_millis(250), |v| v.repaint_delay);
            repaint_at = Instant::now() + delay.min(Duration::from_millis(250));
            force = true;
        }

        if occluded {
            continue;
        }
        // Texture deltas must be uploaded even if the overlay was just hidden,
        // and only in a step that is guaranteed to present.
        let has_delta = !pending_delta.is_empty();
        let force = force || has_delta;
        let white_scale = st.renderer.ui_white_scale();
        let ui_frame = if overlay {
            Some(EguiFrame {
                primitives: &primitives,
                textures_delta: &pending_delta,
                pixels_per_point: ppp,
                white_scale,
            })
        } else if has_delta {
            Some(EguiFrame {
                primitives: &[],
                textures_delta: &pending_delta,
                pixels_per_point: ppp,
                white_scale,
            })
        } else {
            None
        };
        let result = st.renderer.step(ui_frame, force);
        pending_delta.clear();
        match result? {
            PresentResult::DeviceLost => anyhow::bail!(
                "GPU device lost (reason {:?})",
                st.renderer.gpu().removed_reason()
            ),
            PresentResult::Occluded | PresentResult::Ok => {}
        }
    }
}
