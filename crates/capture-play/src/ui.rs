//! Overlay UI (egui), drawn into the HDR back buffer by the renderer.
//!
//! The UI never touches engine objects directly: it edits a copy of the
//! settings and returns [`Action`]s that the render thread applies. All text
//! comes from [`crate::i18n`].

use cp_capture::DeviceInfo;
use cp_core::format::ModeRequest;
use cp_core::settings::{InterpolationBackend, Language, ScalingFilter, Settings};
use cp_core::{ColorDesc, Range, TransferFunction, VideoMode};
use cp_render::{DisplayInfo, RenderStats};

use crate::i18n::{self, Strings};

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    SelectDevice(String),
    SelectMode(ModeRequest),
    RestartCapture,
    SettingsChanged,
    ToggleFullscreen,
    SetFreeze(bool),
}

/// Engine state the UI can only report on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatus {
    Off,
    Running(&'static str),
    Unavailable,
}

/// Read-only state shown by the UI.
#[derive(Debug)]
pub struct UiView<'a> {
    pub devices: &'a [DeviceInfo],
    pub modes: &'a [VideoMode],
    pub current_mode: Option<VideoMode>,
    pub color: Option<ColorDesc>,
    pub capture_error: Option<&'a str>,
    pub display: &'a DisplayInfo,
    pub stats: &'a RenderStats,
    pub rates: &'a Rates,
    pub adapter: &'a str,
    pub interp: BackendStatus,
    pub frozen: bool,
    pub audio: Option<&'a AudioView>,
}

/// Audio passthrough status.
#[derive(Debug, Clone)]
pub struct AudioView {
    pub endpoint: String,
    pub rate: u32,
    pub channels: u32,
    pub fill_ms: u32,
    pub underruns: u64,
}

/// Per-second rates derived from counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rates {
    pub capture_fps: f64,
    pub unique_fps: f64,
    pub present_fps: f64,
    pub interp_fps: f64,
}

/// Loads Microsoft YaHei (if present) as a fallback font for CJK text.
pub fn install_fonts(ctx: &egui::Context) {
    let Ok(bytes) = std::fs::read(r"C:\Windows\Fonts\msyh.ttc") else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "yahei".into(),
        std::sync::Arc::new(egui::FontData::from_owned(bytes)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("yahei".into());
    }
    ctx.set_fonts(fonts);
}

fn transfer_label(t: &Strings, tf: Option<TransferFunction>) -> &'static str {
    match tf {
        None => t.auto,
        Some(TransferFunction::Srgb) => "SDR sRGB",
        Some(TransferFunction::Gamma22) => "SDR Gamma 2.2",
        Some(TransferFunction::Bt1886) => "SDR BT.1886",
        Some(TransferFunction::Pq) => "HDR10 (PQ)",
        Some(TransferFunction::Hlg) => "HLG",
    }
}

fn status_label(t: &Strings, s: BackendStatus) -> String {
    match s {
        BackendStatus::Off => t.status_off.to_string(),
        BackendStatus::Running(name) => format!("{}: {name}", t.status_running),
        BackendStatus::Unavailable => t.status_unavailable.to_string(),
    }
}

fn mode_key(m: &VideoMode) -> String {
    format!("{}x{} @ {} {}", m.width, m.height, m.fps, m.format)
}

#[allow(clippy::too_many_lines)]
pub fn draw(ctx: &egui::Context, s: &mut Settings, v: &UiView<'_>) -> Vec<Action> {
    let t = i18n::resolve(s.ui.language);
    let mut actions = Vec::new();
    let mut changed = false;

    egui::Window::new("CapturePlay")
        .default_pos([16.0, 16.0])
        .default_width(360.0)
        .resizable(false)
        .frame(egui::Frame::window(&ctx.global_style()).fill(egui::Color32::from_black_alpha(245)))
        .show(ctx, |ui| {
            ui.small(t.hint);
            ui.separator();

            egui::CollapsingHeader::new(t.source).default_open(true).show(ui, |ui| {
                let cur_dev = s.capture.device.clone();
                let dev_name = v.devices.iter().find(|d| Some(&d.symbolic_link) == cur_dev.as_ref()).map_or(t.device_auto, |d| d.name.as_str());
                egui::ComboBox::from_label(t.device).selected_text(dev_name).width(230.0).show_ui(ui, |ui| {
                    for d in v.devices {
                        if ui.selectable_label(Some(&d.symbolic_link) == cur_dev.as_ref(), &d.name).clicked() {
                            actions.push(Action::SelectDevice(d.symbolic_link.clone()));
                        }
                    }
                });
                let cur = v.current_mode.map_or_else(|| "—".to_string(), |m| mode_key(&m));
                egui::ComboBox::from_label(t.format).selected_text(cur).width(230.0).height(400.0).show_ui(ui, |ui| {
                    let mut modes: Vec<&VideoMode> = v.modes.iter().filter(|m| m.height >= 720).collect();
                    modes.sort_by_key(|m| std::cmp::Reverse((m.height, m.width, m.fps.as_f64() as u32)));
                    for m in modes {
                        if ui.selectable_label(Some(*m) == v.current_mode, mode_key(m)).clicked() {
                            actions.push(Action::SelectMode(ModeRequest {
                                width: Some(m.width),
                                height: Some(m.height),
                                fps: Some(m.fps.as_f64().round() as u32),
                                format: Some(m.format),
                            }));
                        }
                    }
                });
                if let Some(c) = v.color {
                    ui.label(format!("{}: {} · {:?} · {:?} {}", t.signal, transfer_label(t, Some(c.transfer)), c.primaries, c.range, t.range_suffix));
                }
                if let Some(e) = v.capture_error {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("{}: {e}", t.capture_error));
                } else if v.devices.is_empty() {
                    ui.colored_label(egui::Color32::LIGHT_RED, t.no_device);
                }
                if ui.button(t.reconnect).clicked() {
                    actions.push(Action::RestartCapture);
                }
            });

            egui::CollapsingHeader::new(t.color).default_open(false).show(ui, |ui| {
                egui::ComboBox::from_label(t.transfer).selected_text(transfer_label(t, s.color.transfer)).show_ui(ui, |ui| {
                    for tf in [None, Some(TransferFunction::Pq), Some(TransferFunction::Hlg), Some(TransferFunction::Srgb), Some(TransferFunction::Gamma22), Some(TransferFunction::Bt1886)] {
                        changed |= ui.selectable_value(&mut s.color.transfer, tf, transfer_label(t, tf)).changed();
                    }
                });
                let range_label = |r: Option<Range>| match r {
                    None => t.auto,
                    Some(Range::Limited) => t.range_limited,
                    Some(Range::Full) => t.range_full,
                };
                egui::ComboBox::from_label(t.quant_range).selected_text(range_label(s.color.range)).show_ui(ui, |ui| {
                    for r in [None, Some(Range::Limited), Some(Range::Full)] {
                        changed |= ui.selectable_value(&mut s.color.range, r, range_label(r)).changed();
                    }
                });
                changed |= ui.checkbox(&mut s.color.assume_hdr_for_10bit, t.assume_hdr_10bit).changed();
            });

            egui::CollapsingHeader::new(t.interpolation).default_open(true).show(ui, |ui| {
                let label = |b: InterpolationBackend| match b {
                    InterpolationBackend::Off => t.interp_off,
                    InterpolationBackend::NvidiaVfg => t.interp_vfg,
                };
                egui::ComboBox::from_label(t.backend).selected_text(label(s.interpolation.backend)).show_ui(ui, |ui| {
                    for b in [InterpolationBackend::Off, InterpolationBackend::NvidiaVfg] {
                        changed |= ui.selectable_value(&mut s.interpolation.backend, b, label(b)).changed();
                    }
                });
                egui::ComboBox::from_label(t.multiplier).selected_text(format!("{}×", s.interpolation.max_multiplier)).show_ui(ui, |ui| {
                    for m in [2u32, 3, 4] {
                        changed |= ui.selectable_value(&mut s.interpolation.max_multiplier, m, format!("{m}×")).changed();
                    }
                });
                ui.small(t.multiplier_note);
                if s.interpolation.backend == InterpolationBackend::NvidiaVfg {
                    let q = |m: u32| [t.quality_low, t.quality_medium, t.quality_high][m.min(2) as usize];
                    egui::ComboBox::from_label(t.vfg_quality).selected_text(q(s.interpolation.vfg_quality)).show_ui(ui, |ui| {
                        for m in 0..3u32 {
                            changed |= ui.selectable_value(&mut s.interpolation.vfg_quality, m, q(m)).changed();
                        }
                    });
                }
                ui.label(format!("{}: {}", t.status, status_label(t, v.interp)));
                let mut frozen = v.frozen;
                if ui.checkbox(&mut frozen, t.freeze).changed() {
                    actions.push(Action::SetFreeze(frozen));
                }
            });

            egui::CollapsingHeader::new(t.audio).default_open(false).show(ui, |ui| {
                changed |= ui.checkbox(&mut s.audio.enabled, t.audio_enable).changed();
                changed |= ui.add(egui::Slider::new(&mut s.audio.volume, 0.0..=2.0).text(t.volume)).changed();
                changed |= ui.add(egui::Slider::new(&mut s.audio.av_offset_ms, -50..=200).suffix(" ms").text(t.av_offset)).changed();
                match v.audio {
                    Some(a) => {
                        ui.label(&a.endpoint);
                        ui.small(format!("{} Hz · {} ch · {} {} ms · {} {}", a.rate, a.channels, t.buffer, a.fill_ms, t.underruns, a.underruns));
                    }
                    None => {
                        ui.small(t.audio_none);
                    }
                }
            });

            egui::CollapsingHeader::new(t.display).default_open(false).show(ui, |ui| {
                let d = v.display;
                ui.label(format!(
                    "{} · {} · {} {:.0} nit · {} {:.0} nit · {:.2} Hz",
                    d.device_name,
                    if d.hdr { t.hdr_on } else { "SDR" },
                    t.peak,
                    d.max_nits,
                    t.sdr_white,
                    d.sdr_white_nits,
                    d.refresh_hz
                ));
                changed |= ui.checkbox(&mut s.display.vsync, t.vsync).changed();
                let mut catmull = s.display.scaling == ScalingFilter::CatmullRom;
                if ui.checkbox(&mut catmull, t.catmull).changed() {
                    s.display.scaling = if catmull { ScalingFilter::CatmullRom } else { ScalingFilter::Bilinear };
                    changed = true;
                }
                let mut peak = s.display.peak_nits_override.unwrap_or(d.max_nits);
                let mut use_peak = s.display.peak_nits_override.is_some();
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut use_peak, t.peak_override).changed();
                    changed |= ui.add_enabled(use_peak, egui::Slider::new(&mut peak, 200.0..=4000.0).suffix(" nit")).changed();
                });
                s.display.peak_nits_override = use_peak.then_some(peak);
                if ui.button(t.fullscreen).clicked() {
                    actions.push(Action::ToggleFullscreen);
                }
            });

            ui.horizontal(|ui| {
                ui.label(t.language);
                for (lang, name) in [(Language::Auto, t.auto), (Language::Zh, "中文"), (Language::En, "English")] {
                    changed |= ui.selectable_value(&mut s.ui.language, lang, name).changed();
                }
            });

            ui.separator();
            let st = v.stats;
            let r = v.rates;
            ui.monospace(format!(
                "{} {:5.1} fps  {} {:5.1} fps ({} {:4.1})\n{} {:5.1} fps  {} {:5.1} fps  ×{}\n{} {:5.2} ms\n{} {:5.1} ms\nGPU: {}",
                t.stats_capture,
                r.capture_fps,
                t.stats_unique,
                r.unique_fps,
                t.stats_content,
                st.content_fps,
                t.stats_output,
                r.present_fps,
                t.stats_interp,
                r.interp_fps,
                st.multiplier,
                t.stats_interp_gpu,
                st.interp_gpu_ms,
                t.stats_latency,
                st.latency_ms,
                v.adapter
            ));
        });

    if changed {
        actions.push(Action::SettingsChanged);
    }
    actions
}
