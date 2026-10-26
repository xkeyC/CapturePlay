//! Backend-independent renderer types: configuration, statistics, the
//! per-frame UI payload and the parameters of the decode and output passes.
//! Shared by the D3D12 (Windows) and Vulkan (Linux) renderers.

use cp_core::VideoMode;
use cp_core::pacing::{PacingConfig, Refresh};
use cp_core::settings::RefreshMode;
use egui::{ClippedPrimitive, TexturesDelta};

use crate::present_timing::RefreshKind;

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

/// What happened on present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentResult {
    Ok,
    /// Window occluded/minimized; skip rendering until visible.
    Occluded,
    DeviceLost,
}

/// Everything needed to draw one egui frame.
#[derive(Debug, Clone, Copy)]
pub struct EguiFrame<'a> {
    pub primitives: &'a [ClippedPrimitive],
    pub textures_delta: &'a TexturesDelta,
    pub pixels_per_point: f32,
    /// UI white in output units (scRGB on HDR displays, 1.0 on SDR).
    pub white_scale: f32,
}

/// Display-dependent parameters of the decode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeParams {
    /// Nits at which SDR white is placed.
    pub sdr_white_nits: f32,
    /// Display peak used for the HLG OOTF.
    pub hlg_peak_nits: f32,
}

/// How the working frame maps onto the current display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputParams {
    pub display_hdr: bool,
    /// Display peak (nits), used when `display_hdr`.
    pub display_peak_nits: f32,
    /// SDR white (nits); the target white on SDR displays.
    pub sdr_white_nits: f32,
    /// Mastering peak of the content (nits).
    pub content_peak_nits: f32,
    pub catmull_rom: bool,
}

/// Aspect-preserving fit of `src` into `dst`: `(x, y, w, h)` in pixels.
pub fn letterbox(src: (u32, u32), dst: (u32, u32)) -> (f32, f32, f32, f32) {
    let (sw, sh) = (src.0.max(1) as f32, src.1.max(1) as f32);
    let (dw, dh) = (dst.0.max(1) as f32, dst.1.max(1) as f32);
    let s = (dw / sw).min(dh / sh);
    let (w, h) = ((sw * s).round(), (sh * s).round());
    (((dw - w) * 0.5).floor(), ((dh - h) * 0.5).floor(), w, h)
}

/// Display-driven target changes (window resize) apply after this long, so
/// a window drag does not rebuild the upscaler on every step.
const UPSCALE_DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(300);

/// Debounced choice of the upscaled working size.
#[derive(Debug, Default)]
pub(crate) struct UpscaleTarget {
    src: Option<(u32, u32)>,
    current: Option<(u32, u32)>,
    pending: Option<(Option<(u32, u32)>, std::time::Instant)>,
}

impl UpscaleTarget {
    /// Returns the target to use now and whether it changed.
    pub(crate) fn update(
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

/// How long before a fixed-refresh vblank to present (variable refresh:
/// exactly on time).
pub(crate) fn slack_ns(refresh: Refresh) -> i64 {
    match refresh {
        Refresh::Variable { .. } => 0,
        Refresh::Fixed { period_ns, .. } => (period_ns / 4).min(1_000_000),
    }
}

#[cfg(test)]
mod tests {
    use super::letterbox;

    #[test]
    fn letterbox_fits_and_centers() {
        assert_eq!(
            letterbox((1920, 1080), (3840, 2160)),
            (0.0, 0.0, 3840.0, 2160.0)
        );
        assert_eq!(
            letterbox((1920, 1080), (3440, 1440)),
            (440.0, 0.0, 2560.0, 1440.0)
        );
        assert_eq!(
            letterbox((1920, 1080), (1920, 1200)),
            (0.0, 60.0, 1920.0, 1080.0)
        );
    }
}
