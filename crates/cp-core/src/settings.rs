//! Persistent user settings (TOML). Every field has a default so old files
//! keep loading after new options are added.

use serde::{Deserialize, Serialize};

use crate::color::{Primaries, Range, TransferFunction, YuvMatrix};
use crate::format::ModeRequest;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Settings {
    pub ui: UiSettings,
    pub capture: CaptureSettings,
    pub color: ColorSettings,
    pub interpolation: InterpolationSettings,
    pub enhance: EnhanceSettings,
    pub upscale: UpscaleSettings,
    pub display: DisplaySettings,
    pub audio: AudioSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    /// Follow the Windows display language.
    #[default]
    Auto,
    Zh,
    En,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct UiSettings {
    pub language: Language,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct CaptureSettings {
    /// Device symbolic link (stable across reboots); `None` = first device.
    pub device: Option<String>,
    pub mode: ModeRequest,
}

/// Per-field override of what the device reports. `None` = auto-detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorSettings {
    pub transfer: Option<TransferFunction>,
    pub range: Option<Range>,
    pub matrix: Option<YuvMatrix>,
    pub primaries: Option<Primaries>,
    /// Treat 10-bit (P010) input without color metadata as HDR10. Consoles
    /// only send 10-bit over HDMI when HDR is active, so this is the right
    /// default for PS5/Xbox capture.
    pub assume_hdr_for_10bit: bool,
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            transfer: None,
            range: None,
            matrix: None,
            primaries: None,
            assume_hdr_for_10bit: true,
        }
    }
}

impl ColorSettings {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationBackend {
    /// No interpolation; lowest latency.
    Off,
    /// NVIDIA Video Effects SDK AI frame generation (RTX 40/50). The removed
    /// optical-flow backend's name maps here so old settings files load.
    #[default]
    #[serde(alias = "nv_optical_flow")]
    NvidiaVfg,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterpolationSettings {
    pub backend: InterpolationBackend,
    /// Output fps cap (e.g. 60). `None` = display refresh.
    pub output_fps_cap: Option<u32>,
    pub max_multiplier: u32,
    /// VFG model: 0 low, 1 medium, 2 high.
    pub vfg_quality: u32,
    /// Directory with the NVIDIA Video Effects runtime (VFG). `None` =
    /// `$CAPTUREPLAY_VFX_RUNTIME`, then `runtime/nvidia-vfx` next to the exe.
    #[serde(alias = "vfg_runtime_dir")]
    pub vfx_runtime_dir: Option<String>,
}

impl Default for InterpolationSettings {
    fn default() -> Self {
        // 2x is the lowest-latency interpolation (adds half a source frame);
        // higher multipliers add (M-1)/M of a source frame.
        Self {
            backend: InterpolationBackend::default(),
            output_fps_cap: None,
            max_multiplier: 2,
            vfg_quality: 1,
            vfx_runtime_dir: None,
        }
    }
}

/// Per-frame enhancement stage (runs before interpolation). No backend ships
/// yet; officially published SDKs (e.g. a public DLSS neural rendering SDK)
/// are added as new variants and implemented via `cp_render::Enhancer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct EnhanceSettings {
    pub backend: EnhanceBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnhanceBackend {
    #[default]
    Off,
}

/// Upscaling of the captured frame to the display (e.g. 1440p HDR → 4K).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct UpscaleSettings {
    pub backend: UpscaleBackend,
    pub vsr_quality: VsrQualitySetting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UpscaleBackend {
    /// Output pass only (perceptual-domain Catmull-Rom or bilinear).
    Off,
    /// NVIDIA RTX Video Super Resolution (Video Effects SDK), RTX 20+.
    #[default]
    NvidiaVsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VsrQualitySetting {
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaySettings {
    /// Brightness of SDR white in nits; `None` = Windows "SDR content brightness".
    pub sdr_white_nits: Option<f32>,
    /// Peak used for tone mapping; `None` = display EDID.
    pub peak_nits_override: Option<f32>,
    /// Present with tearing allowed when vsync is off (needed for VRR).
    pub allow_tearing: bool,
    pub vsync: bool,
    pub start_fullscreen: bool,
    /// Integer/aspect scaling filter.
    pub scaling: ScalingFilter,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            sdr_white_nits: None,
            peak_nits_override: None,
            allow_tearing: true,
            vsync: true,
            start_fullscreen: false,
            scaling: ScalingFilter::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScalingFilter {
    Bilinear,
    #[default]
    CatmullRom,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    pub enabled: bool,
    /// Capture endpoint id; `None` = match the video device.
    pub input: Option<String>,
    /// Render endpoint id; `None` = system default.
    pub output: Option<String>,
    /// Target buffered audio in ms (latency vs. dropout safety).
    pub target_latency_ms: u32,
    /// Extra delay applied to audio to match video processing latency.
    pub av_offset_ms: i32,
    pub volume: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            input: None,
            output: None,
            target_latency_ms: 30,
            av_offset_ms: 0,
            volume: 1.0,
        }
    }
}

impl Settings {
    pub fn new() -> Self {
        Self {
            color: ColorSettings::new(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_sections_use_real_defaults() {
        let s: Settings = toml::from_str(
            "[capture.mode]
fps = 60
",
        )
        .expect("parse");
        assert!(s.color.assume_hdr_for_10bit);
        assert_eq!(s.interpolation.max_multiplier, 2);
    }

    #[test]
    fn removed_optical_flow_backend_maps_to_vfg() {
        let s: Settings =
            toml::from_str("[interpolation]\nbackend = \"nv_optical_flow\"\n").expect("parse");
        assert_eq!(s.interpolation.backend, InterpolationBackend::NvidiaVfg);
    }

    #[test]
    fn ignores_settings_of_removed_features() {
        let s: Settings = toml::from_str(
            "[enhance.dlss_nr]
enabled = true
",
        )
        .expect("parse");
        assert_eq!(s.enhance.backend, EnhanceBackend::Off);
    }

    #[test]
    fn round_trips_and_tolerates_missing_fields() {
        let s = Settings::new();
        let text = toml::to_string(&s).expect("serialize");
        let back: Settings = toml::from_str(&text).expect("parse");
        assert_eq!(back, s);
        let partial: Settings =
            toml::from_str("[interpolation]\nbackend = \"off\"\n").expect("partial");
        assert_eq!(partial.interpolation.backend, InterpolationBackend::Off);
        assert_eq!(partial.audio.target_latency_ms, 30);
    }
}
