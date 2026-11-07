//! NGX types shared by the D3D12 (Windows) and Vulkan (Linux) bindings.

/// `NVSDK_NGX_VSR_QualityLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsrQuality {
    Bicubic = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Ultra = 4,
}

/// Create-time options for DLSS Neural Rendering (NGX feature 18).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NrOptions {
    /// DLSSNR.Hint.Render.Preset: 0 leaves the model's default (or auto),
    /// 1 selects the quality model and 2 the performance model on the
    /// 310.9 builds. Earlier builds only know 0.
    pub preset: u32,
}

/// Live model tuning. These keys are read at evaluation time, so changing
/// them does not recreate the feature (the model preset does).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NrTuning {
    /// DLSSNR.Style: 0 default, 1 natural, 2 cinematic.
    pub style: u32,
    /// DLSSNR.Intensity: overall strength of the model's edit.
    pub intensity: f32,
    /// DLSSNR.LocalStructureStrength.
    pub local_structure: f32,
    /// DLSSNR.LocalToneStrength.
    pub local_tone: f32,
    /// DLSSNR.SkinStructureStrength.
    pub skin_structure: f32,
    /// DLSSNR.UseAutoMask.
    pub auto_mask: bool,
    /// DLSSNR.UICorrection.
    pub ui_correction: bool,
}

impl Default for NrTuning {
    fn default() -> Self {
        Self {
            style: 0,
            intensity: 1.0,
            local_structure: 1.0,
            local_tone: 1.0,
            skin_structure: 0.0,
            auto_mask: false,
            ui_correction: false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NgxError {
    #[error(
        "built without the NVIDIA NGX SDK (RTX Video SDK on Windows, DLSS SDK on Linux; see runtime/README.md)"
    )]
    NotBuilt,
    #[error("{call} failed: 0x{code:08X} ({})", result_name(*.code))]
    Call { call: &'static str, code: u32 },
    #[error("NVIDIA NGX feature is not available (GPU or driver)")]
    Unavailable,
}

#[cfg_attr(not(ngx), allow(dead_code))]
pub(crate) fn result_name(code: u32) -> &'static str {
    match code {
        0xBAD0_0001 => "feature not supported",
        0xBAD0_0002 => "platform error",
        0xBAD0_0003 => "feature already exists",
        0xBAD0_0004 => "feature not found",
        0xBAD0_0005 => "invalid parameter",
        0xBAD0_0007 => "not initialized",
        0xBAD0_0008 => "unsupported input format",
        0xBAD0_000B => "unable to initialize feature",
        0xBAD0_000C => "out of date (driver too old)",
        0xBAD0_000E => "unsupported format",
        _ => "error",
    }
}
