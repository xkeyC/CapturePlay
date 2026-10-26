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
