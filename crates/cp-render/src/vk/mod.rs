//! Vulkan renderer for Linux: the same pipeline as the D3D12 backend.
//!
//! `UploadRing` (capture thread memcpy into host-visible buffers) →
//! `ConvertPass` (YUV → linear scRGB FP16, compute) → optional
//! `Interpolator` / `Upscaler` → `OutputPass` (scale + tone map) →
//! `EguiPass` → `Presenter` (FP16 scRGB swapchain).
//!
//! Shaders are the D3D12 HLSL compiled to SPIR-V (see `build.rs`).
//! Descriptors are pushed per dispatch/draw (`VK_KHR_push_descriptor`),
//! rendering is dynamic and synchronization uses `synchronization2`, so
//! there are no pools, render passes or framebuffers.
//!
//! # HDR on Wayland
//! The swapchain uses `EXTENDED_SRGB_LINEAR` (Windows scRGB: 1.0 = 80 nits).
//! The compositor maps its reference white (203 nits for scRGB) onto the
//! output's reference luminance, so the renderer expresses the display in
//! *scRGB-encoded nits*: `encoded = real × 203 / reference`. With that,
//! the tone-mapping math is identical to Windows and the picture lands on
//! the display's real peak after the compositor's scaling.

pub mod arena;
pub mod clock;
pub mod convert;
pub mod device;
pub mod display;
pub mod egui_pass;
pub mod interp_vfg;
pub mod nr_proton;
pub mod output;
pub mod pipeline;
pub mod present_timing_ext;
pub mod presenter;
pub mod recording;
pub mod renderer;
#[cfg(test)]
mod tests;
pub mod texture;
pub mod timer;
pub mod upload;
pub mod upscale_ngx;

pub use interp_vfg::VfgInterpolator;
pub use renderer::{Enhancer, FrameInput, Interpolator, RecordCtx, Renderer, Upscaler};
pub use upscale_ngx::NgxVsrUpscaler;

use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkError {
    #[error("Vulkan: {0}")]
    Vk(#[from] vk::Result),
    #[error("cannot load the Vulkan loader: {0}")]
    Load(String),
    #[error("no Vulkan 1.3 device can present to this window")]
    NoDevice,
    #[error("no suitable memory type")]
    NoMemoryType,
    #[error("surface offers no usable format")]
    NoSurfaceFormat,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, VkError>;
