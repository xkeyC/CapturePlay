//! Renderer for CapturePlay: D3D12 on Windows, Vulkan on Linux (`vk`).
//!
//! Data flow per unique captured frame:
//! `UploadRing` (capture thread memcpy) → `ConvertPass` (YUV → linear
//! scRGB FP16) → optional `Interpolator` → `OutputPass` (scale + tone map
//! for the current display) → `EguiPass` → `Presenter` (FP16 scRGB flip
//! swapchain, waitable, max latency 1).
//!
//! Both backends share the platform-independent pieces: configuration and
//! statistics types, the upload ring, display info and present-timing
//! analysis.

#[cfg(windows)]
pub mod arena;
#[cfg(windows)]
pub mod clock;
pub mod common;
#[cfg(windows)]
pub mod convert;
#[cfg(windows)]
pub mod descriptors;
#[cfg(windows)]
pub mod display;
pub mod display_info;
#[cfg(windows)]
pub mod egui_pass;
#[cfg(windows)]
pub mod gpu;
#[cfg(windows)]
pub mod gpu_timer;
#[cfg(windows)]
pub mod interp_vfg;
#[cfg(windows)]
pub mod nr_ngx;
#[cfg(windows)]
pub mod output;
pub mod present_timing;
#[cfg(windows)]
pub mod presenter;
#[cfg(windows)]
pub mod renderer;
#[cfg(windows)]
pub mod shaders;
#[cfg(windows)]
pub mod texture;
pub mod upload;
#[cfg(windows)]
pub mod upscale_ngx;
#[cfg(target_os = "linux")]
pub mod vk;

#[cfg(test)]
mod half;
#[cfg(all(test, windows))]
mod testutil;
#[cfg(all(test, windows))]
mod upscale_bench;

pub use common::{CompositorVrr, EguiFrame, PresentResult, RenderConfig, RenderStats};
pub use display_info::DisplayInfo;
#[cfg(windows)]
pub use interp_vfg::VfgInterpolator;
#[cfg(windows)]
pub use nr_ngx::{NgxNrEnhancer, NrPost};
pub use present_timing::RefreshKind;
#[cfg(windows)]
pub use renderer::{Enhancer, FrameInput, Interpolator, RecordCtx, Renderer, Upscaler};
#[cfg(windows)]
pub use texture::Texture;
#[cfg(windows)]
pub use upscale_ngx::NgxVsrUpscaler;
#[cfg(target_os = "linux")]
pub use vk::nr_proton::{ProtonNrConfig, ProtonNrEnhancer, discover as nr_discover};
#[cfg(target_os = "linux")]
pub use vk::{
    Enhancer, FrameInput, Interpolator, NgxVsrUpscaler, RecordCtx, Renderer, Upscaler,
    VfgInterpolator,
};
