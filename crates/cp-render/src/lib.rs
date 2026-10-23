//! D3D12 renderer for CapturePlay.
//!
//! Data flow per unique captured frame:
//! `UploadRing` (capture thread memcpy) → `ConvertPass` (YUV → linear
//! scRGB FP16) → optional `Interpolator` → `OutputPass` (scale + tone map
//! for the current display) → `EguiPass` → `Presenter` (FP16 scRGB flip
//! swapchain, waitable, max latency 1).

pub mod arena;
pub mod convert;
pub mod descriptors;
pub mod display;
pub mod egui_pass;
pub mod gpu;
pub mod gpu_timer;
pub mod interp_vfg;
pub mod output;
pub mod presenter;
pub mod renderer;
pub mod shaders;
pub mod texture;
pub mod upload;
pub mod upscale_vsr;

#[cfg(test)]
mod testutil;
#[cfg(test)]
mod upscale_bench;

pub use display::DisplayInfo;
pub use egui_pass::EguiFrame;
pub use interp_vfg::VfgInterpolator;
pub use presenter::PresentResult;
pub use renderer::{
    Enhancer, FrameInput, Interpolator, RecordCtx, RenderConfig, RenderStats, Renderer, Upscaler,
};
pub use texture::Texture;
pub use upscale_vsr::VsrUpscaler;
