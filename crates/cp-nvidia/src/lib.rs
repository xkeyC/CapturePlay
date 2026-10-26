//! Bindings to NVIDIA runtimes used by CapturePlay.
//!
//! Every runtime is loaded dynamically and is optional: a missing DLL only
//! disables the corresponding feature.
//!
//! Windows: CUDA ↔ D3D12, NGX on D3D12. Linux: CUDA ↔ Vulkan (opaque fds,
//! timeline semaphores), NGX on Vulkan ([`ngx_vk`]).
//!
//! * [`cuda`] – CUDA driver API subset for D3D12 interop (ships with the driver).
//! * [`vfx`] – Video Effects SDK plumbing (user-installed runtime).
//! * [`vfg`] – AI frame interpolation (Video Frame Generation).
//! * [`ngx`] – AI upscaling (RTX Video Super Resolution) through NGX on
//!   D3D12, from the RTX Video SDK (linked when present at build time).

#[cfg(any(windows, target_os = "linux"))]
pub mod cuda;
#[cfg(windows)]
pub mod ngx;
#[cfg(any(windows, target_os = "linux"))]
mod ngx_common;
#[cfg(target_os = "linux")]
pub mod ngx_vk;
#[cfg(any(windows, target_os = "linux"))]
pub mod vfg;
#[cfg(windows)]
pub mod vfx;
#[cfg(target_os = "linux")]
#[path = "vfx_linux.rs"]
pub mod vfx;
