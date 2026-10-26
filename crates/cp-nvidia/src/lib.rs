//! Bindings to NVIDIA runtimes used by CapturePlay.
//!
//! Every runtime is loaded dynamically and is optional: a missing DLL only
//! disables the corresponding feature.
//!
//! Windows only for now; the Linux backends (CUDA ↔ Vulkan, NGX on Vulkan)
//! are not implemented yet.
//!
//! * [`cuda`] – CUDA driver API subset for D3D12 interop (ships with the driver).
//! * [`vfx`] – Video Effects SDK plumbing (user-installed runtime).
//! * [`vfg`] – AI frame interpolation (Video Frame Generation).
//! * [`ngx`] – AI upscaling (RTX Video Super Resolution) through NGX on
//!   D3D12, from the RTX Video SDK (linked when present at build time).

#[cfg(windows)]
pub mod cuda;
#[cfg(windows)]
pub mod ngx;
#[cfg(windows)]
pub mod vfg;
#[cfg(windows)]
pub mod vfx;
