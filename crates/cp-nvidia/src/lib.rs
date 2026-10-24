//! Bindings to NVIDIA runtimes used by CapturePlay.
//!
//! Every runtime is loaded dynamically and is optional: a missing DLL only
//! disables the corresponding feature.
//!
//! * [`cuda`] – CUDA driver API subset for D3D12 interop (ships with the driver).
//! * [`vfx`] – Video Effects SDK plumbing (user-installed runtime).
//! * [`vfg`] – AI frame interpolation (Video Frame Generation).
//! * [`ngx`] – AI upscaling (RTX Video Super Resolution) through NGX on
//!   D3D12, from the RTX Video SDK (linked when present at build time).

pub mod cuda;
pub mod ngx;
pub mod vfg;
pub mod vfx;
