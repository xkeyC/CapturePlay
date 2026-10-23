//! Bindings to NVIDIA runtimes used by CapturePlay.
//!
//! Every runtime is loaded dynamically and is optional: a missing DLL only
//! disables the corresponding feature.
//!
//! * [`nvof`] – Optical Flow hardware (ships with the driver).
//! * [`cuda`] – CUDA driver API subset for D3D12 interop (ships with the driver).
//! * [`vfg`] – Video Effects SDK AI frame generation (user-installed runtime).

pub mod cuda;
pub mod nvof;
pub mod vfg;
