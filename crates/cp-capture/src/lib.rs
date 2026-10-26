//! Capture-card video input: Media Foundation on Windows, V4L2 on Linux.
//!
//! One dedicated thread per device reads frames with the lowest latency the
//! platform offers (MF: `MF_LOW_LATENCY`, no format converters and a 2-deep
//! driver queue; V4L2: MMAP streaming with a short queue). Each frame is
//! copied exactly once into consumer-owned memory (see [`FrameSink`]) and
//! fingerprinted for duplicate detection.

pub mod color_detect;
pub mod copy;
#[cfg(windows)]
pub mod mf;
#[cfg(windows)]
mod mf_session;
pub mod session;
pub mod sink;
#[cfg(target_os = "linux")]
pub mod v4l2;
#[cfg(windows)]
pub mod vendor;

#[cfg(windows)]
pub use mf::{DeviceInfo, MfThread, list_devices, list_modes};
pub use session::{CaptureConfig, CaptureSession};
pub use sink::{CaptureEvent, FrameDest, FrameMeta, FrameSink, PlaneDest};
#[cfg(target_os = "linux")]
pub use v4l2::{DeviceInfo, list_devices, list_modes};
