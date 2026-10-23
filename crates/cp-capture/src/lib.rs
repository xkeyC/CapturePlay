//! Capture-card video input via Media Foundation.
//!
//! One dedicated thread per device runs a synchronous `IMFSourceReader`
//! loop with `MF_LOW_LATENCY`, no format converters and a 2-deep driver
//! queue. Each sample is copied exactly once into consumer-owned memory
//! (see [`FrameSink`]) and fingerprinted for duplicate detection.

pub mod color_detect;
pub mod copy;
pub mod mf;
pub mod session;
pub mod sink;
pub mod vendor;

pub use mf::{DeviceInfo, MfThread, list_devices, list_modes};
pub use session::{CaptureConfig, CaptureSession};
pub use sink::{CaptureEvent, FrameDest, FrameMeta, FrameSink, PlaneDest};
