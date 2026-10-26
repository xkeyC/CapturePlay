//! Capture-card audio passthrough.
//!
//! The card's HDMI audio is a regular capture endpoint (WASAPI on Windows,
//! a PipeWire node on Linux);
//! it is played on the selected output with a fixed, small latency target.
//! Clock drift between the two devices is absorbed by sub-percent
//! resampling ([`drift`]), so latency never creeps up and no clicks are
//! introduced.

pub mod config;
pub mod drift;
pub mod matching;
pub mod playout;

#[cfg(windows)]
pub mod endpoints;
#[cfg(windows)]
pub mod passthrough;
#[cfg(target_os = "linux")]
mod pipewire_backend;
#[cfg(target_os = "linux")]
pub use pipewire_backend::{endpoints, passthrough};
#[cfg(not(any(windows, target_os = "linux")))]
mod unsupported;
#[cfg(not(any(windows, target_os = "linux")))]
pub use unsupported::{endpoints, passthrough};

pub use config::{AudioConfig, AudioShared};
pub use matching::{Endpoint, match_capture_endpoint};
pub use passthrough::AudioPassthrough;
