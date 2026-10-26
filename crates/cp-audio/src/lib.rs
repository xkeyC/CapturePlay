//! Capture-card audio passthrough.
//!
//! The card's HDMI audio is a regular capture endpoint (WASAPI on Windows);
//! it is played on the selected output with a fixed, small latency target.
//! Clock drift between the two devices is absorbed by sub-percent
//! resampling ([`drift`]), so latency never creeps up and no clicks are
//! introduced.

pub mod config;
pub mod drift;
pub mod matching;

#[cfg(windows)]
pub mod endpoints;
#[cfg(windows)]
pub mod passthrough;
#[cfg(not(windows))]
mod unsupported;
#[cfg(not(windows))]
pub use unsupported::{endpoints, passthrough};

pub use config::{AudioConfig, AudioShared};
pub use matching::{Endpoint, match_capture_endpoint};
pub use passthrough::AudioPassthrough;
