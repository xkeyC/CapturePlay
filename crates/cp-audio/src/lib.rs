//! Capture-card audio passthrough.
//!
//! The card's HDMI audio is a regular WASAPI capture endpoint; it is played
//! on the selected output with a fixed, small latency target. Clock drift
//! between the two devices is absorbed by sub-percent resampling
//! ([`drift`]), so latency never creeps up and no clicks are introduced.

pub mod drift;
pub mod endpoints;
pub mod passthrough;

pub use endpoints::{Endpoint, match_capture_endpoint};
pub use passthrough::{AudioConfig, AudioPassthrough, AudioShared};
