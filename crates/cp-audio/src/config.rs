//! Passthrough configuration and live counters (platform independent).

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64};

#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Capture endpoint id.
    pub input_id: String,
    /// Render endpoint id; `None` = default console device.
    pub output_id: Option<String>,
    pub target_latency_ms: u32,
    /// Extra audio delay to line up with the video pipeline (may be < 0 to
    /// reduce the target, never below a safety floor).
    pub av_offset_ms: i32,
    pub volume: f32,
}

/// Live counters/controls shared with the UI.
#[derive(Debug, Default)]
pub struct AudioShared {
    pub volume_milli: AtomicU32,
    pub offset_ms: AtomicI32,
    pub underruns: AtomicU64,
    pub fill_ms: AtomicU32,
    pub sample_rate: AtomicU32,
    pub channels: AtomicU32,
}
