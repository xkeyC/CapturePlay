//! Hand-off of captured frames to the consumer (the renderer).
//!
//! The capture thread copies each frame exactly once, straight from the
//! driver's buffer into memory the consumer owns (on the renderer side this
//! is a persistently mapped D3D12 upload heap). The consumer decides plane
//! offsets and pitches so it can copy to GPU textures without repacking.

use cp_core::cadence::FrameSignature;
use cp_core::time::HostTime;
use cp_core::{ColorDesc, VideoMode};

/// Destination of one plane inside a [`FrameDest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneDest {
    pub offset: usize,
    pub pitch: usize,
}

/// Writable memory for one frame, valid until passed to
/// [`FrameSink::publish`] or [`FrameSink::cancel`].
#[derive(Debug)]
pub struct FrameDest {
    pub slot: u32,
    pub base: *mut u8,
    pub len: usize,
    pub planes: [PlaneDest; 2],
}

// SAFETY: a FrameDest is an exclusive lease of sink memory for one frame;
// the sink guarantees nobody else touches that slot until it is returned.
unsafe impl Send for FrameDest {}

/// Everything known about a captured frame besides its pixels.
#[derive(Debug, Clone)]
pub struct FrameMeta {
    /// Monotonic sequence number of delivered frames.
    pub seq: u64,
    /// Host time the sample was returned by Media Foundation.
    pub arrival: HostTime,
    /// Device timestamp converted to the host clock.
    pub device_time: HostTime,
    pub mode: VideoMode,
    pub color: ColorDesc,
    pub signature: FrameSignature,
}

pub trait FrameSink: Send + Sync {
    /// Leases memory for the next frame in `mode`. `None` drops the frame
    /// (consumer busy); the capture loop never blocks on the consumer.
    fn acquire(&self, mode: &VideoMode) -> Option<FrameDest>;
    /// Hands a fully written frame to the consumer.
    fn publish(&self, dest: FrameDest, meta: FrameMeta);
    /// Returns a lease without publishing (copy failed).
    fn cancel(&self, dest: FrameDest);
}

/// Events other than frames.
#[derive(Debug, Clone)]
pub enum CaptureEvent {
    /// Streaming started (or the format changed). `modes` lists everything
    /// the device offers, for the UI.
    Started {
        mode: VideoMode,
        color: ColorDesc,
        modes: Vec<VideoMode>,
    },
    /// The stream stopped; `error` is `None` on a requested stop.
    Stopped { error: Option<String> },
}
