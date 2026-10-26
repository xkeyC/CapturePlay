//! The capture thread: reads samples as fast as the device produces them and
//! forwards each one to the [`FrameSink`] with a single copy.
//!
//! The thread body (`ThreadCtx::run`) is platform specific: Media Foundation
//! on Windows (`mf_session`), V4L2 on Linux (`v4l2::stream`). Everything
//! around it — lifetime, color resolution, delivery and stats — is shared.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cp_core::cadence::FrameSignature;
use cp_core::format::ModeRequest;
use cp_core::settings::ColorSettings;
use cp_core::time::HostTime;
use cp_core::{ColorDesc, VideoMode};

use crate::DeviceInfo;
use crate::color_detect::{DetectedColor, resolve};
use crate::sink::{CaptureEvent, FrameDest, FrameMeta, FrameSink};

#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub device: DeviceInfo,
    pub request: ModeRequest,
    pub color: ColorSettings,
    /// Driver-side queue depth. 2 keeps latency at ≤1 frame of buffering
    /// while tolerating one slow read.
    pub max_buffers: u32,
}

/// Statistics readable while capturing.
#[derive(Debug, Default)]
pub struct CaptureStats {
    pub frames: std::sync::atomic::AtomicU64,
    /// Frames dropped because the consumer had no free slot.
    pub dropped: std::sync::atomic::AtomicU64,
}

pub struct CaptureSession {
    stop: Arc<AtomicBool>,
    color: Arc<Mutex<ColorSettings>>,
    stats: Arc<CaptureStats>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for CaptureSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptureSession")
            .field("running", &self.thread.is_some())
            .finish()
    }
}

impl CaptureSession {
    pub fn start(
        cfg: CaptureConfig,
        sink: Arc<dyn FrameSink>,
        events: impl Fn(CaptureEvent) + Send + 'static,
    ) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let color = Arc::new(Mutex::new(cfg.color));
        let stats = Arc::new(CaptureStats::default());
        let ctx = ThreadCtx {
            cfg,
            sink,
            stop: stop.clone(),
            color: color.clone(),
            stats: stats.clone(),
        };
        let thread = std::thread::Builder::new()
            .name("capture".into())
            .spawn(move || {
                let error = ctx.run(&events).err().map(|e| e.to_string());
                if let Some(e) = &error {
                    tracing::error!("capture stopped: {e}");
                }
                events(CaptureEvent::Stopped { error });
            })?;
        Ok(Self {
            stop,
            color,
            stats,
            thread: Some(thread),
        })
    }

    /// Changes color overrides; applied from the next frame.
    pub fn set_color_settings(&self, c: ColorSettings) {
        if let Ok(mut g) = self.color.lock() {
            *g = c;
        }
    }

    pub fn stats(&self) -> &CaptureStats {
        &self.stats
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        // The capture thread polls this flag at least every `IDLE_POLL`,
        // even without an HDMI signal, so joining is bounded.
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[cfg(windows)]
    #[error("Media Foundation: {0}")]
    Os(#[from] windows::core::Error),
    #[cfg(target_os = "linux")]
    #[error("V4L2: {0}")]
    Os(#[from] std::io::Error),
    #[error("no mode matches the request (device offers {0} modes)")]
    NoMatchingMode(usize),
    #[error("driver accepted {got} instead of {want}")]
    ModeMismatch { want: VideoMode, got: String },
    #[error("device reported an error or end of stream")]
    StreamEnded,
    #[error(transparent)]
    Copy(#[from] crate::copy::CopyError),
}

pub(crate) struct ThreadCtx {
    pub(crate) cfg: CaptureConfig,
    pub(crate) sink: Arc<dyn FrameSink>,
    pub(crate) stop: Arc<AtomicBool>,
    pub(crate) color: Arc<Mutex<ColorSettings>>,
    pub(crate) stats: Arc<CaptureStats>,
}

pub(crate) const IDLE_POLL: Duration = Duration::from_millis(50);

/// Maps device timestamps onto the host clock. The offset is the minimum of
/// `arrival - device` seen so far (arrival jitter only ever adds delay), with
/// a slow upward relaxation so clock drift cannot pin it forever.
#[derive(Debug, Default)]
pub struct ClockMapper {
    offset: Option<i64>,
}

impl ClockMapper {
    const RELAX_NS: i64 = 2_000;

    pub fn map(&mut self, device_ns: i64, arrival_ns: i64) -> i64 {
        let candidate = arrival_ns - device_ns;
        let off = match self.offset {
            Some(o) => candidate.min(o + Self::RELAX_NS),
            None => candidate,
        };
        self.offset = Some(off);
        device_ns + off
    }
}

impl ThreadCtx {
    pub(crate) fn stopping(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    pub(crate) fn resolve_color(&self, detected: DetectedColor, mode: &VideoMode) -> ColorDesc {
        let overrides = self.color.lock().map(|g| *g).unwrap_or_default();
        resolve(detected, &overrides, mode.format, mode.height)
    }

    /// Copies one frame into the sink with `copy` and publishes it; counts
    /// a drop when the consumer has no free slot.
    pub(crate) fn deliver(
        &self,
        mode: &VideoMode,
        copy: impl FnOnce(&FrameDest) -> Result<FrameSignature, CaptureError>,
        next_meta: impl FnOnce() -> (u64, HostTime, HostTime, ColorDesc),
    ) {
        let Some(dest) = self.sink.acquire(mode) else {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        match copy(&dest) {
            Ok(signature) => {
                let (seq, arrival, device_time, color) = next_meta();
                self.sink.publish(
                    dest,
                    FrameMeta {
                        seq,
                        arrival,
                        device_time,
                        mode: *mode,
                        color,
                        signature,
                    },
                );
                self.stats.frames.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                self.sink.cancel(dest);
                tracing::warn!("frame copy failed: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ClockMapper;

    #[test]
    fn clock_mapper_tracks_minimum_delay() {
        let mut c = ClockMapper::default();
        // Device clock starts at 0, host at 1 s; arrivals jitter by 0–3 ms.
        let jitter = [3_000_000, 0, 1_000_000, 2_500_000, 0];
        let mut last = 0;
        for (i, j) in jitter.iter().enumerate() {
            let dev = i as i64 * 16_666_667;
            last = c.map(dev, 1_000_000_000 + dev + j);
        }
        assert!((last - (1_000_000_000 + 4 * 16_666_667)).abs() < 10_000);
    }
}
