//! The capture thread: reads samples as fast as the device produces them and
//! forwards each one to the [`FrameSink`] with a single copy.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cp_core::format::{ModeRequest, select_mode};
use cp_core::settings::ColorSettings;
use cp_core::time::HostTime;
use cp_core::{ColorDesc, VideoMode};
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW,
};
use windows::core::{HRESULT, Interface, Ref, implement, w};

use crate::color_detect::{DetectedColor, resolve};
use crate::copy::{SrcImage, copy_frame};
use crate::mf::{self, DeviceInfo, MfThread, VIDEO_STREAM};
use crate::sink::{CaptureEvent, FrameMeta, FrameSink};
use crate::vendor;

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

/// One completed asynchronous read, stamped with its arrival time.
struct ReadResult {
    status: HRESULT,
    flags: u32,
    timestamp: i64,
    sample: Option<IMFSample>,
    arrival: HostTime,
}
// SAFETY: Media Foundation samples are free-threaded objects.
unsafe impl Send for ReadResult {}

/// Source-reader callback: forwards each sample to the capture thread. All
/// processing happens there so that stopping never waits on a driver call.
#[implement(IMFSourceReaderCallback)]
struct ReaderCallback {
    tx: Mutex<mpsc::Sender<ReadResult>>,
}

impl IMFSourceReaderCallback_Impl for ReaderCallback_Impl {
    fn OnReadSample(
        &self,
        status: HRESULT,
        _stream: u32,
        flags: u32,
        timestamp: i64,
        sample: Ref<IMFSample>,
    ) -> windows::core::Result<()> {
        let arrival = HostTime::now();
        let msg = ReadResult {
            status,
            flags,
            timestamp,
            sample: sample.cloned(),
            arrival,
        };
        if let Ok(tx) = self.tx.lock() {
            let _ = tx.send(msg);
        }
        Ok(())
    }

    fn OnFlush(&self, _stream: u32) -> windows::core::Result<()> {
        Ok(())
    }

    fn OnEvent(&self, _stream: u32, _event: Ref<IMFMediaEvent>) -> windows::core::Result<()> {
        Ok(())
    }
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
    #[error("Media Foundation: {0}")]
    Os(#[from] windows::core::Error),
    #[error("no mode matches the request (device offers {0} modes)")]
    NoMatchingMode(usize),
    #[error("driver accepted {got} instead of {want}")]
    ModeMismatch { want: VideoMode, got: String },
    #[error("device reported an error or end of stream")]
    StreamEnded,
    #[error(transparent)]
    Copy(#[from] crate::copy::CopyError),
}

struct ThreadCtx {
    cfg: CaptureConfig,
    sink: Arc<dyn FrameSink>,
    stop: Arc<AtomicBool>,
    color: Arc<Mutex<ColorSettings>>,
    stats: Arc<CaptureStats>,
}

const IDLE_POLL: Duration = Duration::from_millis(50);

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
    fn run(&self, events: &dyn Fn(CaptureEvent)) -> Result<(), CaptureError> {
        let _mf = MfThread::init()?;
        let mut task = 0u32;
        // SAFETY: registers this thread with MMCSS; reverted below.
        let mmcss = unsafe { AvSetMmThreadCharacteristicsW(w!("Capture"), &mut task) }.ok();

        let source = mf::open_source(&self.cfg.device, self.cfg.max_buffers)?;
        let result = self.read_loop(&source, events);
        // SAFETY: we own the source; Shutdown is idempotent.
        unsafe {
            let _ = source.Shutdown();
            if let Some(h) = mmcss {
                let _ = AvRevertMmThreadCharacteristics(h);
            }
        }
        result
    }

    fn read_loop(
        &self,
        source: &IMFMediaSource,
        events: &dyn Fn(CaptureEvent),
    ) -> Result<(), CaptureError> {
        let (tx, rx) = mpsc::channel();
        let callback: IMFSourceReaderCallback = ReaderCallback { tx: Mutex::new(tx) }.into();
        let reader = mf::create_reader(source, Some(&callback))?;
        let types = mf::native_types(&reader);
        let mut modes: Vec<VideoMode> = types.iter().map(|(_, m)| *m).collect();
        let chosen = select_mode(&modes, &self.cfg.request)
            .ok_or(CaptureError::NoMatchingMode(modes.len()))?;
        let (mt, _) = types
            .iter()
            .find(|(_, m)| *m == chosen)
            .ok_or(CaptureError::NoMatchingMode(modes.len()))?;
        modes.dedup();

        // SAFETY: valid reader and media type from its own native list.
        unsafe {
            reader.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)?;
            reader.SetStreamSelection(VIDEO_STREAM, true)?;
            reader.SetCurrentMediaType(VIDEO_STREAM, None, mt)?;
        }
        let (mut mode, mut detected) = self.current_type(&reader)?;
        if mode != chosen {
            return Err(CaptureError::ModeMismatch {
                want: chosen,
                got: mode.to_string(),
            });
        }
        let tonemap = vendor::configure_hdr_passthrough(
            &self.cfg.device,
            source,
            mode.format.bit_depth() > 8,
        );
        tracing::info!(%mode, ?detected, ?tonemap, "capture started");
        events(CaptureEvent::Started {
            mode,
            color: self.resolve_color(detected, &mode),
            modes: modes.clone(),
        });

        let mut clock = ClockMapper::default();
        let mut seq = 0u64;
        let request = || {
            // SAFETY: async mode; results arrive through the callback.
            unsafe { reader.ReadSample(VIDEO_STREAM, 0, None, None, None, None) }
        };
        request()?;
        while !self.stop.load(Ordering::Acquire) {
            let ReadResult {
                status,
                flags,
                timestamp: ts,
                sample,
                arrival,
            } = match rx.recv_timeout(IDLE_POLL) {
                Ok(r) => r,
                Err(RecvTimeoutError::Timeout) => continue, // no signal
                Err(RecvTimeoutError::Disconnected) => break,
            };
            status.ok()?;
            // Keep exactly one read outstanding, issued before the copy so the
            // driver can deliver the next frame while we work on this one.
            request()?;
            if flags & (MF_SOURCE_READERF_ERROR.0 | MF_SOURCE_READERF_ENDOFSTREAM.0) as u32 != 0 {
                return Err(CaptureError::StreamEnded);
            }
            if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
                (mode, detected) = self.current_type(&reader)?;
                tracing::info!(%mode, ?detected, "capture media type changed");
                events(CaptureEvent::Started {
                    mode,
                    color: self.resolve_color(detected, &mode),
                    modes: modes.clone(),
                });
            }
            let Some(sample) = sample else { continue };

            let color = self.resolve_color(detected, &mode);
            let device_time = HostTime(clock.map(ts * 100, arrival.0));
            match self.deliver(&sample, &mode, || {
                seq += 1;
                (seq, arrival, device_time, color)
            }) {
                Ok(true) => {
                    self.stats.frames.fetch_add(1, Ordering::Relaxed);
                }
                Ok(false) => {
                    self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => tracing::warn!("frame copy failed: {e}"),
            }
        }
        // SAFETY: drop pending reads before the reader goes away.
        unsafe {
            let _ = reader.Flush(VIDEO_STREAM);
        }
        Ok(())
    }

    fn current_type(
        &self,
        reader: &IMFSourceReader,
    ) -> Result<(VideoMode, DetectedColor), CaptureError> {
        // SAFETY: valid reader.
        let cur = unsafe { reader.GetCurrentMediaType(VIDEO_STREAM)? };
        let mode = mf::video_mode(&cur).ok_or_else(|| CaptureError::ModeMismatch {
            want: VideoMode {
                width: 0,
                height: 0,
                fps: cp_core::Rational::new(0, 1),
                format: cp_core::PixelFormat::Nv12,
            },
            got: "unsupported subtype".into(),
        })?;
        Ok((mode, mf::detected_color(&cur)))
    }

    fn resolve_color(&self, detected: DetectedColor, mode: &VideoMode) -> ColorDesc {
        let overrides = self.color.lock().map(|g| *g).unwrap_or_default();
        resolve(detected, &overrides, mode.format, mode.height)
    }

    /// Copies one sample into the sink. `Ok(false)` = no free slot (dropped).
    fn deliver(
        &self,
        sample: &IMFSample,
        mode: &VideoMode,
        next_meta: impl FnOnce() -> (u64, HostTime, HostTime, ColorDesc),
    ) -> Result<bool, CaptureError> {
        let Some(dest) = self.sink.acquire(mode) else {
            return Ok(false);
        };
        let copied = (|| {
            // SAFETY: capture samples carry a single buffer; this returns it
            // without copying in that case.
            let buffer = unsafe { sample.ConvertToContiguousBuffer()? };
            let locked = LockedBuffer::lock(buffer, mode)?;
            // SAFETY: `locked.image` describes memory locked for reading
            // until `locked` drops; `dest` is an exclusive sink lease.
            Ok::<_, CaptureError>(unsafe {
                copy_frame(&locked.image, &dest, mode.format, mode.width, mode.height)?
            })
        })();
        match copied {
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
                Ok(true)
            }
            Err(e) => {
                self.sink.cancel(dest);
                Err(e)
            }
        }
    }
}

/// A media buffer locked for reading; unlocks on drop.
enum LockedKind {
    TwoD(IMF2DBuffer),
    Plain(IMFMediaBuffer),
}

struct LockedBuffer {
    kind: LockedKind,
    image: SrcImage,
}

impl LockedBuffer {
    fn lock(buffer: IMFMediaBuffer, mode: &VideoMode) -> Result<Self, CaptureError> {
        let planes = mode.format.planes(mode.width, mode.height);
        let (row_bytes, height) = planes.planes[0];
        // SAFETY: each branch locks the buffer and records how to unlock it;
        // pointers stay valid until the matching Unlock in Drop.
        unsafe {
            if let Ok(b2) = buffer.cast::<IMF2DBuffer2>() {
                let (mut scan0, mut pitch, mut start, mut len) =
                    (std::ptr::null_mut(), 0i32, std::ptr::null_mut(), 0u32);
                b2.Lock2DSize(
                    MF2DBuffer_LockFlags_Read,
                    &mut scan0,
                    &mut pitch,
                    &mut start,
                    &mut len,
                )?;
                let pitch = pitch as isize;
                let before = scan0.offset_from(start).unsigned_abs();
                let avail = if pitch >= 0 {
                    (len as usize).saturating_sub(before)
                } else {
                    before + pitch.unsigned_abs()
                };
                let plane0_rows =
                    plane0_rows(len as usize, pitch.unsigned_abs(), height, planes.count);
                let image = SrcImage {
                    scanline0: scan0,
                    pitch,
                    plane0_rows,
                    len: avail,
                };
                return Ok(Self {
                    kind: LockedKind::TwoD(b2.cast()?),
                    image,
                });
            }
            let mut ptr = std::ptr::null_mut();
            let mut cur = 0u32;
            buffer.Lock(&mut ptr, None, Some(&mut cur))?;
            let image = SrcImage {
                scanline0: ptr,
                pitch: row_bytes as isize,
                plane0_rows: height,
                len: cur as usize,
            };
            Ok(Self {
                kind: LockedKind::Plain(buffer),
                image,
            })
        }
    }
}

/// Rows in plane 0 including driver padding (e.g. 1088 for 1080p NV12),
/// derived from the total buffer size when it matches exactly.
fn plane0_rows(total: usize, pitch: usize, height: usize, plane_count: usize) -> usize {
    if plane_count < 2 || pitch == 0 {
        return height;
    }
    [
        height,
        height.next_multiple_of(16),
        height.next_multiple_of(32),
        height.next_multiple_of(64),
    ]
    .into_iter()
    .find(|&rows| rows * pitch + rows.div_ceil(2) * pitch == total)
    .unwrap_or(height)
}

impl Drop for LockedBuffer {
    fn drop(&mut self) {
        // SAFETY: matches the successful lock in `lock`.
        unsafe {
            let _ = match &self.kind {
                LockedKind::TwoD(b) => b.Unlock2D(),
                LockedKind::Plain(b) => b.Unlock(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ClockMapper;

    #[test]
    fn padded_plane_rows() {
        assert_eq!(super::plane0_rows(1920 * 1088 * 3 / 2, 1920, 1080, 2), 1088);
        assert_eq!(super::plane0_rows(1920 * 1080 * 3 / 2, 1920, 1080, 2), 1080);
        assert_eq!(
            super::plane0_rows(1920 * 1080 * 2 + 77, 1920, 1080, 2),
            1080
        );
    }

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
