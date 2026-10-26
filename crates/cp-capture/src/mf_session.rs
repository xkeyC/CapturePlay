//! Media Foundation body of the capture thread: an asynchronous
//! `IMFSourceReader` with exactly one read outstanding.

use std::sync::Mutex;
use std::sync::mpsc::{self, RecvTimeoutError};

use cp_core::VideoMode;
use cp_core::format::select_mode;
use cp_core::time::HostTime;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW,
};
use windows::core::{HRESULT, Interface, Ref, implement, w};

use crate::color_detect::DetectedColor;
use crate::copy::{SrcImage, copy_frame, plane0_rows};
use crate::mf::{self, MfThread, VIDEO_STREAM};
use crate::session::{CaptureError, ClockMapper, IDLE_POLL, ThreadCtx};
use crate::sink::CaptureEvent;
use crate::vendor;

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

impl ThreadCtx {
    pub(crate) fn run(&self, events: &dyn Fn(CaptureEvent)) -> Result<(), CaptureError> {
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
        while !self.stopping() {
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
            self.deliver(
                &mode,
                |dest| {
                    // SAFETY: capture samples carry a single buffer; this
                    // returns it without copying in that case.
                    let buffer = unsafe { sample.ConvertToContiguousBuffer()? };
                    let locked = LockedBuffer::lock(buffer, &mode)?;
                    // SAFETY: `locked.image` describes memory locked for
                    // reading until `locked` drops; `dest` is an exclusive
                    // sink lease.
                    Ok(unsafe {
                        copy_frame(&locked.image, dest, mode.format, mode.width, mode.height)?
                    })
                },
                || {
                    seq += 1;
                    (seq, arrival, device_time, color)
                },
            );
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
