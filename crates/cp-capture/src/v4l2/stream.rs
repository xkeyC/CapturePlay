//! V4L2 body of the capture thread: MMAP streaming, one copy per frame
//! straight from the driver buffer into the sink.

use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;

use cp_core::format::select_mode;
use cp_core::time::HostTime;

use super::{Device, pixel_format, sys};
use crate::color_detect::DetectedColor;
use crate::copy::{SrcImage, copy_frame, plane0_rows};
use crate::session::{CaptureError, ClockMapper, IDLE_POLL, ThreadCtx};
use crate::sink::CaptureEvent;

impl ThreadCtx {
    pub(crate) fn run(&self, events: &dyn Fn(CaptureEvent)) -> Result<(), CaptureError> {
        let dev = Device::open(Path::new(&self.cfg.device.id))?;
        let modes = dev.modes();
        let chosen = select_mode(&modes, &self.cfg.request)
            .ok_or(CaptureError::NoMatchingMode(modes.len()))?;
        let (pix, fps) = dev.set_mode(&chosen)?;
        let got = pixel_format(pix.pixelformat);
        if (pix.width, pix.height, got) != (chosen.width, chosen.height, Some(chosen.format))
            || (fps.as_f64() - chosen.fps.as_f64()).abs() > chosen.fps.as_f64() * 0.005
        {
            return Err(CaptureError::ModeMismatch {
                want: chosen,
                got: format!(
                    "{}x{} {} @ {fps}",
                    pix.width,
                    pix.height,
                    sys::c_str(&pix.pixelformat.to_le_bytes())
                ),
            });
        }
        // Keep the enumerated rate: drivers may report the same interval
        // with a different fraction.
        let mode = chosen;
        let detected = DetectedColor::from_v4l2(
            pix.colorspace,
            pix.xfer_func,
            pix.ycbcr_enc,
            pix.quantization,
        );
        let planes = mode.format.planes(mode.width, mode.height);
        let pitch = pix.bytesperline as usize;
        let rows0 = plane0_rows(
            pix.sizeimage as usize,
            pitch,
            planes.planes[0].1,
            planes.count,
        );

        // `max_buffers` is the driver-side queue (as on Media Foundation);
        // one more buffer is in our hands while it is being copied, one
        // more absorbs a late wakeup.
        let stream = Stream::start(&dev, self.cfg.max_buffers.max(1) + 2)?;
        tracing::info!(
            %mode,
            ?detected,
            pitch,
            rows0,
            buffers = stream.maps.len(),
            "capture started"
        );
        let color = self.resolve_color(detected, &mode);
        events(CaptureEvent::Started { mode, color, modes });

        let mut clock = ClockMapper::default();
        let mut seq = 0u64;
        while !self.stopping() {
            let Some(buf) = stream.dequeue()? else {
                continue; // timeout: no signal, or stop requested
            };
            let arrival = HostTime::now();
            let map = &stream.maps[buf.index as usize];
            if buf.flags & sys::BUF_FLAG_ERROR == 0 {
                let device_time = if buf.flags & sys::BUF_FLAG_TIMESTAMP_MASK
                    == sys::BUF_FLAG_TIMESTAMP_MONOTONIC
                {
                    HostTime(clock.map(buf.timestamp_ns(), arrival.0))
                } else {
                    arrival
                };
                let image = SrcImage {
                    scanline0: map.ptr,
                    pitch: pitch as isize,
                    plane0_rows: rows0,
                    len: (buf.bytesused as usize).min(map.len),
                };
                let color = self.resolve_color(detected, &mode);
                self.deliver(
                    &mode,
                    |dest| {
                        // SAFETY: the buffer is dequeued, so the driver does
                        // not write it until it is queued again below;
                        // `image.len` is within the mapping. `dest` is an
                        // exclusive sink lease.
                        Ok(unsafe {
                            copy_frame(&image, dest, mode.format, mode.width, mode.height)?
                        })
                    },
                    || {
                        seq += 1;
                        (seq, arrival, device_time, color)
                    },
                );
            } else {
                // Incomplete transfer (e.g. USB packet loss); skip the frame.
                tracing::debug!(sequence = buf.sequence, "corrupted frame dropped");
            }
            stream.queue(buf.index)?;
        }
        Ok(())
    }
}

struct Mapping {
    ptr: *const u8,
    len: usize,
}

/// MMAP buffers queued to a streaming device; stops and unmaps on drop.
struct Stream<'a> {
    dev: &'a Device,
    maps: Vec<Mapping>,
}

impl<'a> Stream<'a> {
    fn start(dev: &'a Device, count: u32) -> io::Result<Self> {
        let mut req = sys::RequestBuffers {
            count,
            type_: sys::BUF_TYPE_VIDEO_CAPTURE,
            memory: sys::MEMORY_MMAP,
            ..Default::default()
        };
        // SAFETY: REQBUFS takes a v4l2_requestbuffers.
        unsafe { sys::ioctl(dev.fd(), sys::VIDIOC_REQBUFS, &mut req)? };
        let mut stream = Self {
            dev,
            maps: Vec::with_capacity(req.count as usize),
        };
        for index in 0..req.count {
            let mut buf = sys::Buffer::mmap(index);
            // SAFETY: QUERYBUF takes a v4l2_buffer.
            unsafe { sys::ioctl(dev.fd(), sys::VIDIOC_QUERYBUF, &mut buf)? };
            let len = buf.length as usize;
            // SAFETY: maps the driver buffer at the offset it reported;
            // unmapped in Drop.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    dev.fd().as_raw_fd(),
                    libc::off_t::from(buf.offset()),
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            stream.maps.push(Mapping {
                ptr: ptr.cast(),
                len,
            });
            stream.queue(index)?;
        }
        let mut ty = sys::BUF_TYPE_VIDEO_CAPTURE as i32;
        // SAFETY: STREAMON takes the buffer type as an int.
        unsafe { sys::ioctl(dev.fd(), sys::VIDIOC_STREAMON, &mut ty)? };
        Ok(stream)
    }

    fn queue(&self, index: u32) -> io::Result<()> {
        let mut buf = sys::Buffer::mmap(index);
        // SAFETY: QBUF takes a v4l2_buffer.
        unsafe { sys::ioctl(self.dev.fd(), sys::VIDIOC_QBUF, &mut buf) }
    }

    /// Waits up to [`IDLE_POLL`] for a filled buffer.
    fn dequeue(&self) -> io::Result<Option<sys::Buffer>> {
        let mut pfd = libc::pollfd {
            fd: self.dev.fd().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd.
        let n = unsafe { libc::poll(&mut pfd, 1, IDLE_POLL.as_millis() as i32) };
        if n < 0 {
            let e = io::Error::last_os_error();
            return if e.raw_os_error() == Some(libc::EINTR) {
                Ok(None)
            } else {
                Err(e)
            };
        }
        if n == 0 {
            return Ok(None);
        }
        let mut buf = sys::Buffer::mmap(0);
        // SAFETY: DQBUF takes a v4l2_buffer; the fd is non-blocking.
        match unsafe { sys::ioctl(self.dev.fd(), sys::VIDIOC_DQBUF, &mut buf) } {
            Ok(()) => Ok(Some(buf)),
            Err(e) if e.raw_os_error() == Some(libc::EAGAIN) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        let mut ty = sys::BUF_TYPE_VIDEO_CAPTURE as i32;
        // SAFETY: STREAMOFF takes the buffer type; it also dequeues all
        // buffers, so nothing writes the mappings afterwards.
        let _ = unsafe { sys::ioctl(self.dev.fd(), sys::VIDIOC_STREAMOFF, &mut ty) };
        for m in &self.maps {
            // SAFETY: mapped in `start` with this length.
            unsafe { libc::munmap(m.ptr.cast_mut().cast(), m.len) };
        }
        let mut req = sys::RequestBuffers {
            count: 0,
            type_: sys::BUF_TYPE_VIDEO_CAPTURE,
            memory: sys::MEMORY_MMAP,
            ..Default::default()
        };
        // SAFETY: REQBUFS(0) frees the driver buffers.
        let _ = unsafe { sys::ioctl(self.dev.fd(), sys::VIDIOC_REQBUFS, &mut req) };
    }
}
