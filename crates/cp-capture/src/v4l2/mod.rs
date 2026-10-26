//! Linux capture through V4L2: device discovery, mode enumeration and the
//! capture thread body (`stream`).
//!
//! USB capture cards are plain UVC devices (`uvcvideo`); they offer the same
//! uncompressed formats as on Windows (NV12, P010, YUY2) through
//! single-planar MMAP streaming.

mod stream;
mod sys;

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use cp_core::{PixelFormat, Rational, VideoMode};

/// A video capture device node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    /// Stable identifier, used to reopen the device and kept in settings:
    /// the `/dev/v4l/by-id` (or `by-path`) link, else the `/dev/videoN` node.
    pub id: String,
}

/// Lists capture devices that support streaming, in node order.
pub fn list_devices() -> io::Result<Vec<DeviceInfo>> {
    let mut nodes: Vec<(u32, PathBuf)> = std::fs::read_dir("/dev")?
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name();
            let n = name.to_str()?.strip_prefix("video")?.parse().ok()?;
            Some((n, e.path()))
        })
        .collect();
    nodes.sort();
    let links = stable_links();
    let mut out = Vec::new();
    for (_, node) in nodes {
        // Nodes we cannot open (permissions, busy) are simply not offered.
        let Ok(dev) = Device::open(&node) else {
            continue;
        };
        let Ok(cap) = dev.capability() else {
            continue;
        };
        if !is_streaming_capture(&cap) {
            continue;
        }
        let canonical = node.canonicalize().unwrap_or_else(|_| node.clone());
        let id = links
            .iter()
            .find(|(_, target)| *target == canonical)
            .map_or(node, |(link, _)| link.clone());
        out.push(DeviceInfo {
            name: display_name(&sys::c_str(&cap.card)),
            id: id.to_string_lossy().into_owned(),
        });
    }
    Ok(out)
}

/// Lists the supported modes of a device.
pub fn list_modes(dev: &DeviceInfo) -> io::Result<Vec<VideoMode>> {
    Ok(Device::open(Path::new(&dev.id))?.modes())
}

/// `/dev/v4l/by-id` links first (stable across ports), then `by-path`,
/// each with its canonical `/dev/videoN` target.
fn stable_links() -> Vec<(PathBuf, PathBuf)> {
    let mut out = Vec::new();
    for dir in ["/dev/v4l/by-id", "/dev/v4l/by-path"] {
        let Ok(rd) = std::fs::read_dir(dir) else {
            continue;
        };
        let mut links: Vec<_> = rd
            .filter_map(Result::ok)
            .filter_map(|e| Some((e.path(), e.path().canonicalize().ok()?)))
            .collect();
        // `...-video-index0` before other interfaces of the same device.
        links.sort();
        out.extend(links);
    }
    out
}

/// uvcvideo names devices `"<product>: <entity>"` truncated to 31 bytes,
/// e.g. `"Live Gamer Ultra 2.1: Live Game"`. A truncated repeat of the
/// product after the colon is dropped.
fn display_name(card: &str) -> String {
    match card.split_once(": ") {
        Some((product, rest)) if !rest.is_empty() && product.starts_with(rest) => {
            product.to_owned()
        }
        _ => card.to_owned(),
    }
}

fn is_streaming_capture(cap: &sys::Capability) -> bool {
    let caps = if cap.capabilities & sys::CAP_DEVICE_CAPS != 0 {
        cap.device_caps
    } else {
        cap.capabilities
    };
    caps & sys::CAP_VIDEO_CAPTURE != 0 && caps & sys::CAP_STREAMING != 0
}

pub(crate) fn pixel_format(fourcc: u32) -> Option<PixelFormat> {
    match fourcc {
        sys::PIX_FMT_NV12 => Some(PixelFormat::Nv12),
        sys::PIX_FMT_P010 => Some(PixelFormat::P010),
        sys::PIX_FMT_YUYV => Some(PixelFormat::Yuy2),
        sys::PIX_FMT_UYVY => Some(PixelFormat::Uyvy),
        sys::PIX_FMT_XBGR32 | sys::PIX_FMT_BGR32 => Some(PixelFormat::Bgra8),
        _ => None,
    }
}

fn fourcc(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Nv12 => sys::PIX_FMT_NV12,
        PixelFormat::P010 => sys::PIX_FMT_P010,
        PixelFormat::Yuy2 => sys::PIX_FMT_YUYV,
        PixelFormat::Uyvy => sys::PIX_FMT_UYVY,
        PixelFormat::Bgra8 => sys::PIX_FMT_XBGR32,
    }
}

/// An open, non-blocking device node.
#[derive(Debug)]
pub(crate) struct Device {
    file: File,
}

impl Device {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)?;
        Ok(Self { file })
    }

    pub(crate) fn fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }

    fn capability(&self) -> io::Result<sys::Capability> {
        let mut cap = sys::Capability::default();
        // SAFETY: QUERYCAP takes a v4l2_capability.
        unsafe { sys::ioctl(self.fd(), sys::VIDIOC_QUERYCAP, &mut cap)? };
        Ok(cap)
    }

    /// Every discrete (format, size, rate) the driver offers, in driver
    /// order; formats the converter does not understand are skipped.
    pub(crate) fn modes(&self) -> Vec<VideoMode> {
        let mut out = Vec::new();
        for i in 0.. {
            let mut desc = sys::FmtDesc {
                index: i,
                type_: sys::BUF_TYPE_VIDEO_CAPTURE,
                ..Default::default()
            };
            // SAFETY: ENUM_FMT takes a v4l2_fmtdesc; EINVAL ends the list.
            if unsafe { sys::ioctl(self.fd(), sys::VIDIOC_ENUM_FMT, &mut desc) }.is_err() {
                break;
            }
            let Some(format) = pixel_format(desc.pixelformat) else {
                continue;
            };
            for j in 0.. {
                let mut size = sys::FrmSizeEnum {
                    index: j,
                    pixel_format: desc.pixelformat,
                    ..Default::default()
                };
                // SAFETY: ENUM_FRAMESIZES takes a v4l2_frmsizeenum.
                if unsafe { sys::ioctl(self.fd(), sys::VIDIOC_ENUM_FRAMESIZES, &mut size) }.is_err()
                    || size.type_ != sys::FRMSIZE_TYPE_DISCRETE
                {
                    break;
                }
                let (width, height) = (size.size[0], size.size[1]);
                for k in 0.. {
                    let mut ival = sys::FrmIvalEnum {
                        index: k,
                        pixel_format: desc.pixelformat,
                        width,
                        height,
                        ..Default::default()
                    };
                    // SAFETY: ENUM_FRAMEINTERVALS takes a v4l2_frmivalenum.
                    if unsafe { sys::ioctl(self.fd(), sys::VIDIOC_ENUM_FRAMEINTERVALS, &mut ival) }
                        .is_err()
                        || ival.type_ != sys::FRMIVAL_TYPE_DISCRETE
                    {
                        break;
                    }
                    let iv = ival.interval[0];
                    if iv.numerator == 0 {
                        continue;
                    }
                    out.push(VideoMode {
                        width,
                        height,
                        fps: Rational::new(iv.denominator, iv.numerator),
                        format,
                    });
                }
            }
        }
        out
    }

    /// Applies `mode` and returns what the driver actually set: the pixel
    /// layout and the frame interval.
    pub(crate) fn set_mode(&self, mode: &VideoMode) -> io::Result<(sys::PixFormat, Rational)> {
        let mut fmt = sys::Format::capture();
        fmt.fmt.pix = sys::PixFormat {
            width: mode.width,
            height: mode.height,
            pixelformat: fourcc(mode.format),
            field: sys::FIELD_NONE,
            ..Default::default()
        };
        // SAFETY: S_FMT takes a v4l2_format.
        unsafe { sys::ioctl(self.fd(), sys::VIDIOC_S_FMT, &mut fmt)? };
        let mut parm = sys::StreamParm::capture();
        parm.capture.timeperframe = sys::Fract {
            numerator: mode.fps.den,
            denominator: mode.fps.num,
        };
        // SAFETY: S_PARM takes a v4l2_streamparm.
        unsafe { sys::ioctl(self.fd(), sys::VIDIOC_S_PARM, &mut parm)? };
        let tpf = parm.capture.timeperframe;
        Ok((
            fmt.pix(),
            Rational::new(tpf.denominator, tpf.numerator.max(1)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uvc_names() {
        assert_eq!(
            display_name("Live Gamer Ultra 2.1: Live Game"),
            "Live Gamer Ultra 2.1"
        );
        assert_eq!(display_name("USB Video: USB Video"), "USB Video");
        assert_eq!(
            display_name("Cam Link 4K: Video Capture"),
            "Cam Link 4K: Video Capture"
        );
        assert_eq!(display_name("plain"), "plain");
    }

    #[test]
    fn fourcc_round_trip() {
        for f in [
            PixelFormat::Nv12,
            PixelFormat::P010,
            PixelFormat::Yuy2,
            PixelFormat::Uyvy,
            PixelFormat::Bgra8,
        ] {
            assert_eq!(pixel_format(fourcc(f)), Some(f));
        }
        assert_eq!(fourcc(PixelFormat::P010), u32::from_le_bytes(*b"P010"));
    }
}
