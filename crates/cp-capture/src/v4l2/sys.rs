//! The subset of the V4L2 kernel ABI (`linux/videodev2.h`) used for
//! single-planar MMAP capture. Layouts are checked against the 64-bit kernel
//! sizes at compile time.

#![allow(dead_code)]

use std::os::fd::{AsRawFd, BorrowedFd};

pub const BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
pub const MEMORY_MMAP: u32 = 1;
pub const FIELD_NONE: u32 = 1;

pub const CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
pub const CAP_STREAMING: u32 = 0x0400_0000;
pub const CAP_DEVICE_CAPS: u32 = 0x8000_0000;

pub const FRMSIZE_TYPE_DISCRETE: u32 = 1;
pub const FRMIVAL_TYPE_DISCRETE: u32 = 1;

pub const BUF_FLAG_ERROR: u32 = 0x0000_0040;
pub const BUF_FLAG_TIMESTAMP_MASK: u32 = 0x0000_e000;
pub const BUF_FLAG_TIMESTAMP_MONOTONIC: u32 = 0x0000_2000;

pub const fn fourcc(code: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*code)
}

pub const PIX_FMT_NV12: u32 = fourcc(b"NV12");
pub const PIX_FMT_P010: u32 = fourcc(b"P010");
pub const PIX_FMT_YUYV: u32 = fourcc(b"YUYV");
pub const PIX_FMT_UYVY: u32 = fourcc(b"UYVY");
/// B, G, R, X in memory (Media Foundation RGB32).
pub const PIX_FMT_XBGR32: u32 = fourcc(b"XR24");
/// Deprecated alias with the same memory layout.
pub const PIX_FMT_BGR32: u32 = fourcc(b"BGR4");

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Capability {
    pub driver: [u8; 16],
    pub card: [u8; 32],
    pub bus_info: [u8; 32],
    pub version: u32,
    pub capabilities: u32,
    pub device_caps: u32,
    pub reserved: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FmtDesc {
    pub index: u32,
    pub type_: u32,
    pub flags: u32,
    pub description: [u8; 32],
    pub pixelformat: u32,
    pub mbus_code: u32,
    pub reserved: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FrmSizeEnum {
    pub index: u32,
    pub pixel_format: u32,
    pub type_: u32,
    /// `discrete {width, height}` or `stepwise {min_w, max_w, step_w,
    /// min_h, max_h, step_h}`.
    pub size: [u32; 6],
    pub reserved: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fract {
    pub numerator: u32,
    pub denominator: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FrmIvalEnum {
    pub index: u32,
    pub pixel_format: u32,
    pub width: u32,
    pub height: u32,
    pub type_: u32,
    /// `discrete` is the first [`Fract`]; stepwise uses all three.
    pub interval: [Fract; 3],
    pub reserved: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PixFormat {
    pub width: u32,
    pub height: u32,
    pub pixelformat: u32,
    pub field: u32,
    pub bytesperline: u32,
    pub sizeimage: u32,
    pub colorspace: u32,
    pub priv_: u32,
    pub flags: u32,
    pub ycbcr_enc: u32,
    pub quantization: u32,
    pub xfer_func: u32,
}

/// `struct v4l2_format`; the kernel union holds pointers, so it is 8-byte
/// aligned after the 4-byte type.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Format {
    pub type_: u32,
    pub fmt: FormatUnion,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union FormatUnion {
    pub pix: PixFormat,
    raw: [u64; 25],
}

impl Format {
    pub fn capture() -> Self {
        Self {
            type_: BUF_TYPE_VIDEO_CAPTURE,
            fmt: FormatUnion { raw: [0; 25] },
        }
    }

    pub fn pix(&self) -> PixFormat {
        // SAFETY: every bit pattern is a valid PixFormat (plain u32 fields).
        unsafe { self.fmt.pix }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CaptureParm {
    pub capability: u32,
    pub capturemode: u32,
    pub timeperframe: Fract,
    pub extendedmode: u32,
    pub readbuffers: u32,
    pub reserved: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StreamParm {
    pub type_: u32,
    pub capture: CaptureParm,
    raw: [u8; 160],
}

impl StreamParm {
    pub fn capture() -> Self {
        Self {
            type_: BUF_TYPE_VIDEO_CAPTURE,
            capture: CaptureParm::default(),
            raw: [0; 160],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RequestBuffers {
    pub count: u32,
    pub type_: u32,
    pub memory: u32,
    pub capabilities: u32,
    pub flags: u8,
    pub reserved: [u8; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Timecode {
    pub type_: u32,
    pub flags: u32,
    pub frames: u8,
    pub seconds: u8,
    pub minutes: u8,
    pub hours: u8,
    pub userbits: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Buffer {
    pub index: u32,
    pub type_: u32,
    pub bytesused: u32,
    pub flags: u32,
    pub field: u32,
    pub timestamp: libc::timeval,
    pub timecode: Timecode,
    pub sequence: u32,
    pub memory: u32,
    /// `m.offset` for MMAP buffers (the union is pointer sized).
    pub m: u64,
    pub length: u32,
    pub reserved2: u32,
    pub request_fd: i32,
}

impl Buffer {
    pub fn mmap(index: u32) -> Self {
        Self {
            index,
            type_: BUF_TYPE_VIDEO_CAPTURE,
            bytesused: 0,
            flags: 0,
            field: 0,
            timestamp: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            timecode: Timecode::default(),
            sequence: 0,
            memory: MEMORY_MMAP,
            m: 0,
            length: 0,
            reserved2: 0,
            request_fd: 0,
        }
    }

    /// Offset to pass to `mmap` (low 32 bits of the union, little endian).
    pub fn offset(&self) -> u32 {
        self.m as u32
    }

    #[allow(
        clippy::unnecessary_cast,
        reason = "timeval fields are 32-bit on some targets"
    )]
    pub fn timestamp_ns(&self) -> i64 {
        self.timestamp.tv_sec as i64 * 1_000_000_000 + self.timestamp.tv_usec as i64 * 1_000
    }
}

const _: () = {
    assert!(size_of::<Capability>() == 104);
    assert!(size_of::<FmtDesc>() == 64);
    assert!(size_of::<FrmSizeEnum>() == 44);
    assert!(size_of::<FrmIvalEnum>() == 52);
    assert!(size_of::<PixFormat>() == 48);
    assert!(size_of::<Format>() == 208);
    assert!(size_of::<StreamParm>() == 204);
    assert!(size_of::<RequestBuffers>() == 20);
    assert!(size_of::<Buffer>() == 88);
};

// asm-generic ioctl encoding: dir << 30 | size << 16 | type << 8 | nr.
const fn ioc(dir: u64, nr: u64, size: usize) -> u64 {
    (dir << 30) | ((size as u64) << 16) | ((b'V' as u64) << 8) | nr
}
const fn ior<T>(nr: u64) -> u64 {
    ioc(2, nr, size_of::<T>())
}
const fn iow<T>(nr: u64) -> u64 {
    ioc(1, nr, size_of::<T>())
}
const fn iowr<T>(nr: u64) -> u64 {
    ioc(3, nr, size_of::<T>())
}

pub const VIDIOC_QUERYCAP: u64 = ior::<Capability>(0);
pub const VIDIOC_ENUM_FMT: u64 = iowr::<FmtDesc>(2);
pub const VIDIOC_G_FMT: u64 = iowr::<Format>(4);
pub const VIDIOC_S_FMT: u64 = iowr::<Format>(5);
pub const VIDIOC_REQBUFS: u64 = iowr::<RequestBuffers>(8);
pub const VIDIOC_QUERYBUF: u64 = iowr::<Buffer>(9);
pub const VIDIOC_QBUF: u64 = iowr::<Buffer>(15);
pub const VIDIOC_DQBUF: u64 = iowr::<Buffer>(17);
pub const VIDIOC_STREAMON: u64 = iow::<i32>(18);
pub const VIDIOC_STREAMOFF: u64 = iow::<i32>(19);
pub const VIDIOC_G_PARM: u64 = iowr::<StreamParm>(21);
pub const VIDIOC_S_PARM: u64 = iowr::<StreamParm>(22);
pub const VIDIOC_ENUM_FRAMESIZES: u64 = iowr::<FrmSizeEnum>(74);
pub const VIDIOC_ENUM_FRAMEINTERVALS: u64 = iowr::<FrmIvalEnum>(75);

/// `ioctl(fd, request, arg)`, retried on `EINTR`.
///
/// # Safety
/// `request` must be an ioctl whose argument type is `T`.
pub unsafe fn ioctl<T>(fd: BorrowedFd<'_>, request: u64, arg: &mut T) -> std::io::Result<()> {
    loop {
        // SAFETY: caller guarantees `request` takes a `*mut T`.
        let r = unsafe { libc::ioctl(fd.as_raw_fd(), request as _, std::ptr::from_mut(arg)) };
        if r != -1 {
            return Ok(());
        }
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() != Some(libc::EINTR) {
            return Err(e);
        }
    }
}

/// NUL-terminated byte array as a string.
pub fn c_str(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers_match_kernel() {
        // Values from videodev2.h on x86_64.
        assert_eq!(VIDIOC_QUERYCAP, 0x8068_5600);
        assert_eq!(VIDIOC_S_FMT, 0xc0d0_5605);
        assert_eq!(VIDIOC_REQBUFS, 0xc014_5608);
        assert_eq!(VIDIOC_DQBUF, 0xc058_5611);
        assert_eq!(VIDIOC_STREAMON, 0x4004_5612);
        assert_eq!(VIDIOC_S_PARM, 0xc0cc_5616);
        assert_eq!(VIDIOC_ENUM_FRAMEINTERVALS, 0xc034_564b);
    }
}
