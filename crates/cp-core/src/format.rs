//! Video modes and pixel formats offered by capture devices.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Exact rational frame rate (e.g. 60000/1001).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    pub num: u32,
    pub den: u32,
}

impl Rational {
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    pub fn as_f64(self) -> f64 {
        if self.den == 0 {
            0.0
        } else {
            f64::from(self.num) / f64::from(self.den)
        }
    }

    /// Frame duration in 100 ns units (Media Foundation time base).
    pub fn frame_duration_hns(self) -> i64 {
        if self.num == 0 {
            0
        } else {
            (10_000_000i64 * i64::from(self.den)) / i64::from(self.num)
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.as_f64();
        if (v - v.round()).abs() < 0.005 {
            write!(f, "{}", v.round() as u32)
        } else {
            write!(f, "{v:.2}")
        }
    }
}

/// Uncompressed pixel layouts the GPU converter understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
    /// 8-bit 4:2:0, Y plane + interleaved CbCr plane.
    Nv12,
    /// 10-bit 4:2:0 in 16-bit words (MSB aligned), NV12 layout.
    P010,
    /// 8-bit 4:2:2 packed Y0 Cb Y1 Cr.
    Yuy2,
    /// 8-bit 4:2:2 packed Cb Y0 Cr Y1.
    Uyvy,
    /// 8-bit BGRX (Media Foundation RGB32).
    Bgra8,
}

impl PixelFormat {
    /// Significant bits per component.
    pub fn bit_depth(self) -> u32 {
        match self {
            Self::P010 => 10,
            _ => 8,
        }
    }

    /// Bits per component as stored in memory.
    pub fn storage_bits(self) -> u32 {
        match self {
            Self::P010 => 16,
            _ => 8,
        }
    }

    pub fn is_yuv(self) -> bool {
        !matches!(self, Self::Bgra8)
    }

    /// Bytes needed for one frame with tightly packed rows.
    pub fn frame_bytes(self, width: u32, height: u32) -> usize {
        let (w, h) = (width as usize, height as usize);
        match self {
            Self::Nv12 => w * h * 3 / 2,
            Self::P010 => w * h * 3,
            Self::Yuy2 | Self::Uyvy => w * h * 2,
            Self::Bgra8 => w * h * 4,
        }
    }

    /// `(row_bytes, rows)` of each memory plane, top to bottom.
    pub fn planes(self, width: u32, height: u32) -> PlaneSizes {
        let w = width as usize;
        let h = height as usize;
        match self {
            Self::Nv12 => PlaneSizes::two((w, h), (w, h.div_ceil(2))),
            Self::P010 => PlaneSizes::two((w * 2, h), (w * 2, h.div_ceil(2))),
            Self::Yuy2 | Self::Uyvy => PlaneSizes::one((w * 2, h)),
            Self::Bgra8 => PlaneSizes::one((w * 4, h)),
        }
    }

    /// Preference when several formats offer the same size/rate. Higher wins.
    /// 10-bit first (needed for HDR), then 4:2:2 over 4:2:0, RGB last.
    pub fn preference(self) -> u8 {
        match self {
            Self::P010 => 4,
            Self::Yuy2 | Self::Uyvy => 3,
            Self::Nv12 => 2,
            Self::Bgra8 => 1,
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nv12 => "NV12",
            Self::P010 => "P010",
            Self::Yuy2 => "YUY2",
            Self::Uyvy => "UYVY",
            Self::Bgra8 => "RGB32",
        })
    }
}

/// Byte geometry of up to two memory planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneSizes {
    pub count: usize,
    /// `(row_bytes, rows)` per plane; unused entries are `(0, 0)`.
    pub planes: [(usize, usize); 2],
}

impl PlaneSizes {
    fn one(p: (usize, usize)) -> Self {
        Self {
            count: 1,
            planes: [p, (0, 0)],
        }
    }

    fn two(a: (usize, usize), b: (usize, usize)) -> Self {
        Self {
            count: 2,
            planes: [a, b],
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.planes[..self.count].iter().copied()
    }
}

/// One concrete capture mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoMode {
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    pub format: PixelFormat,
}

impl fmt::Display for VideoMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} @ {} {}",
            self.width, self.height, self.fps, self.format
        )
    }
}

/// What the user asked for; any field left `None` is "best available".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ModeRequest {
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Requested frame rate in whole fps (matched within ±0.05, so 60.0002 matches 60 but 59.94 does not).
    pub fps: Option<u32>,
    pub format: Option<PixelFormat>,
}

impl ModeRequest {
    fn matches(&self, m: &VideoMode) -> bool {
        self.width.is_none_or(|w| w == m.width)
            && self.height.is_none_or(|h| h == m.height)
            && self
                .fps
                .is_none_or(|f| (m.fps.as_f64() - f64::from(f)).abs() < 0.05)
            && self.format.is_none_or(|f| f == m.format)
    }
}

/// Picks the best mode satisfying `req`: largest area, then highest rate,
/// then the preferred pixel format. Returns `None` if nothing matches.
pub fn select_mode(modes: &[VideoMode], req: &ModeRequest) -> Option<VideoMode> {
    modes
        .iter()
        .filter(|m| req.matches(m))
        .copied()
        .max_by(|a, b| {
            let area = |m: &VideoMode| u64::from(m.width) * u64::from(m.height);
            area(a)
                .cmp(&area(b))
                .then(a.fps.as_f64().total_cmp(&b.fps.as_f64()))
                .then(a.format.preference().cmp(&b.format.preference()))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(w: u32, h: u32, fps: u32, format: PixelFormat) -> VideoMode {
        VideoMode {
            width: w,
            height: h,
            fps: Rational::new(fps * 1000, 1000),
            format,
        }
    }

    #[test]
    fn selects_highest_area_rate_then_format() {
        let modes = [
            m(1920, 1080, 120, PixelFormat::Nv12),
            m(3840, 2160, 30, PixelFormat::P010),
            m(3840, 2160, 60, PixelFormat::Nv12),
            m(3840, 2160, 30, PixelFormat::Nv12),
        ];
        assert_eq!(select_mode(&modes, &ModeRequest::default()), Some(modes[2]));
        let req = ModeRequest {
            fps: Some(30),
            height: Some(2160),
            ..Default::default()
        };
        assert_eq!(select_mode(&modes, &req), Some(modes[1]));
        let req = ModeRequest {
            format: Some(PixelFormat::Yuy2),
            ..Default::default()
        };
        assert_eq!(select_mode(&modes, &req), None);
    }

    #[test]
    fn ntsc_rate_matches_and_displays() {
        let r = Rational::new(60000, 1001);
        let mode = VideoMode {
            width: 1920,
            height: 1080,
            fps: r,
            format: PixelFormat::Nv12,
        };
        let req = ModeRequest {
            fps: Some(60),
            ..Default::default()
        };
        assert!(
            select_mode(&[mode], &req).is_none(),
            "59.94 must not match a 60 request"
        );
        assert_eq!(r.to_string(), "59.94");
        assert_eq!(Rational::new(60, 1).frame_duration_hns(), 166_666);
    }

    #[test]
    fn frame_sizes() {
        assert_eq!(PixelFormat::Nv12.frame_bytes(4, 2), 12);
        assert_eq!(PixelFormat::P010.frame_bytes(4, 2), 24);
        assert_eq!(PixelFormat::Yuy2.frame_bytes(4, 2), 16);
    }
}
