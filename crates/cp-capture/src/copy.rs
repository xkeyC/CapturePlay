//! Plane copy from a driver buffer into a [`FrameDest`], plus the luma
//! fingerprint used for duplicate detection.

use cp_core::PixelFormat;
use cp_core::cadence::FrameSignature;

use crate::sink::FrameDest;

/// A locked source buffer. `pitch` may be negative (bottom-up RGB).
#[derive(Debug, Clone, Copy)]
pub struct SrcImage {
    pub scanline0: *const u8,
    pub pitch: isize,
    /// Rows between the start of plane 0 and plane 1 (≥ height; drivers
    /// may pad, e.g. 1088 for 1080p).
    pub plane0_rows: usize,
    /// Bytes readable from `scanline0` in the direction of `pitch`.
    pub len: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum CopyError {
    #[error("source buffer too small: need {need} bytes, have {have}")]
    SourceTooSmall { need: usize, have: usize },
    #[error("destination too small: need {need} bytes, have {have}")]
    DestTooSmall { need: usize, have: usize },
}

/// Copies all planes of `format` and returns the frame signature.
///
/// # Safety
/// `src.scanline0` must be valid for reads of every row described by
/// `src`, and `dest.base` valid for writes of `dest.len` bytes.
pub unsafe fn copy_frame(
    src: &SrcImage,
    dest: &FrameDest,
    format: PixelFormat,
    width: u32,
    height: u32,
) -> Result<FrameSignature, CopyError> {
    let planes = format.planes(width, height);
    let abs_pitch = src.pitch.unsigned_abs();

    // Validate both sides before touching memory.
    let mut src_need = 0usize;
    for (i, (row_bytes, rows)) in planes.iter().enumerate() {
        let start_row = if i == 0 { 0 } else { src.plane0_rows };
        src_need = src_need.max((start_row + rows - 1) * abs_pitch + row_bytes);
        let dp = dest.planes[i];
        let need = dp.offset + (rows - 1) * dp.pitch + row_bytes;
        if need > dest.len || dp.pitch < row_bytes {
            return Err(CopyError::DestTooSmall {
                need,
                have: dest.len,
            });
        }
    }
    if src_need > src.len {
        return Err(CopyError::SourceTooSmall {
            need: src_need,
            have: src.len,
        });
    }

    for (i, (row_bytes, rows)) in planes.iter().enumerate() {
        let start_row = if i == 0 { 0 } else { src.plane0_rows };
        let dp = dest.planes[i];
        for r in 0..rows {
            // SAFETY: bounds validated above for both buffers.
            unsafe {
                let s = src.scanline0.offset((start_row + r) as isize * src.pitch);
                let d = dest.base.add(dp.offset + r * dp.pitch);
                std::ptr::copy_nonoverlapping(s, d, row_bytes);
            }
        }
    }

    // Fingerprint from the source: the destination is usually write-combined
    // upload memory, which is very slow to read back.
    let read = |x: u32, y: u32| -> u16 {
        let row = y as isize * src.pitch;
        // SAFETY: (x, y) lies inside plane 0, validated above.
        unsafe {
            let p = src.scanline0.offset(row);
            match format {
                PixelFormat::Nv12 => u16::from(*p.add(x as usize)) << 8,
                PixelFormat::P010 => p.add(x as usize * 2).cast::<u16>().read_unaligned(),
                PixelFormat::Yuy2 => u16::from(*p.add(x as usize * 2)) << 8,
                PixelFormat::Uyvy => u16::from(*p.add(x as usize * 2 + 1)) << 8,
                PixelFormat::Bgra8 => u16::from(*p.add(x as usize * 4 + 1)) << 8,
            }
        }
    };
    Ok(FrameSignature::from_plane(width, height, read))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::PlaneDest;

    #[test]
    fn copies_padded_nv12_into_aligned_dest() {
        let (w, h) = (8u32, 4u32);
        let pitch = 16usize;
        let plane0_rows = 6; // padded
        let mut src = vec![0u8; pitch * plane0_rows + pitch * 2];
        for r in 0..4 {
            for x in 0..8 {
                src[r * pitch + x] = (r * 10 + x) as u8;
            }
        }
        for r in 0..2 {
            for x in 0..8 {
                src[(plane0_rows + r) * pitch + x] = 200 + (r * 8 + x) as u8;
            }
        }
        let mut dst = vec![0u8; 256 * 6];
        let dest = FrameDest {
            slot: 0,
            base: dst.as_mut_ptr(),
            len: dst.len(),
            planes: [
                PlaneDest {
                    offset: 0,
                    pitch: 256,
                },
                PlaneDest {
                    offset: 1024,
                    pitch: 256,
                },
            ],
        };
        let image = SrcImage {
            scanline0: src.as_ptr(),
            pitch: pitch as isize,
            plane0_rows,
            len: src.len(),
        };
        // SAFETY: buffers sized above.
        let sig = unsafe { copy_frame(&image, &dest, PixelFormat::Nv12, w, h) }.expect("copy");
        assert_eq!(dst[256 * 3 + 7], 37);
        assert_eq!(dst[1024 + 256 + 7], 215);
        assert_eq!(
            sig.samples.len(),
            FrameSignature::GRID_W * FrameSignature::GRID_H
        );
    }

    #[test]
    fn rejects_short_source() {
        let src = [0u8; 10];
        let mut dst = vec![0u8; 4096];
        let dest = FrameDest {
            slot: 0,
            base: dst.as_mut_ptr(),
            len: dst.len(),
            planes: [
                PlaneDest {
                    offset: 0,
                    pitch: 256,
                },
                PlaneDest {
                    offset: 0,
                    pitch: 0,
                },
            ],
        };
        let image = SrcImage {
            scanline0: src.as_ptr(),
            pitch: 16,
            plane0_rows: 4,
            len: src.len(),
        };
        // SAFETY: function validates lengths before reading.
        let r = unsafe { copy_frame(&image, &dest, PixelFormat::Yuy2, 8, 4) };
        assert!(matches!(r, Err(CopyError::SourceTooSmall { .. })));
    }
}
