//! Choosing the upscaling target for the current display area.

/// Scale factors NVIDIA VSR supports.
pub const VSR_FACTORS: [f32; 5] = [4.0 / 3.0, 1.5, 2.0, 3.0, 4.0];

/// Upscaled size for a `src` frame shown in a `display` area (aspect
/// preserved), using the smallest supported factor that covers the needed
/// scale. `None` when the frame already fits (no upscaling needed).
pub fn upscale_target(src: (u32, u32), display: (u32, u32), factors: &[f32]) -> Option<(u32, u32)> {
    if src.0 == 0 || src.1 == 0 {
        return None;
    }
    let need = (display.0 as f32 / src.0 as f32).min(display.1 as f32 / src.1 as f32);
    // Within 2 % of the source size: let the output pass handle it.
    if need <= 1.02 {
        return None;
    }
    let factor = factors
        .iter()
        .copied()
        .find(|&f| f + 1e-3 >= need)
        .or_else(|| {
            factors
                .iter()
                .copied()
                .fold(None, |m: Option<f32>, f| Some(m.map_or(f, |m| m.max(f))))
        })?;
    // Even dimensions keep chroma-friendly sizes and match VSR's outputs.
    let dim = |v: u32| (((v as f32 * factor).round() as u32) / 2) * 2;
    Some((dim(src.0), dim(src.1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_smallest_covering_factor() {
        assert_eq!(
            upscale_target((2560, 1440), (3840, 2160), &VSR_FACTORS),
            Some((3840, 2160))
        );
        assert_eq!(
            upscale_target((1920, 1080), (3840, 2160), &VSR_FACTORS),
            Some((3840, 2160))
        );
        assert_eq!(
            upscale_target((1920, 1080), (2560, 1440), &VSR_FACTORS),
            Some((2560, 1440))
        );
        // Ultrawide 3440x1440 display, 16:9 source: height-limited = 1.0.
        assert_eq!(
            upscale_target((2560, 1440), (3440, 1440), &VSR_FACTORS),
            None
        );
        // Window smaller than the source.
        assert_eq!(
            upscale_target((2560, 1440), (1600, 900), &VSR_FACTORS),
            None
        );
        // Between steps: 1440p in a 4K-ish window of 3200x1800 (1.25x) -> 4/3.
        assert_eq!(
            upscale_target((2560, 1440), (3200, 1800), &VSR_FACTORS),
            Some((3412, 1920))
        );
        // Beyond the largest step: capped at 4x.
        assert_eq!(
            upscale_target((640, 360), (7680, 4320), &VSR_FACTORS),
            Some((2560, 1440))
        );
    }
}
