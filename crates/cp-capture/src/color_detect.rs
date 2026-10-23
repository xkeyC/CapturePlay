//! Resolves the signal's color description from Media Foundation attributes,
//! user overrides and capture-card realities.
//!
//! Capture drivers routinely mislabel HDR: the AVerMedia Live Gamer Ultra 2.1
//! for example tags its P010 modes as BT.709. Consoles only emit 10-bit over
//! HDMI when HDR is on, so 10-bit without an explicit HDR tag is treated as
//! HDR10 by default (see [`ColorSettings::assume_hdr_for_10bit`]).

use cp_core::settings::ColorSettings;
use cp_core::{ColorDesc, PixelFormat, Primaries, Range, TransferFunction, YuvMatrix};

/// What the driver claims; `None` = unknown / not reported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DetectedColor {
    pub transfer: Option<TransferFunction>,
    pub primaries: Option<Primaries>,
    pub matrix: Option<YuvMatrix>,
    pub range: Option<Range>,
}

impl DetectedColor {
    /// Maps raw `MF_MT_TRANSFER_FUNCTION`, `MF_MT_VIDEO_PRIMARIES`,
    /// `MF_MT_YUV_MATRIX` and `MF_MT_VIDEO_NOMINAL_RANGE` values.
    pub fn from_mf(
        transfer: Option<u32>,
        primaries: Option<u32>,
        matrix: Option<u32>,
        range: Option<u32>,
    ) -> Self {
        Self {
            transfer: transfer.and_then(|v| match v {
                5 | 11 | 13 => Some(TransferFunction::Bt1886), // 709, 709_sym, 2020
                4 => Some(TransferFunction::Gamma22),
                7 => Some(TransferFunction::Srgb),
                15 => Some(TransferFunction::Pq),
                16 => Some(TransferFunction::Hlg),
                _ => None,
            }),
            primaries: primaries.and_then(|v| match v {
                2 => Some(Primaries::Bt709),
                9 => Some(Primaries::Bt2020),
                _ => None,
            }),
            matrix: matrix.and_then(|v| match v {
                1 => Some(YuvMatrix::Bt709),
                2 => Some(YuvMatrix::Bt601),
                4 | 5 => Some(YuvMatrix::Bt2020Ncl),
                _ => None,
            }),
            range: range.and_then(|v| match v {
                1 => Some(Range::Full),
                2 => Some(Range::Limited),
                _ => None,
            }),
        }
    }
}

/// Final color description: override > explicit HDR tag > 10-bit heuristic >
/// driver tag > resolution-based default.
pub fn resolve(
    detected: DetectedColor,
    overrides: &ColorSettings,
    format: PixelFormat,
    height: u32,
) -> ColorDesc {
    let detected_hdr = detected.transfer.filter(|t| t.is_hdr());
    let transfer = overrides.transfer.or(detected_hdr).unwrap_or_else(|| {
        if format.bit_depth() > 8 && overrides.assume_hdr_for_10bit {
            TransferFunction::Pq
        } else {
            // SDR game consoles target sRGB-ish displays; BT.1886 tags from
            // capture drivers are boilerplate, so the sRGB EOTF matches what
            // the console author saw.
            TransferFunction::Srgb
        }
    });

    let (matrix, primaries) = if transfer.is_hdr() {
        // HDR on HDMI is always BT.2020; ignore stale 709 tags.
        (YuvMatrix::Bt2020Ncl, Primaries::Bt2020)
    } else {
        let default_matrix = if height >= 720 {
            YuvMatrix::Bt709
        } else {
            YuvMatrix::Bt601
        };
        (
            detected.matrix.unwrap_or(default_matrix),
            detected.primaries.unwrap_or(Primaries::Bt709),
        )
    };

    let default_range = if format.is_yuv() {
        Range::Limited
    } else {
        Range::Full
    };
    ColorDesc {
        matrix: overrides.matrix.unwrap_or(matrix),
        primaries: overrides.primaries.unwrap_or(primaries),
        range: overrides.range.or(detected.range).unwrap_or(default_range),
        transfer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mislabeled_p010_becomes_hdr10() {
        let det = DetectedColor::from_mf(None, Some(2), Some(1), Some(2));
        let c = resolve(det, &ColorSettings::new(), PixelFormat::P010, 1440);
        assert_eq!(c, ColorDesc::HDR10);
    }

    #[test]
    fn nv12_stays_sdr_709() {
        let det = DetectedColor::from_mf(None, Some(2), Some(1), None);
        let c = resolve(det, &ColorSettings::new(), PixelFormat::Nv12, 2160);
        assert_eq!(c, ColorDesc::SDR_709);
    }

    #[test]
    fn explicit_hlg_and_overrides_win() {
        let det = DetectedColor::from_mf(Some(16), None, None, None);
        let c = resolve(det, &ColorSettings::new(), PixelFormat::P010, 2160);
        assert_eq!(c.transfer, TransferFunction::Hlg);
        let ov = ColorSettings {
            transfer: Some(TransferFunction::Srgb),
            range: Some(Range::Full),
            ..ColorSettings::new()
        };
        let c = resolve(det, &ov, PixelFormat::P010, 2160);
        assert_eq!(
            (c.transfer, c.range, c.matrix),
            (TransferFunction::Srgb, Range::Full, YuvMatrix::Bt709)
        );
    }

    #[test]
    fn heuristic_can_be_disabled_and_sd_uses_601() {
        let ov = ColorSettings {
            assume_hdr_for_10bit: false,
            ..ColorSettings::new()
        };
        let c = resolve(DetectedColor::default(), &ov, PixelFormat::P010, 480);
        assert_eq!(
            (c.transfer, c.matrix),
            (TransferFunction::Srgb, YuvMatrix::Bt601)
        );
    }
}
