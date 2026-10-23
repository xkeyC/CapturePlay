//! Color description and reference color math.
//!
//! Working space convention used throughout the renderer: **linear scRGB**,
//! i.e. linear light with BT.709 primaries where `1.0 == 80 nits`. Values
//! outside `[0, 1]` (wide gamut / HDR highlights) are legal.
//!
//! `shaders/color.hlsli` mirrors the functions in this module; the unit tests
//! here are the reference values for the shader implementation.

use serde::{Deserialize, Serialize};

/// Nits represented by `1.0` in scRGB.
pub const SCRGB_UNIT_NITS: f32 = 80.0;

/// Opto-electronic transfer characteristic of the incoming signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TransferFunction {
    /// Piecewise sRGB EOTF (what Windows assumes for SDR content).
    #[default]
    Srgb,
    /// Pure power 2.2.
    Gamma22,
    /// BT.1886 (power 2.4, reference display).
    Bt1886,
    /// SMPTE ST 2084 perceptual quantizer (HDR10).
    Pq,
    /// ARIB STD-B67 / BT.2100 hybrid log-gamma.
    Hlg,
}

impl TransferFunction {
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }
}

/// Color primaries of the signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Primaries {
    #[default]
    Bt709,
    Bt2020,
}

/// Y'CbCr → R'G'B' matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum YuvMatrix {
    Bt601,
    #[default]
    Bt709,
    /// BT.2020 non-constant luminance.
    Bt2020Ncl,
}

impl YuvMatrix {
    /// `(Kr, Kb)` luma coefficients.
    pub fn kr_kb(self) -> (f64, f64) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt2020Ncl => (0.2627, 0.0593),
        }
    }
}

/// Quantization range of the code values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Range {
    /// "TV" / studio swing: Y 16–235, C 16–240 (8-bit).
    #[default]
    Limited,
    /// "PC" / full swing.
    Full,
}

/// Complete description of how to interpret a decoded signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ColorDesc {
    pub matrix: YuvMatrix,
    pub range: Range,
    pub transfer: TransferFunction,
    pub primaries: Primaries,
}

impl ColorDesc {
    /// Typical SDR HDMI signal from a console.
    pub const SDR_709: Self = Self {
        matrix: YuvMatrix::Bt709,
        range: Range::Limited,
        transfer: TransferFunction::Srgb,
        primaries: Primaries::Bt709,
    };
    /// HDR10 signal (PS5 / Xbox HDR output).
    pub const HDR10: Self = Self {
        matrix: YuvMatrix::Bt2020Ncl,
        range: Range::Limited,
        transfer: TransferFunction::Pq,
        primaries: Primaries::Bt2020,
    };
    /// HLG broadcast signal.
    pub const HLG: Self = Self {
        matrix: YuvMatrix::Bt2020Ncl,
        range: Range::Limited,
        transfer: TransferFunction::Hlg,
        primaries: Primaries::Bt2020,
    };

    pub fn is_hdr(&self) -> bool {
        self.transfer.is_hdr()
    }
}

/// Builds the affine 3×4 matrix that maps *normalized texture samples*
/// `(y, cb, cr, 1)` straight to non-linear `R'G'B'` in `[0, 1]`.
///
/// `bits` is the significant bit depth of the code values and `storage_bits`
/// the bit width of the UNORM texture channel they are sampled from. P010 for
/// example stores 10-bit codes in the MSBs of 16-bit words (`bits = 10`,
/// `storage_bits = 16`), so a sample `v` corresponds to code
/// `v * 65535 / 64`.
pub fn yuv_to_rgb_matrix(
    matrix: YuvMatrix,
    range: Range,
    bits: u32,
    storage_bits: u32,
) -> [[f32; 4]; 3] {
    assert!(bits >= 8 && bits <= storage_bits && storage_bits <= 16);
    let storage_max = f64::from((1u32 << storage_bits) - 1);
    // Multiplier turning a normalized sample into a code value of `bits` depth.
    let to_code = storage_max / f64::from(1u32 << (storage_bits - bits));
    let scale = f64::from(1u32 << (bits - 8));

    // y_norm = code * ys - yo ; c_norm = code * cs - co (chroma centered on 0)
    let (ys, yo, cs, co) = match range {
        Range::Limited => {
            let ys = 1.0 / (219.0 * scale);
            let cs = 1.0 / (224.0 * scale);
            (ys, 16.0 * scale * ys, cs, 128.0 * scale * cs)
        }
        Range::Full => {
            let max = f64::from((1u32 << bits) - 1);
            (1.0 / max, 0.0, 1.0 / max, 128.0 * scale / max)
        }
    };

    let (kr, kb) = matrix.kr_kb();
    let kg = 1.0 - kr - kb;
    // R' = Y + a*Cr ; G' = Y - b*Cb - c*Cr ; B' = Y + d*Cb
    let a = 2.0 * (1.0 - kr);
    let d = 2.0 * (1.0 - kb);
    let b = d * kb / kg;
    let c = a * kr / kg;
    let rows = [[1.0, 0.0, a], [1.0, -b, -c], [1.0, d, 0.0]];

    let mut out = [[0f32; 4]; 3];
    for (row, coef) in out.iter_mut().zip(rows) {
        let [cy, ccb, ccr] = coef;
        row[0] = (cy * ys * to_code) as f32;
        row[1] = (ccb * cs * to_code) as f32;
        row[2] = (ccr * cs * to_code) as f32;
        row[3] = (-cy * yo - ccb * co - ccr * co) as f32;
    }
    out
}

/// Row-major 3×3 matrix converting linear BT.2020 RGB to linear BT.709 RGB.
pub const BT2020_TO_BT709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641, -0.072_850],
    [-0.124_550, 1.132_9, -0.008_349],
    [-0.018_151, -0.100_579, 1.118_73],
];

/// Row-major 3×3 matrix converting linear BT.709 RGB to linear BT.2020 RGB.
pub const BT709_TO_BT2020: [[f32; 3]; 3] = [
    [0.627_404, 0.329_283, 0.043_313],
    [0.069_097, 0.919_540, 0.011_362],
    [0.016_391, 0.088_013, 0.895_595],
];

pub fn mul3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

mod pq {
    pub const M1: f32 = 2610.0 / 16384.0;
    pub const M2: f32 = 2523.0 / 4096.0 * 128.0;
    pub const C1: f32 = 3424.0 / 4096.0;
    pub const C2: f32 = 2413.0 / 4096.0 * 32.0;
    pub const C3: f32 = 2392.0 / 4096.0 * 32.0;
}

/// ST 2084 EOTF: signal `[0,1]` → absolute luminance in nits.
pub fn pq_to_nits(e: f32) -> f32 {
    let p = e.clamp(0.0, 1.0).powf(1.0 / pq::M2);
    let num = (p - pq::C1).max(0.0);
    let den = pq::C2 - pq::C3 * p;
    10_000.0 * (num / den).powf(1.0 / pq::M1)
}

/// ST 2084 inverse EOTF: absolute nits → signal `[0,1]`.
pub fn nits_to_pq(nits: f32) -> f32 {
    let y = (nits / 10_000.0).clamp(0.0, 1.0).powf(pq::M1);
    ((pq::C1 + pq::C2 * y) / (1.0 + pq::C3 * y)).powf(pq::M2)
}

const HLG_A: f32 = 0.178_832_77;
const HLG_B: f32 = 0.284_668_92;
const HLG_C: f32 = 0.559_910_7;

/// HLG inverse OETF: signal → normalized scene light `[0,1]`.
pub fn hlg_to_scene(e: f32) -> f32 {
    let e = e.clamp(0.0, 1.0);
    if e <= 0.5 {
        e * e / 3.0
    } else {
        (((e - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// HLG OOTF for a display with peak `lw` nits, applied to linear BT.2020
/// scene light. Returns display light in nits.
pub fn hlg_ootf(rgb: [f32; 3], lw: f32) -> [f32; 3] {
    let gamma = 1.2 + 0.42 * (lw / 1000.0).log10();
    let ys = 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2];
    let k = lw * ys.max(1e-6).powf(gamma - 1.0);
    rgb.map(|c| c * k)
}

/// Piecewise sRGB EOTF.
pub fn srgb_to_linear(e: f32) -> f32 {
    if e <= 0.040_45 {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// Piecewise sRGB inverse EOTF.
pub fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// BT.2390 EETF luminance compression performed in the PQ domain.
///
/// Maps `nits` mastered for a `source_peak` display onto a display with
/// `target_peak`. Identity when the target can show the full source range.
pub fn bt2390_eetf(nits: f32, source_peak: f32, target_peak: f32) -> f32 {
    if target_peak >= source_peak {
        return nits.min(source_peak);
    }
    let src_max = nits_to_pq(source_peak);
    let e1 = nits_to_pq(nits) / src_max;
    let max_lum = nits_to_pq(target_peak) / src_max;
    let ks = 1.5 * max_lum - 0.5;
    let e2 = if e1 < ks {
        e1
    } else {
        let t = ((e1 - ks) / (1.0 - ks)).min(1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        (2.0 * t3 - 3.0 * t2 + 1.0) * ks
            + (t3 - 2.0 * t2 + t) * (1.0 - ks)
            + (-2.0 * t3 + 3.0 * t2) * max_lum
    };
    pq_to_nits(e2 * src_max)
}

/// Decodes one non-linear R'G'B' triple (as produced by
/// [`yuv_to_rgb_matrix`]) into linear scRGB, for an SDR signal shown at
/// `sdr_white_nits`. Reference implementation of the shader path.
pub fn decode_to_scrgb(
    rgb: [f32; 3],
    desc: &ColorDesc,
    sdr_white_nits: f32,
    hlg_peak_nits: f32,
) -> [f32; 3] {
    let nits = match desc.transfer {
        TransferFunction::Pq => rgb.map(pq_to_nits),
        TransferFunction::Hlg => hlg_ootf(rgb.map(hlg_to_scene), hlg_peak_nits),
        TransferFunction::Srgb => rgb.map(|c| srgb_to_linear(c.clamp(0.0, 1.0)) * sdr_white_nits),
        TransferFunction::Gamma22 => rgb.map(|c| c.clamp(0.0, 1.0).powf(2.2) * sdr_white_nits),
        TransferFunction::Bt1886 => rgb.map(|c| c.clamp(0.0, 1.0).powf(2.4) * sdr_white_nits),
    };
    let lin709 = match desc.primaries {
        Primaries::Bt709 => nits,
        Primaries::Bt2020 => mul3(&BT2020_TO_BT709, nits),
    };
    lin709.map(|c| c / SCRGB_UNIT_NITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(m: &[[f32; 4]; 3], v: [f32; 3]) -> [f32; 3] {
        let f = |r: &[f32; 4]| r[0] * v[0] + r[1] * v[1] + r[2] * v[2] + r[3];
        [f(&m[0]), f(&m[1]), f(&m[2])]
    }

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn limited_8bit_black_and_white() {
        let m = yuv_to_rgb_matrix(YuvMatrix::Bt709, Range::Limited, 8, 8);
        let black = apply(&m, [16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]);
        let white = apply(&m, [235.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]);
        for c in black {
            assert!(close(c, 0.0, 1e-5), "{black:?}");
        }
        for c in white {
            assert!(close(c, 1.0, 1e-5), "{white:?}");
        }
    }

    #[test]
    fn p010_limited_white_and_red() {
        // P010 stores the 10-bit code in the top bits of a 16-bit word.
        let s = |code: u32| (code << 6) as f32 / 65535.0;
        let m = yuv_to_rgb_matrix(YuvMatrix::Bt2020Ncl, Range::Limited, 10, 16);
        let white = apply(&m, [s(940), s(512), s(512)]);
        for c in white {
            assert!(close(c, 1.0, 1e-4), "{white:?}");
        }
        // Pure BT.2020 red: Y = Kr, Cb = -Kr/2/(1-Kb), Cr = 0.5
        let (kr, kb) = (0.2627f64, 0.0593f64);
        let y = 64.0 + 876.0 * kr;
        let cb = 512.0 + 896.0 * (-kr / (2.0 * (1.0 - kb)));
        let cr: f64 = 512.0 + 896.0 * 0.5;
        let red = apply(
            &m,
            [
                s(y.round() as u32),
                s(cb.round() as u32),
                s(cr.round() as u32),
            ],
        );
        assert!(
            close(red[0], 1.0, 2e-3) && close(red[1], 0.0, 2e-3) && close(red[2], 0.0, 2e-3),
            "{red:?}"
        );
    }

    #[test]
    fn full_range_8bit() {
        let m = yuv_to_rgb_matrix(YuvMatrix::Bt601, Range::Full, 8, 8);
        let white = apply(&m, [1.0, 128.0 / 255.0, 128.0 / 255.0]);
        for c in white {
            assert!(close(c, 1.0, 1e-5), "{white:?}");
        }
    }

    #[test]
    fn pq_round_trip_and_reference_points() {
        assert!(close(pq_to_nits(1.0), 10_000.0, 0.5));
        assert!(close(pq_to_nits(0.0), 0.0, 1e-6));
        // 100 nits ≈ 0.5081 PQ, 1000 nits ≈ 0.7518 PQ.
        assert!(close(nits_to_pq(100.0), 0.5081, 1e-3));
        assert!(close(nits_to_pq(1000.0), 0.7518, 1e-3));
        for nits in [0.1f32, 1.0, 80.0, 203.0, 600.0, 4000.0] {
            let rt = pq_to_nits(nits_to_pq(nits));
            assert!(close(rt, nits, nits * 1e-3), "{nits} -> {rt}");
        }
    }

    #[test]
    fn hlg_reference_white() {
        // HLG reference white (75% signal) on a 1000-nit display is 203 nits.
        let e = hlg_to_scene(0.75);
        let out = hlg_ootf([e, e, e], 1000.0);
        assert!(close(out[0], 203.0, 2.0), "{out:?}");
    }

    #[test]
    fn srgb_round_trip() {
        for v in [0.0f32, 0.01, 0.2, 0.5, 0.9, 1.0] {
            assert!(close(linear_to_srgb(srgb_to_linear(v)), v, 1e-5));
        }
    }

    #[test]
    fn eetf_is_identity_below_knee_and_bounded() {
        assert!(close(bt2390_eetf(100.0, 1000.0, 600.0), 100.0, 0.5));
        assert!(bt2390_eetf(1000.0, 1000.0, 600.0) <= 600.5);
        assert!(close(bt2390_eetf(4000.0, 1000.0, 1500.0), 1000.0, 1e-3));
        let mut prev = 0.0;
        for i in 0..=100 {
            let v = bt2390_eetf(i as f32 * 10.0, 1000.0, 400.0);
            assert!(v + 1e-3 >= prev, "EETF must be monotonic");
            prev = v;
        }
    }

    #[test]
    fn primaries_matrices_are_inverse() {
        let v = [0.3, 0.6, 0.1];
        let rt = mul3(&BT2020_TO_BT709, mul3(&BT709_TO_BT2020, v));
        for (a, b) in rt.iter().zip(v) {
            assert!(close(*a, b, 1e-4));
        }
    }

    #[test]
    fn sdr_white_maps_to_paper_white() {
        let out = decode_to_scrgb([1.0; 3], &ColorDesc::SDR_709, 240.0, 1000.0);
        for c in out {
            assert!(close(c, 3.0, 1e-5));
        }
    }
}
