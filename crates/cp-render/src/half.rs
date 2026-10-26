//! Half-float conversion for GPU test readbacks (both backends).

pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let mant = x & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    // Round to nearest.
    let m = (mant + 0x1000) >> 13;
    sign | (((exp as u16) << 10) + m as u16)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f32::from(h & 0x3ff);
    if exp == 0 {
        return sign * mant * 2f32.powi(-24);
    }
    if exp == 31 {
        return sign * f32::INFINITY;
    }
    sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
}

#[test]
fn half_float_round_trip() {
    for v in [0.0f32, 0.25, 1.0, 3.5, 12.5, -2.0] {
        assert!((f16_to_f32(f32_to_f16(v)) - v).abs() < 1e-3 * v.abs().max(1.0));
    }
}
