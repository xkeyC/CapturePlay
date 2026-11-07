//! The DLSSNR (NGX feature 18) parameters: one list for every caller — the
//! D3D12 binding on Windows (`cp-nvidia`) and the Vulkan path of the Proton
//! host (`nr-host`). Callers pass a setter for their parameter block; the
//! resources (`DLSSNR.Color`, `.Depth`, `.MVec`, `.Output`) they set
//! themselves, their types differ per API.

use std::ffi::CStr;

/// A parameter value, typed as NGX's `Set` overloads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    U32(u32),
    I32(i32),
    F32(f32),
}

/// Live model tuning. These keys are read at evaluation time, so changing
/// them does not recreate the feature (the model preset does).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tuning {
    /// DLSSNR.Style: 0 default, 1 natural, 2 cinematic.
    pub style: u32,
    /// DLSSNR.Intensity: overall strength of the model's edit.
    pub intensity: f32,
    /// DLSSNR.LocalStructureStrength.
    pub local_structure: f32,
    /// DLSSNR.LocalToneStrength.
    pub local_tone: f32,
    /// DLSSNR.SkinStructureStrength.
    pub skin_structure: f32,
    /// DLSSNR.UseAutoMask.
    pub auto_mask: bool,
    /// DLSSNR.UICorrection.
    pub ui_correction: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            style: 0,
            intensity: 1.0,
            local_structure: 1.0,
            local_tone: 1.0,
            skin_structure: 0.0,
            auto_mask: false,
            ui_correction: false,
        }
    }
}

/// The four keys of one resource's subrect.
struct Subrect {
    base_x: &'static CStr,
    base_y: &'static CStr,
    width: &'static CStr,
    height: &'static CStr,
}

const COLOR_SUBRECT: Subrect = Subrect {
    base_x: c"DLSSNR.ColorSubrectBaseX",
    base_y: c"DLSSNR.ColorSubrectBaseY",
    width: c"DLSSNR.ColorSubrectWidth",
    height: c"DLSSNR.ColorSubrectHeight",
};
const DEPTH_SUBRECT: Subrect = Subrect {
    base_x: c"DLSSNR.DepthSubrectBaseX",
    base_y: c"DLSSNR.DepthSubrectBaseY",
    width: c"DLSSNR.DepthSubrectWidth",
    height: c"DLSSNR.DepthSubrectHeight",
};
const MVEC_SUBRECT: Subrect = Subrect {
    base_x: c"DLSSNR.MVecSubrectBaseX",
    base_y: c"DLSSNR.MVecSubrectBaseY",
    width: c"DLSSNR.MVecSubrectWidth",
    height: c"DLSSNR.MVecSubrectHeight",
};
const OUTPUT_SUBRECT: Subrect = Subrect {
    base_x: c"DLSSNR.OutputSubrectBaseX",
    base_y: c"DLSSNR.OutputSubrectBaseY",
    width: c"DLSSNR.OutputSubrectWidth",
    height: c"DLSSNR.OutputSubrectHeight",
};

/// Creation parameters for a feature of `width × height` (no upscaling).
/// `preset` is `DLSSNR.Hint.Render.Preset` (0 auto, 1 quality, 2
/// performance on the 310.9 builds).
pub fn create(width: u32, height: u32, preset: u32, set: &mut impl FnMut(&CStr, Value)) {
    use Value::{F32, I32, U32};
    set(c"CreationNodeMask", U32(1));
    set(c"VisibilityNodeMask", U32(1));
    // Neither the generic nor the DLSSNR-prefixed size keys are documented,
    // so both are set.
    for key in [
        c"Width",
        c"OutWidth",
        c"DLSSNR.InputWidth",
        c"DLSSNR.OutputWidth",
        c"DLSSNR.Width",
    ] {
        set(key, U32(width));
    }
    for key in [
        c"Height",
        c"OutHeight",
        c"DLSSNR.InputHeight",
        c"DLSSNR.OutputHeight",
        c"DLSSNR.Height",
    ] {
        set(key, U32(height));
    }
    set(c"DLSSNR.Enabled", I32(1));
    set(c"DLSSNR.Upscaling", I32(0));
    set(c"DLSSNR.Scale", F32(1.0));
    set(c"DLSSNR.ScalingRatio", F32(1.0));
    set(c"DLSSNR.Hint.Render.Preset", I32(preset as i32));
}

/// Per-evaluation parameters: the color, depth and motion guides are
/// `input` sized, the output `output` sized, all at the origin.
pub fn evaluate(
    input: (u32, u32),
    output: (u32, u32),
    reset: bool,
    t: &Tuning,
    set: &mut impl FnMut(&CStr, Value),
) {
    use Value::{F32, I32, U32};
    set(c"DLSSNR.Enabled", I32(1));
    set(c"DLSSNR.Reset", I32(i32::from(reset)));
    set(c"DLSSNR.DepthInverted", I32(0));
    for (keys, size) in [
        (COLOR_SUBRECT, input),
        (DEPTH_SUBRECT, input),
        (MVEC_SUBRECT, input),
        (OUTPUT_SUBRECT, output),
    ] {
        set(keys.base_x, U32(0));
        set(keys.base_y, U32(0));
        set(keys.width, U32(size.0));
        set(keys.height, U32(size.1));
    }
    set(c"DLSSNR.MVecScaleX", F32(1.0));
    set(c"DLSSNR.MVecScaleY", F32(1.0));
    set(c"DLSSNR.Style", U32(t.style));
    set(c"DLSSNR.Intensity", F32(t.intensity));
    set(c"DLSSNR.LocalStructureStrength", F32(t.local_structure));
    set(c"DLSSNR.LocalToneStrength", F32(t.local_tone));
    set(c"DLSSNR.SkinStructureStrength", F32(t.skin_structure));
    set(c"DLSSNR.UseAutoMask", U32(u32::from(t.auto_mask)));
    set(c"DLSSNR.UICorrection", U32(u32::from(t.ui_correction)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subrects_follow_their_sizes() {
        let mut got = Vec::new();
        evaluate(
            (640, 360),
            (1280, 720),
            true,
            &Tuning::default(),
            &mut |k, v| {
                got.push((k.to_owned(), v));
            },
        );
        let find = |k: &CStr| got.iter().find(|(n, _)| n.as_c_str() == k).map(|(_, v)| *v);
        assert_eq!(find(c"DLSSNR.DepthSubrectWidth"), Some(Value::U32(640)));
        assert_eq!(find(c"DLSSNR.OutputSubrectHeight"), Some(Value::U32(720)));
        assert_eq!(find(c"DLSSNR.Reset"), Some(Value::I32(1)));
        // Enabled, Reset, DepthInverted, 4 × 4 subrect keys, 2 motion
        // scales and 7 tuning keys.
        assert_eq!(got.len(), 28);
    }
}
