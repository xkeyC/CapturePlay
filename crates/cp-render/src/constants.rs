//! Constant blocks of the shaders, shared by both backends: D3D12 binds them
//! as root constants, Vulkan as uniform buffers (the SPIR-V is compiled with
//! `-fvk-use-dx-layout`, so the D3D cbuffer layout holds for both). What
//! goes into them is computed here once; the backends only bind.

use cp_core::color::{TransferFunction, yuv_to_rgb_matrix};
use cp_core::{ColorDesc, Primaries, VideoMode};

use crate::common::{DecodeParams, OutputParams, letterbox};

/// `convert.hlsl` (24 dwords).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ConvertConstants {
    rows: [[f32; 4]; 3],
    transfer: u32,
    primaries_2020: u32,
    sdr_white: f32,
    hlg_peak: f32,
    hlg_gamma: f32,
    width: u32,
    height: u32,
    hdr_scale: f32,
    chroma_texel: [f32; 2],
    pad1: [f32; 2],
}

impl ConvertConstants {
    /// Decoding `mode` frames described by `color`; `chroma` is the size of
    /// the chroma plane (the luma plane for packed formats).
    pub fn new(
        mode: &VideoMode,
        color: &ColorDesc,
        params: DecodeParams,
        chroma: (u32, u32),
    ) -> Self {
        Self {
            rows: yuv_to_rgb_matrix(
                color.matrix,
                color.range,
                mode.format.bit_depth(),
                mode.format.storage_bits(),
            ),
            transfer: match color.transfer {
                TransferFunction::Srgb => 0,
                TransferFunction::Gamma22 => 1,
                TransferFunction::Bt1886 => 2,
                TransferFunction::Pq => 3,
                TransferFunction::Hlg => 4,
            },
            primaries_2020: u32::from(color.primaries == Primaries::Bt2020),
            sdr_white: params.sdr_white_nits / cp_core::color::SCRGB_UNIT_NITS,
            hlg_peak: params.hlg_peak_nits,
            hlg_gamma: 1.2 + 0.42 * (params.hlg_peak_nits / 1000.0).log10(),
            width: mode.width,
            height: mode.height,
            hdr_scale: params.hdr_scale,
            chroma_texel: [1.0 / chroma.0 as f32, 1.0 / chroma.1 as f32],
            pad1: [0.0; 2],
        }
    }
}

/// `output.hlsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OutputConstants {
    uv_scale: [f32; 2],
    uv_offset: [f32; 2],
    src_size: [f32; 2],
    inv_src_size: [f32; 2],
    sdr_display: u32,
    target_peak: f32,
    src_peak: f32,
    out_scale: f32,
    catmull_rom: u32,
    pad: [f32; 3],
}

impl OutputConstants {
    /// `src` letterboxed into a target of `size`, mapped as `p` says.
    pub fn new(src: (u32, u32), size: (u32, u32), p: &OutputParams) -> Self {
        let (x, y, w, h) = letterbox(src, size);
        let (target_peak, out_scale) = if p.display_hdr {
            (p.display_peak_nits, 1.0 / cp_core::color::SCRGB_UNIT_NITS)
        } else {
            (p.sdr_white_nits, 1.0 / p.sdr_white_nits)
        };
        Self {
            uv_scale: [1.0 / w, 1.0 / h],
            uv_offset: [-x / w, -y / h],
            src_size: [src.0 as f32, src.1 as f32],
            inv_src_size: [1.0 / src.0 as f32, 1.0 / src.1 as f32],
            sdr_display: u32::from(!p.display_hdr),
            target_peak,
            src_peak: p.content_peak_nits,
            out_scale,
            catmull_rom: u32::from(p.catmull_rom && (w as u32 != src.0 || h as u32 != src.1)),
            pad: [0.0; 3],
        }
    }
}

/// `egui.hlsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct EguiConstants {
    screen_size_points: [f32; 2],
    white_scale: f32,
    pad: f32,
}

impl EguiConstants {
    pub fn new(size_px: (u32, u32), pixels_per_point: f32, white_scale: f32) -> Self {
        Self {
            screen_size_points: [
                size_px.0 as f32 / pixels_per_point,
                size_px.1 as f32 / pixels_per_point,
            ],
            white_scale,
            pad: 0.0,
        }
    }
}

/// `vfg.hlsl`, the packed encodings of the VFG interpolator and the VSR
/// upscaler.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackConstants {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    /// 1 = PQ / BT.2020 encoding, 0 = sRGB relative to `sdr_white`.
    pub hdr: u32,
    pub sdr_white: f32,
    pub pad: [f32; 3],
}

/// `nr.hlsl` encode (working frame → sRGB proxy).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NrEncodeConstants {
    pub width: u32,
    pub height: u32,
    pub hdr: u32,
    pub sdr_white: f32,
}

/// `nr.hlsl` composite (the model's edit over the working frame).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NrCompositeConstants {
    pub width: u32,
    pub height: u32,
    pub hdr: u32,
    pub sdr_white: f32,
    pub detail: f32,
    pub colour: f32,
    pub _pad: [f32; 2],
}

/// `nr.hlsl` guide fills.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NrFillConstants {
    pub width: u32,
    pub height: u32,
    pub value: f32,
    pub _pad: u32,
}
