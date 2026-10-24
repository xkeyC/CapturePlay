//! Precompiled shaders (see `build.rs`) and pipeline creation.

use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

use crate::gpu::shader_bytecode;

macro_rules! shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".cso"))
    };
}

pub const CONVERT_PLANAR: &[u8] = shader!("convert_planar");
pub const CONVERT_YUYV: &[u8] = shader!("convert_yuyv");
pub const CONVERT_UYVY: &[u8] = shader!("convert_uyvy");
pub const CONVERT_BGRA: &[u8] = shader!("convert_bgra");
pub const OUTPUT_VS: &[u8] = shader!("output_vs");
pub const OUTPUT_PS: &[u8] = shader!("output_ps");
pub const EGUI_VS: &[u8] = shader!("egui_vs");
pub const EGUI_PS: &[u8] = shader!("egui_ps");
pub const VFG_ENCODE: &[u8] = shader!("vfg_encode");
pub const VFG_DECODE: &[u8] = shader!("vfg_decode");
pub const NGX_ENCODE: &[u8] = shader!("ngx_encode");
pub const NGX_DECODE: &[u8] = shader!("ngx_decode");

/// A compute shader whose root signature is embedded in the bytecode.
pub struct ComputePipeline {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl std::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ComputePipeline")
    }
}

impl ComputePipeline {
    pub fn new(device: &ID3D12Device, cs: &[u8]) -> Result<Self> {
        // SAFETY: bytecode contains an embedded root signature (RTS0 part).
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, cs)?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                CS: shader_bytecode(cs),
                ..Default::default()
            };
            let pso = device.CreateComputePipelineState(&desc);
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            Ok(Self { root, pso: pso? })
        }
    }

    /// Binds pipeline + root signature.
    pub fn bind(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: valid objects owned by self.
        unsafe {
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
        }
    }
}

/// Sets root constants (parameter 0) from a `Pod` struct.
pub fn set_compute_constants<T: bytemuck::Pod>(list: &ID3D12GraphicsCommandList, c: &T) {
    let words = std::mem::size_of::<T>() / 4;
    // SAFETY: `c` is plain old data of `words` 32-bit values.
    unsafe { list.SetComputeRoot32BitConstants(0, words as u32, std::ptr::from_ref(c).cast(), 0) };
}

pub fn set_graphics_constants<T: bytemuck::Pod>(list: &ID3D12GraphicsCommandList, c: &T) {
    let words = std::mem::size_of::<T>() / 4;
    // SAFETY: as above.
    unsafe { list.SetGraphicsRoot32BitConstants(0, words as u32, std::ptr::from_ref(c).cast(), 0) };
}

pub fn dispatch_2d(list: &ID3D12GraphicsCommandList, w: u32, h: u32) {
    // SAFETY: plain dispatch; group size 8x8 matches every compute shader.
    unsafe { list.Dispatch(w.div_ceil(8), h.div_ceil(8), 1) };
}
