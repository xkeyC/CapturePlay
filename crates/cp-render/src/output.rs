//! Final scale + display mapping into the back buffer.

use std::mem::ManuallyDrop;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Result;

use crate::descriptors::GpuDescriptorRing;
use crate::gpu::{Gpu, shader_bytecode};
use crate::presenter::BACKBUFFER_FORMAT;
use crate::shaders::{self, set_graphics_constants};
use crate::texture::{Texture, flush_barriers};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Constants {
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

/// How the working frame maps onto the current display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputParams {
    pub display_hdr: bool,
    /// Display peak (nits), used when `display_hdr`.
    pub display_peak_nits: f32,
    /// SDR white (nits); the target white on SDR displays.
    pub sdr_white_nits: f32,
    /// Mastering peak of the content (nits).
    pub content_peak_nits: f32,
    pub catmull_rom: bool,
}

/// Aspect-preserving fit of `src` into `dst`: `(x, y, w, h)` in pixels.
pub fn letterbox(src: (u32, u32), dst: (u32, u32)) -> (f32, f32, f32, f32) {
    let (sw, sh) = (src.0.max(1) as f32, src.1.max(1) as f32);
    let (dw, dh) = (dst.0.max(1) as f32, dst.1.max(1) as f32);
    let s = (dw / sw).min(dh / sh);
    let (w, h) = ((sw * s).round(), (sh * s).round());
    (((dw - w) * 0.5).floor(), ((dh - h) * 0.5).floor(), w, h)
}

pub struct OutputPass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl std::fmt::Debug for OutputPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OutputPass")
    }
}

impl OutputPass {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        // SAFETY: PSO creation from embedded-root-signature bytecode.
        unsafe {
            let root: ID3D12RootSignature =
                gpu.device.CreateRootSignature(0, shaders::OUTPUT_PS)?;
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                VS: shader_bytecode(shaders::OUTPUT_VS),
                PS: shader_bytecode(shaders::OUTPUT_PS),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            desc.RTVFormats[0] = BACKBUFFER_FORMAT;
            let pso = gpu.device.CreateGraphicsPipelineState(&desc);
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            Ok(Self { root, pso: pso? })
        }
    }

    /// Draws `src` letterboxed into the bound render target of `size`.
    /// A `None` source clears to black.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        gpu: &Gpu,
        ring: &mut GpuDescriptorRing,
        list: &ID3D12GraphicsCommandList,
        src: Option<&Texture>,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        size: (u32, u32),
        p: &OutputParams,
    ) {
        // SAFETY: valid list and RTV handle.
        unsafe {
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size.0 as f32,
                Height: size.1 as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: size.0 as i32,
                bottom: size.1 as i32,
            }]);
        }
        let Some(src) = src else {
            // SAFETY: valid RTV.
            unsafe { list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None) };
            return;
        };
        let mut barriers = Vec::new();
        src.transition(&mut barriers, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        flush_barriers(list, &mut barriers);

        let (x, y, w, h) = letterbox((src.width, src.height), size);
        let (target_peak, out_scale) = if p.display_hdr {
            (p.display_peak_nits, 1.0 / cp_core::color::SCRGB_UNIT_NITS)
        } else {
            (p.sdr_white_nits, 1.0 / p.sdr_white_nits)
        };
        let c = Constants {
            uv_scale: [1.0 / w, 1.0 / h],
            uv_offset: [-x / w, -y / h],
            src_size: [src.width as f32, src.height as f32],
            inv_src_size: [1.0 / src.width as f32, 1.0 / src.height as f32],
            sdr_display: u32::from(!p.display_hdr),
            target_peak,
            src_peak: p.content_peak_nits,
            out_scale,
            catmull_rom: u32::from(
                p.catmull_rom && (w as u32 != src.width || h as u32 != src.height),
            ),
            pad: [0.0; 3],
        };
        let table = ring.table(&gpu.device, &[src.srv.handle]);
        // SAFETY: valid pipeline objects and descriptor table.
        unsafe {
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            set_graphics_constants(list, &c);
            list.SetGraphicsRootDescriptorTable(1, table);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::letterbox;

    #[test]
    fn letterbox_fits_and_centers() {
        assert_eq!(
            letterbox((1920, 1080), (3840, 2160)),
            (0.0, 0.0, 3840.0, 2160.0)
        );
        assert_eq!(
            letterbox((1920, 1080), (3440, 1440)),
            (440.0, 0.0, 2560.0, 1440.0)
        );
        assert_eq!(
            letterbox((1920, 1080), (1920, 1200)),
            (0.0, 60.0, 1920.0, 1080.0)
        );
    }
}
