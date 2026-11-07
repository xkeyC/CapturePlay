//! Final scale + display mapping into the back buffer.

use std::mem::ManuallyDrop;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Result;

use crate::common::OutputParams;
use crate::constants::OutputConstants;
use crate::descriptors::GpuDescriptorRing;
use crate::gpu::{Gpu, shader_bytecode};
use crate::presenter::BACKBUFFER_FORMAT;
use crate::shaders::{self, set_graphics_constants};
use crate::texture::{Texture, flush_barriers};

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

        let c = OutputConstants::new((src.width, src.height), size, p);
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
