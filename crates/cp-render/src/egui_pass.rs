//! Minimal egui renderer for D3D12, drawing into the scRGB back buffer.

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use egui::epaint::{ImageDelta, Primitive, Vertex};
use egui::{ClippedPrimitive, ImageData, TextureId, TexturesDelta};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Result, s};

use crate::arena::UploadArena;
use crate::descriptors::{GpuDescriptorRing, StagingHeap};
use crate::gpu::{Gpu, borrow_com, shader_bytecode};
use crate::presenter::BACKBUFFER_FORMAT;
use crate::shaders::{self, set_graphics_constants};
use crate::texture::{Texture, TextureDesc, flush_barriers};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Constants {
    screen_size_points: [f32; 2],
    white_scale: f32,
    pad: f32,
}

pub struct EguiPass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    textures: HashMap<TextureId, Texture>,
}

impl std::fmt::Debug for EguiPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EguiPass")
            .field("textures", &self.textures.len())
            .finish()
    }
}

/// Everything needed to draw one egui frame.
#[derive(Debug)]
pub struct EguiFrame<'a> {
    pub primitives: &'a [ClippedPrimitive],
    pub textures_delta: &'a TexturesDelta,
    pub pixels_per_point: f32,
    /// UI white in output units (scRGB on HDR displays, 1.0 on SDR).
    pub white_scale: f32,
}

impl EguiPass {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let layout = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32_FLOAT,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("TEXCOORD"),
                Format: DXGI_FORMAT_R32G32_FLOAT,
                AlignedByteOffset: 8,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("COLOR"),
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                AlignedByteOffset: 16,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
        ];
        // SAFETY: PSO creation from embedded-root-signature bytecode;
        // `layout` outlives the call.
        unsafe {
            let root: ID3D12RootSignature = gpu.device.CreateRootSignature(0, shaders::EGUI_VS)?;
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                VS: shader_bytecode(shaders::EGUI_VS),
                PS: shader_bytecode(shaders::EGUI_PS),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: layout.as_ptr(),
                    NumElements: layout.len() as u32,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let rt = &mut desc.BlendState.RenderTarget[0];
            rt.BlendEnable = true.into();
            rt.SrcBlend = D3D12_BLEND_ONE;
            rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
            rt.BlendOp = D3D12_BLEND_OP_ADD;
            rt.SrcBlendAlpha = D3D12_BLEND_INV_DEST_ALPHA;
            rt.DestBlendAlpha = D3D12_BLEND_ONE;
            rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            desc.RTVFormats[0] = BACKBUFFER_FORMAT;
            let pso = gpu.device.CreateGraphicsPipelineState(&desc);
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            Ok(Self {
                root,
                pso: pso?,
                textures: HashMap::new(),
            })
        }
    }

    fn upload_image(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        arena: &mut UploadArena,
        list: &ID3D12GraphicsCommandList,
        id: TextureId,
        delta: &ImageDelta,
    ) -> Result<()> {
        let ImageData::Color(img) = &delta.image;
        let (w, h) = (img.size[0] as u32, img.size[1] as u32);
        if w == 0 || h == 0 {
            return Ok(());
        }
        if delta.pos.is_none() {
            let mut d = TextureDesc::new(w, h, DXGI_FORMAT_R8G8B8A8_UNORM);
            d.srv_format = Some(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
            self.textures.insert(id, Texture::new(gpu, heap, d)?);
        }
        let Some(tex) = self.textures.get(&id) else {
            return Ok(());
        };
        let pitch = (w as usize * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize);
        let Some(a) = arena.alloc(
            pitch * h as usize,
            D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as usize,
        ) else {
            tracing::warn!("egui texture upload does not fit the frame arena");
            return Ok(());
        };
        for (y, row) in img.pixels.chunks(w as usize).enumerate() {
            let bytes: &[u8] = bytemuck::cast_slice(row);
            // SAFETY: row fits within the reserved arena region.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), a.ptr.add(y * pitch), bytes.len())
            };
        }
        let mut barriers = Vec::new();
        tex.transition(&mut barriers, D3D12_RESOURCE_STATE_COPY_DEST);
        flush_barriers(list, &mut barriers);
        let [x, y] = delta.pos.map_or([0, 0], |p| [p[0] as u32, p[1] as u32]);
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrow_com(&tex.resource),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrow_com(&arena.buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: a.offset as u64,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: w,
                        Height: h,
                        Depth: 1,
                        RowPitch: pitch as u32,
                    },
                },
            },
        };
        // SAFETY: locations reference live resources; region is in bounds.
        unsafe { list.CopyTextureRegion(&dst, x, y, 0, &src, None) };
        tex.transition(&mut barriers, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        flush_barriers(list, &mut barriers);
        Ok(())
    }

    /// Applies texture updates and draws the primitives on the bound target.
    /// Freed textures are released after the GPU is done (call
    /// [`EguiPass::free_textures`] once this frame's fence has passed).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        ring: &mut GpuDescriptorRing,
        arena: &mut UploadArena,
        list: &ID3D12GraphicsCommandList,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        size_px: (u32, u32),
        frame: &EguiFrame<'_>,
    ) -> Result<()> {
        for (id, deltas) in &frame.textures_delta.set {
            for delta in deltas {
                self.upload_image(gpu, heap, arena, list, *id, delta)?;
            }
        }
        if frame.primitives.is_empty() {
            return Ok(());
        }
        let ppp = frame.pixels_per_point;
        let c = Constants {
            screen_size_points: [size_px.0 as f32 / ppp, size_px.1 as f32 / ppp],
            white_scale: frame.white_scale,
            pad: 0.0,
        };
        // SAFETY: valid pipeline and render target.
        unsafe {
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size_px.0 as f32,
                Height: size_px.1 as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        set_graphics_constants(list, &c);

        for prim in frame.primitives {
            let Primitive::Mesh(mesh) = &prim.primitive else {
                continue;
            };
            if mesh.indices.is_empty() {
                continue;
            }
            let Some(tex) = self.textures.get(&mesh.texture_id) else {
                continue;
            };
            let clip = prim.clip_rect;
            let left = (clip.min.x * ppp).round().clamp(0.0, size_px.0 as f32) as i32;
            let top = (clip.min.y * ppp).round().clamp(0.0, size_px.1 as f32) as i32;
            let right = (clip.max.x * ppp).round().clamp(0.0, size_px.0 as f32) as i32;
            let bottom = (clip.max.y * ppp).round().clamp(0.0, size_px.1 as f32) as i32;
            if right <= left || bottom <= top {
                continue;
            }
            let (Some(vb), Some(ib)) = (
                arena.push::<Vertex>(&mesh.vertices, 16),
                arena.push::<u32>(&mesh.indices, 16),
            ) else {
                tracing::warn!("egui geometry does not fit the frame arena");
                break;
            };
            let table = ring.table(&gpu.device, &[tex.srv.handle]);
            // SAFETY: buffer views point into the mapped arena of this frame.
            unsafe {
                list.RSSetScissorRects(&[RECT {
                    left,
                    top,
                    right,
                    bottom,
                }]);
                list.SetGraphicsRootDescriptorTable(1, table);
                list.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: vb.gpu_va,
                        SizeInBytes: (mesh.vertices.len() * size_of::<Vertex>()) as u32,
                        StrideInBytes: size_of::<Vertex>() as u32,
                    }]),
                );
                list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: ib.gpu_va,
                    SizeInBytes: (mesh.indices.len() * 4) as u32,
                    Format: DXGI_FORMAT_R32_UINT,
                }));
                list.DrawIndexedInstanced(mesh.indices.len() as u32, 1, 0, 0, 0);
            }
        }
        Ok(())
    }

    /// Drops textures egui no longer needs. Call only when the GPU no
    /// longer references them.
    pub fn free_textures(&mut self, ids: &[TextureId]) {
        for id in ids {
            self.textures.remove(id);
        }
    }
}
