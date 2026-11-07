//! Minimal egui renderer for Vulkan, drawing into the swapchain image.
//!
//! Textures are `R8G8B8A8_SRGB`, so sampling linearizes them like the
//! D3D12 backend's `*_UNORM_SRGB` views.

use std::collections::HashMap;

use ash::vk;
use egui::epaint::{ImageDelta, Primitive, Vertex};
use egui::{ImageData, TextureId};

use super::Result;
use super::arena::Arena;
use super::device::Gpu;
use super::pipeline::{self, Layouts, Pipeline};
use super::texture::{Texture, barrier};
use crate::common::EguiFrame;
use crate::constants::EguiConstants;

pub struct EguiPass {
    pipeline: Pipeline,
    textures: HashMap<TextureId, Texture>,
}

impl std::fmt::Debug for EguiPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EguiPass")
            .field("textures", &self.textures.len())
            .finish()
    }
}

impl EguiPass {
    pub fn new(gpu: &Gpu, layouts: &Layouts, format: vk::Format) -> Result<Self> {
        let bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ];
        // Premultiplied alpha, same factors as the D3D12 pass.
        let blend = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        Ok(Self {
            pipeline: Pipeline::graphics(
                gpu,
                layouts,
                pipeline::EGUI_VS,
                pipeline::EGUI_PS,
                format,
                Some((&bindings, &attributes)),
                blend,
            )?,
            textures: HashMap::new(),
        })
    }

    fn upload_image(
        &mut self,
        gpu: &Gpu,
        arena: &mut Arena,
        cmd: vk::CommandBuffer,
        id: TextureId,
        delta: &ImageDelta,
        retire: &mut Vec<Texture>,
    ) -> Result<()> {
        let ImageData::Color(img) = &delta.image;
        let (w, h) = (img.size[0] as u32, img.size[1] as u32);
        if w == 0 || h == 0 {
            return Ok(());
        }
        if delta.pos.is_none() {
            let tex = Texture::new(
                gpu,
                cmd,
                w,
                h,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            retire.extend(self.textures.insert(id, tex));
        }
        let Some(tex) = self.textures.get(&id) else {
            return Ok(());
        };
        let Some(a) = arena.push(&img.pixels, 16) else {
            tracing::warn!("egui texture upload does not fit the frame arena");
            return Ok(());
        };
        let [x, y] = delta.pos.map_or([0, 0], |p| [p[0] as i32, p[1] as i32]);
        let region = vk::BufferImageCopy::default()
            .buffer_offset(a.offset)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x, y, z: 0 })
            .image_extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            });
        barrier(&gpu.device, cmd);
        // SAFETY: tightly packed pixels in the arena; the region is within
        // the texture (egui guarantees it for partial updates).
        unsafe {
            gpu.device.cmd_copy_buffer_to_image(
                cmd,
                arena.buffer.buffer,
                tex.image,
                vk::ImageLayout::GENERAL,
                &[region],
            );
        }
        Ok(())
    }

    /// Applies texture updates. Must be recorded outside dynamic rendering;
    /// replaced textures go to `retire` for deferred destruction.
    pub fn upload(
        &mut self,
        gpu: &Gpu,
        arena: &mut Arena,
        cmd: vk::CommandBuffer,
        frame: &EguiFrame<'_>,
        retire: &mut Vec<Texture>,
    ) -> Result<()> {
        for (id, deltas) in &frame.textures_delta.set {
            for delta in deltas {
                self.upload_image(gpu, arena, cmd, *id, delta, retire)?;
            }
        }
        barrier(&gpu.device, cmd);
        Ok(())
    }

    /// Draws the primitives inside dynamic rendering on a target of
    /// `size_px`.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        layouts: &Layouts,
        arena: &mut Arena,
        cmd: vk::CommandBuffer,
        size_px: (u32, u32),
        frame: &EguiFrame<'_>,
    ) -> Result<()> {
        if frame.primitives.is_empty() {
            return Ok(());
        }
        let d = &gpu.device;
        let ppp = frame.pixels_per_point;
        let c = EguiConstants::new(size_px, ppp, frame.white_scale);
        let Some(cbuf) = arena.constants(&c, gpu.min_uniform_alignment) else {
            return Err(super::VkError::Other("frame arena exhausted".into()));
        };
        let cbufs = [cbuf];
        let sampler = layouts.sampler_info();
        super::output::set_viewport(d, cmd, size_px);
        // SAFETY: recording inside dynamic rendering.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline)
        };

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
                arena.push(&mesh.vertices, 16),
                arena.push(&mesh.indices, 16),
            ) else {
                tracing::warn!("egui geometry does not fit the frame arena");
                break;
            };
            let image = pipeline::sampled(tex.view);
            let writes = [
                pipeline::image_write(
                    pipeline::BINDING_SRV,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &image,
                ),
                pipeline::sampler_write(&sampler),
                pipeline::constants_write(&cbufs),
            ];
            // SAFETY: buffers point into this frame's mapped arena; the
            // descriptors reference live views.
            unsafe {
                d.cmd_set_scissor(
                    cmd,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: left, y: top },
                        extent: vk::Extent2D {
                            width: (right - left) as u32,
                            height: (bottom - top) as u32,
                        },
                    }],
                );
                gpu.push_fn.cmd_push_descriptor_set(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layouts.graphics,
                    0,
                    &writes,
                );
                d.cmd_bind_vertex_buffers(cmd, 0, &[arena.buffer.buffer], &[vb.offset]);
                d.cmd_bind_index_buffer(cmd, arena.buffer.buffer, ib.offset, vk::IndexType::UINT32);
                d.cmd_draw_indexed(cmd, mesh.indices.len() as u32, 1, 0, 0, 0);
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
