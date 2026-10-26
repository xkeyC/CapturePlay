//! Final scale + display mapping into the swapchain image.

use ash::vk;

use super::Result;
use super::arena::Arena;
use super::device::Gpu;
use super::pipeline::{self, Layouts, Pipeline};
use super::texture::Texture;
use crate::common::{OutputParams, letterbox};

/// Constants of `output.hlsl` (D3D cbuffer layout, 20 dwords).
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

pub struct OutputPass {
    pipeline: Pipeline,
    pub format: vk::Format,
}

impl std::fmt::Debug for OutputPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OutputPass")
    }
}

/// Viewport + scissor covering `size`.
pub fn set_viewport(device: &ash::Device, cmd: vk::CommandBuffer, size: (u32, u32)) {
    // SAFETY: recording into an open command buffer.
    unsafe {
        device.cmd_set_viewport(
            cmd,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: size.0 as f32,
                height: size.1 as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        device.cmd_set_scissor(
            cmd,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size.0,
                    height: size.1,
                },
            }],
        );
    }
}

impl OutputPass {
    /// Pipeline for render targets of `format` (the swapchain format).
    pub fn new(gpu: &Gpu, layouts: &Layouts, format: vk::Format) -> Result<Self> {
        Ok(Self {
            pipeline: Pipeline::graphics(
                gpu,
                layouts,
                pipeline::OUTPUT_VS,
                pipeline::OUTPUT_PS,
                format,
                None,
                pipeline::opaque(),
            )?,
            format,
        })
    }

    /// Draws `src` letterboxed into the current render target of `size`
    /// (inside dynamic rendering that cleared it). `None` leaves the clear.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        gpu: &Gpu,
        layouts: &Layouts,
        arena: &mut Arena,
        cmd: vk::CommandBuffer,
        src: Option<&Texture>,
        size: (u32, u32),
        p: &OutputParams,
    ) -> Result<()> {
        let Some(src) = src else {
            return Ok(());
        };
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
        let Some(cbuf) = arena.constants(&c, gpu.min_uniform_alignment) else {
            return Err(super::VkError::Other("frame arena exhausted".into()));
        };
        let d = &gpu.device;
        let image = pipeline::sampled(src.view);
        let cbufs = [cbuf];
        let sampler = layouts.sampler_info();
        let writes = [
            pipeline::image_write(
                pipeline::BINDING_SRV,
                vk::DescriptorType::SAMPLED_IMAGE,
                &image,
            ),
            pipeline::sampler_write(&sampler),
            pipeline::constants_write(&cbufs),
        ];
        set_viewport(d, cmd, size);
        // SAFETY: recording inside dynamic rendering; descriptors reference
        // live views and the frame arena.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            gpu.push_fn.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layouts.graphics,
                0,
                &writes,
            );
            d.cmd_draw(cmd, 3, 1, 0, 0);
        }
        Ok(())
    }
}
