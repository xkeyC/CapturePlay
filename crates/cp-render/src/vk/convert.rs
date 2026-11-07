//! Upload slot → plane textures → linear scRGB working frame.

use ash::vk;
use cp_core::color::{TransferFunction, yuv_to_rgb_matrix};
use cp_core::{ColorDesc, PixelFormat, Primaries, VideoMode};

use super::Result;
use super::arena::Arena;
use super::device::Gpu;
use super::pipeline::{self, Layouts, Pipeline};
use super::texture::{Buffer, Texture, barrier};
use crate::common::DecodeParams;
use crate::upload::SlotLayout;

/// Constants of `convert.hlsl` (D3D cbuffer layout, 24 dwords).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Constants {
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

struct PlaneTextures {
    mode: VideoMode,
    planes: Vec<Texture>,
}

pub struct ConvertPass {
    planar: Pipeline,
    yuyv: Pipeline,
    uyvy: Pipeline,
    bgra: Pipeline,
    inputs: Option<PlaneTextures>,
}

impl std::fmt::Debug for ConvertPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvertPass").finish_non_exhaustive()
    }
}

/// Texture formats/extents of each plane as stored in the upload slot, with
/// the bytes per texel (for the copy's row length).
fn plane_formats(format: PixelFormat, w: u32, h: u32) -> Vec<(vk::Format, u32, u32, u32)> {
    match format {
        PixelFormat::Nv12 => vec![
            (vk::Format::R8_UNORM, w, h, 1),
            (vk::Format::R8G8_UNORM, w / 2, h.div_ceil(2), 2),
        ],
        PixelFormat::P010 => vec![
            (vk::Format::R16_UNORM, w, h, 2),
            (vk::Format::R16G16_UNORM, w / 2, h.div_ceil(2), 4),
        ],
        PixelFormat::Yuy2 | PixelFormat::Uyvy => vec![(vk::Format::R8G8B8A8_UNORM, w / 2, h, 4)],
        PixelFormat::Bgra8 => vec![(vk::Format::B8G8R8A8_UNORM, w, h, 4)],
    }
}

impl ConvertPass {
    pub fn new(gpu: &Gpu, layouts: &Layouts) -> Result<Self> {
        Ok(Self {
            planar: Pipeline::compute(gpu, layouts, pipeline::CONVERT_PLANAR)?,
            yuyv: Pipeline::compute(gpu, layouts, pipeline::CONVERT_YUYV)?,
            uyvy: Pipeline::compute(gpu, layouts, pipeline::CONVERT_UYVY)?,
            bgra: Pipeline::compute(gpu, layouts, pipeline::CONVERT_BGRA)?,
            inputs: None,
        })
    }

    /// Plane textures for `mode`; the previous ones are returned for
    /// deferred destruction (the GPU may still read them).
    fn ensure_inputs(
        &mut self,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        mode: &VideoMode,
        retire: &mut Vec<Texture>,
    ) -> Result<()> {
        if self.inputs.as_ref().is_some_and(|i| i.mode == *mode) {
            return Ok(());
        }
        let planes = plane_formats(mode.format, mode.width, mode.height)
            .into_iter()
            .map(|(fmt, w, h, _)| {
                Texture::new(
                    gpu,
                    cmd,
                    w,
                    h,
                    fmt,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(old) = self.inputs.replace(PlaneTextures {
            mode: *mode,
            planes,
        }) {
            retire.extend(old.planes);
        }
        Ok(())
    }

    /// Records upload-slot copy + conversion into `out` (RGBA16F storage).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        gpu: &Gpu,
        layouts: &Layouts,
        arena: &mut Arena,
        cmd: vk::CommandBuffer,
        upload: &Buffer,
        layout: &SlotLayout,
        mode: &VideoMode,
        color: &ColorDesc,
        params: DecodeParams,
        out: &Texture,
        retire: &mut Vec<Texture>,
    ) -> Result<()> {
        self.ensure_inputs(gpu, cmd, mode, retire)?;
        let Some(inputs) = self.inputs.as_ref() else {
            return Ok(());
        };
        let d = &gpu.device;
        let formats = plane_formats(mode.format, mode.width, mode.height);
        barrier(d, cmd);
        for (i, (p, &(_, _, _, texel))) in inputs.planes.iter().zip(&formats).enumerate() {
            let region = vk::BufferImageCopy::default()
                .buffer_offset(layout.planes[i].offset as u64)
                .buffer_row_length((layout.planes[i].pitch / texel as usize) as u32)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: p.width,
                    height: p.height,
                    depth: 1,
                });
            // SAFETY: the region lies within the upload slot written by the
            // capture thread (same layout) and within the image.
            unsafe {
                d.cmd_copy_buffer_to_image(
                    cmd,
                    upload.buffer,
                    p.image,
                    vk::ImageLayout::GENERAL,
                    &[region],
                );
            }
        }
        barrier(d, cmd);

        let pipeline = match mode.format {
            PixelFormat::Nv12 | PixelFormat::P010 => &self.planar,
            PixelFormat::Yuy2 => &self.yuyv,
            PixelFormat::Uyvy => &self.uyvy,
            PixelFormat::Bgra8 => &self.bgra,
        };
        let chroma = inputs.planes.get(1).unwrap_or(&inputs.planes[0]);
        let c = Constants {
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
            chroma_texel: [1.0 / chroma.width as f32, 1.0 / chroma.height as f32],
            pad1: [0.0; 2],
        };
        let Some(cbuf) = arena.constants(&c, gpu.min_uniform_alignment) else {
            return Err(super::VkError::Other("frame arena exhausted".into()));
        };
        let luma = pipeline::sampled(inputs.planes[0].view);
        let chroma_info = pipeline::sampled(chroma.view);
        let storage = pipeline::sampled(out.view);
        let cbufs = [cbuf];
        let sampler = layouts.sampler_info();
        let writes = [
            pipeline::image_write(
                pipeline::BINDING_SRV,
                vk::DescriptorType::SAMPLED_IMAGE,
                &luma,
            ),
            pipeline::image_write(
                pipeline::BINDING_SRV + 1,
                vk::DescriptorType::SAMPLED_IMAGE,
                &chroma_info,
            ),
            pipeline::image_write(
                pipeline::BINDING_UAV,
                vk::DescriptorType::STORAGE_IMAGE,
                &storage,
            ),
            pipeline::sampler_write(&sampler),
            pipeline::constants_write(&cbufs),
        ];
        // SAFETY: recording into an open command buffer; descriptors
        // reference live views and the frame arena.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            gpu.push_fn.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layouts.compute,
                0,
                &writes,
            );
        }
        pipeline::dispatch_2d(d, cmd, mode.width, mode.height);
        Ok(())
    }
}
