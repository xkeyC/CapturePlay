//! SPIR-V shaders (see `build.rs`), pipeline layouts and pipelines.
//!
//! One push-descriptor set layout per pipeline kind mirrors the binding
//! model the shaders are compiled with: sampled images at 0.., storage images
//! at 8.., the linear clamp sampler at 16 and the constant buffer at 24.
//!
//! The sampler is pushed with every draw/dispatch instead of being an
//! immutable sampler of the layout: with immutable samplers in a
//! push-descriptor layout the NVIDIA 615 driver filtered as NEAREST
//! (caught by `vk::tests`).

use std::ffi::CStr;

use ash::vk;

use super::Result;
use super::device::Gpu;

macro_rules! shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
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
pub const NGX_ENCODE: &[u8] = shader!("ngx_encode");
pub const VFG_ENCODE: &[u8] = shader!("vfg_encode");
pub const VFG_DECODE: &[u8] = shader!("vfg_decode");
pub const NGX_DECODE: &[u8] = shader!("ngx_decode");

pub const BINDING_SRV: u32 = 0;
pub const BINDING_UAV: u32 = 8;
pub const BINDING_SAMPLER: u32 = 16;
pub const BINDING_CONSTANTS: u32 = 24;

const ENTRY_MAIN: &CStr = c"main";
const ENTRY_VS: &CStr = c"vs_main";
const ENTRY_PS: &CStr = c"ps_main";

/// Shared layouts and the linear clamp sampler.
pub struct Layouts {
    device: ash::Device,
    pub sampler: vk::Sampler,
    /// Compute: 2 sampled images, 2 storage images, sampler, constants.
    pub compute_set: vk::DescriptorSetLayout,
    pub compute: vk::PipelineLayout,
    /// Graphics: 1 sampled image, sampler, constants.
    pub graphics_set: vk::DescriptorSetLayout,
    pub graphics: vk::PipelineLayout,
}

impl std::fmt::Debug for Layouts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Layouts")
    }
}

impl Layouts {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let d = &gpu.device;
        // SAFETY: plain object creation.
        unsafe {
            let sampler = d.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?;
            let binding = |b: u32, ty: vk::DescriptorType, stages: vk::ShaderStageFlags| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(stages)
            };
            let cs = vk::ShaderStageFlags::COMPUTE;
            let compute_bindings = [
                binding(BINDING_SRV, vk::DescriptorType::SAMPLED_IMAGE, cs),
                binding(BINDING_SRV + 1, vk::DescriptorType::SAMPLED_IMAGE, cs),
                binding(BINDING_UAV, vk::DescriptorType::STORAGE_IMAGE, cs),
                binding(BINDING_UAV + 1, vk::DescriptorType::STORAGE_IMAGE, cs),
                binding(BINDING_SAMPLER, vk::DescriptorType::SAMPLER, cs),
                binding(BINDING_CONSTANTS, vk::DescriptorType::UNIFORM_BUFFER, cs),
            ];
            let fs = vk::ShaderStageFlags::FRAGMENT;
            let graphics_bindings = [
                binding(BINDING_SRV, vk::DescriptorType::SAMPLED_IMAGE, fs),
                binding(BINDING_SAMPLER, vk::DescriptorType::SAMPLER, fs),
                binding(
                    BINDING_CONSTANTS,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::ShaderStageFlags::VERTEX | fs,
                ),
            ];
            let set_layout = |bindings: &[vk::DescriptorSetLayoutBinding<'_>]| {
                d.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(bindings),
                    None,
                )
            };
            let compute_set = set_layout(&compute_bindings)?;
            let graphics_set = set_layout(&graphics_bindings)?;
            let compute = d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[compute_set]),
                None,
            )?;
            let graphics = d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[graphics_set]),
                None,
            )?;
            Ok(Self {
                device: d.clone(),
                sampler,
                compute_set,
                compute,
                graphics_set,
                graphics,
            })
        }
    }
}

impl Drop for Layouts {
    fn drop(&mut self) {
        // SAFETY: pipelines using these layouts are destroyed first; the GPU
        // is idle, so the sampler is no longer referenced.
        unsafe {
            self.device.destroy_pipeline_layout(self.compute, None);
            self.device.destroy_pipeline_layout(self.graphics, None);
            self.device
                .destroy_descriptor_set_layout(self.compute_set, None);
            self.device
                .destroy_descriptor_set_layout(self.graphics_set, None);
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

fn module(device: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(spv))
        .map_err(|e| super::VkError::Other(format!("invalid SPIR-V: {e}")))?;
    // SAFETY: `words` is valid SPIR-V produced by the build script.
    Ok(unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    })
}

/// A pipeline owning its handle (layouts are shared).
pub struct Pipeline {
    device: ash::Device,
    pub pipeline: vk::Pipeline,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Pipeline")
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // SAFETY: owners drop pipelines only after the GPU is idle.
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
    }
}

impl Pipeline {
    pub fn compute(gpu: &Gpu, layouts: &Layouts, spv: &[u8]) -> Result<Self> {
        Self::compute_in(gpu, layouts.compute, spv)
    }

    /// A compute pipeline with its own pipeline `layout`.
    pub fn compute_in(gpu: &Gpu, layout: vk::PipelineLayout, spv: &[u8]) -> Result<Self> {
        let d = &gpu.device;
        let m = module(d, spv)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(m)
            .name(ENTRY_MAIN);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: valid module and layout; the module is destroyed after
        // pipeline creation (allowed).
        let r = unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: no longer needed once the pipeline exists (or failed).
        unsafe { d.destroy_shader_module(m, None) };
        let pipeline = r.map_err(|(_, e)| e)?[0];
        Ok(Self {
            device: d.clone(),
            pipeline,
        })
    }

    /// Fullscreen-triangle or indexed-mesh pipeline drawing into one color
    /// attachment of `format` (dynamic rendering, dynamic viewport/scissor).
    pub fn graphics(
        gpu: &Gpu,
        layouts: &Layouts,
        vs: &[u8],
        ps: &[u8],
        format: vk::Format,
        vertex: Option<(
            &[vk::VertexInputBindingDescription],
            &[vk::VertexInputAttributeDescription],
        )>,
        blend: vk::PipelineColorBlendAttachmentState,
    ) -> Result<Self> {
        let d = &gpu.device;
        let vsm = module(d, vs)?;
        let psm = match module(d, ps) {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: created above, unused.
                unsafe { d.destroy_shader_module(vsm, None) };
                return Err(e);
            }
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vsm)
                .name(ENTRY_VS),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(psm)
                .name(ENTRY_PS),
        ];
        let (bindings, attributes) = vertex.unwrap_or((&[], &[]));
        let vi = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(bindings)
            .vertex_attribute_descriptions(attributes);
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vp = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [blend];
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let ds = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let formats = [format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .viewport_state(&vp)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .color_blend_state(&cb)
            .dynamic_state(&ds)
            .layout(layouts.graphics)
            .push_next(&mut rendering);
        // SAFETY: all referenced state lives until the call returns.
        let r = unsafe { d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: modules are no longer needed.
        unsafe {
            d.destroy_shader_module(vsm, None);
            d.destroy_shader_module(psm, None);
        }
        let pipeline = r.map_err(|(_, e)| e)?[0];
        Ok(Self {
            device: d.clone(),
            pipeline,
        })
    }
}

/// A push-descriptor compute layout with custom binding types (shaders that
/// read or write raw buffers).
pub struct ComputeLayout {
    device: ash::Device,
    pub set: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
}

impl std::fmt::Debug for ComputeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ComputeLayout")
    }
}

impl ComputeLayout {
    /// `bindings`: (binding, type) pairs; the constant buffer at
    /// [`BINDING_CONSTANTS`] is added.
    pub fn new(gpu: &Gpu, bindings: &[(u32, vk::DescriptorType)]) -> Result<Self> {
        let d = &gpu.device;
        let all: Vec<_> = bindings
            .iter()
            .copied()
            .chain(std::iter::once((
                BINDING_CONSTANTS,
                vk::DescriptorType::UNIFORM_BUFFER,
            )))
            .map(|(b, ty)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        // SAFETY: plain object creation; the set layout is destroyed if the
        // pipeline layout fails.
        unsafe {
            let set = d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&all),
                None,
            )?;
            match d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set]),
                None,
            ) {
                Ok(layout) => Ok(Self {
                    device: d.clone(),
                    set,
                    layout,
                }),
                Err(e) => {
                    d.destroy_descriptor_set_layout(set, None);
                    Err(e.into())
                }
            }
        }
    }
}

impl Drop for ComputeLayout {
    fn drop(&mut self) {
        // SAFETY: pipelines using the layout are destroyed first.
        unsafe {
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set, None);
        }
    }
}

/// Write of one storage buffer at `binding`.
pub fn buffer_write(binding: u32, info: &[vk::DescriptorBufferInfo]) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(info)
}

/// Opaque color writes (no blending).
pub fn opaque() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)
}

/// Sampled-image descriptor in `GENERAL` layout.
pub fn sampled(view: vk::ImageView) -> [vk::DescriptorImageInfo; 1] {
    [vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL)]
}

/// Write of one image descriptor at `binding`.
pub fn image_write<'a>(
    binding: u32,
    ty: vk::DescriptorType,
    info: &'a [vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_binding(binding)
        .descriptor_type(ty)
        .image_info(info)
}

impl Layouts {
    /// Descriptor info of the linear clamp sampler (see the module docs).
    pub fn sampler_info(&self) -> [vk::DescriptorImageInfo; 1] {
        [vk::DescriptorImageInfo::default().sampler(self.sampler)]
    }
}

/// Write of the sampler.
pub fn sampler_write(info: &[vk::DescriptorImageInfo]) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_binding(BINDING_SAMPLER)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .image_info(info)
}

/// Write of the constant buffer.
pub fn constants_write(info: &[vk::DescriptorBufferInfo]) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_binding(BINDING_CONSTANTS)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(info)
}

/// 8×8 groups covering `w × h` (every compute shader uses 8×8).
pub fn dispatch_2d(device: &ash::Device, cmd: vk::CommandBuffer, w: u32, h: u32) {
    // SAFETY: recording into an open command buffer with a bound pipeline.
    unsafe { device.cmd_dispatch(cmd, w.div_ceil(8), h.div_ceil(8), 1) };
}
