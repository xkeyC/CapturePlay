//! Upscaling with NVIDIA RTX Video Super Resolution through NGX on Vulkan.
//!
//! The Vulkan twin of the D3D12 `upscale_ngx`: encode the working frame as
//! PQ/BT.2020 into an `A2B10G10R10_UNORM` image, evaluate VSR into a second
//! one, decode into the working frame of the target size — all recorded on
//! the frame's command buffer.
//!
//! The feature (model) is created on a loader thread. Vulkan queues need
//! external synchronization, so the loader only records the creation work
//! into its own command buffer; the render thread submits it.

use std::path::PathBuf;

use ash::vk;
use cp_nvidia::ngx_vk::{self, VkContext, VkImage, VsrFeature, VsrQuality};

use super::device::Gpu;
use super::pipeline::{self, Pipeline};
use super::renderer::{RecordCtx, Upscaler};
use super::texture::{Texture, barrier};
use super::{Result, VkError};
use crate::constants::PackConstants;

type Size = (u32, u32);

const SURFACE_FORMAT: vk::Format = vk::Format::A2B10G10R10_UNORM_PACK32;

/// The created feature and the recorded creation work, handed from the
/// loader thread.
struct Loaded {
    feature: VsrFeature,
    device: ash::Device,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}
// SAFETY: the NGX handle is only used under the NGX lock (see
// `cp_nvidia::ngx_vk`); the command pool is used by one thread at a time
// (the loader, then the render thread).
unsafe impl Send for Loaded {}

type Loader = std::thread::JoinHandle<std::result::Result<Loaded, String>>;

/// RGB10A2 surfaces NGX reads and writes, for one `(src, dst)` pair.
struct Surfaces {
    src: Size,
    dst: Size,
    input: Texture,
    output: Texture,
}

pub struct NgxVsrUpscaler {
    encode: Pipeline,
    decode: Pipeline,
    quality: VsrQuality,
    feature_dirs: Vec<PathBuf>,
    data_dir: PathBuf,
    feature: Option<VsrFeature>,
    loading: Option<Loader>,
    failed: bool,
    surfaces: Option<Surfaces>,
}

impl std::fmt::Debug for NgxVsrUpscaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgxVsrUpscaler")
            .field("quality", &self.quality)
            .finish_non_exhaustive()
    }
}

impl NgxVsrUpscaler {
    /// Fails fast when NGX was not built in, the device lacks what NGX
    /// needs or no VSR snippet is found in `feature_dirs`.
    pub fn new(
        gpu: &Gpu,
        layouts: &super::pipeline::Layouts,
        feature_dirs: Vec<PathBuf>,
        data_dir: PathBuf,
        quality: VsrQuality,
    ) -> Result<Self> {
        if !ngx_vk::built() {
            return Err(VkError::Other(ngx_vk::NgxError::NotBuilt.to_string()));
        }
        if !gpu.features.ngx || gpu.vendor_id != super::device::VENDOR_NVIDIA {
            return Err(VkError::Other(
                "the Vulkan device lacks the extensions NGX needs".into(),
            ));
        }
        if !feature_dirs.iter().any(|d| ngx_vk::runtime_present(d)) {
            return Err(VkError::Other(
                "libnvidia-ngx-vsr.so not found (NVIDIA Video Effects runtime, see runtime/README.md)"
                    .into(),
            ));
        }
        Ok(Self {
            encode: Pipeline::compute(gpu, layouts, pipeline::NGX_ENCODE)?,
            decode: Pipeline::compute(gpu, layouts, pipeline::NGX_DECODE)?,
            quality,
            feature_dirs,
            data_dir,
            feature: None,
            loading: None,
            failed: false,
            surfaces: None,
        })
    }

    pub fn set_quality(&mut self, quality: VsrQuality) {
        self.quality = quality;
    }

    fn spawn_loader(&mut self, gpu: &Gpu) {
        let ctx = gpu.ngx_context();
        let device = gpu.device.clone();
        let family = gpu.queue_family;
        let dirs = self.feature_dirs.clone();
        let data = self.data_dir.clone();
        let handle = std::thread::Builder::new()
            .name("ngx-load".into())
            .spawn(move || load(ctx, device, family, &dirs, &data));
        match handle {
            Ok(h) => self.loading = Some(h),
            Err(e) => {
                tracing::error!("cannot start NGX loader: {e}");
                self.failed = true;
            }
        }
    }

    /// Submits the loader's creation work and waits for it.
    fn finish_load(gpu: &Gpu, l: Loaded) -> Result<VsrFeature> {
        let r = gpu.submit(l.cmd, &[], &[]).and_then(|v| gpu.wait(v));
        // SAFETY: the creation work completed (or failed to submit); the
        // pool is no longer used.
        unsafe { l.device.destroy_command_pool(l.pool, None) };
        r.map(|()| l.feature)
    }
}

/// Initializes NGX and records the feature creation on a private command
/// buffer (submitted by the render thread).
fn load(
    ctx: VkContext,
    device: ash::Device,
    family: u32,
    dirs: &[PathBuf],
    data_dir: &std::path::Path,
) -> std::result::Result<Loaded, String> {
    let _ = std::fs::create_dir_all(data_dir);
    let t0 = std::time::Instant::now();
    // SAFETY: plain object creation; the pool and buffer belong to this
    // thread until handed over.
    unsafe {
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
            .map_err(|e| e.to_string())?;
        let cmd = match device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        ) {
            Ok(c) => c[0],
            Err(e) => {
                device.destroy_command_pool(pool, None);
                return Err(e.to_string());
            }
        };
        let recorded = device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| e.to_string())
            .and_then(|()| VsrFeature::create(&ctx, cmd, dirs, data_dir).map_err(|e| e.to_string()))
            .and_then(|f| {
                device
                    .end_command_buffer(cmd)
                    .map(|()| f)
                    .map_err(|e| e.to_string())
            });
        match recorded {
            Ok(feature) => {
                tracing::info!(
                    ms = t0.elapsed().as_millis() as u64,
                    "NVIDIA RTX VSR (NGX Vulkan) created"
                );
                Ok(Loaded {
                    feature,
                    device,
                    pool,
                    cmd,
                })
            }
            Err(e) => {
                device.destroy_command_pool(pool, None);
                Err(e)
            }
        }
    }
}

fn image(t: &Texture) -> VkImage {
    VkImage {
        image: t.image,
        view: t.view,
        format: t.format,
        width: t.width,
        height: t.height,
    }
}

impl Upscaler for NgxVsrUpscaler {
    fn name(&self) -> &'static str {
        "NVIDIA RTX VSR"
    }

    fn factors(&self) -> &'static [f32] {
        &cp_core::scale::VSR_FACTORS
    }

    fn prepare(&mut self, gpu: &Gpu, _src: Size, _dst: Size) -> bool {
        if self.failed {
            return false;
        }
        if self.feature.is_some() {
            return true;
        }
        match self.loading.take() {
            None => {
                self.spawn_loader(gpu);
                false
            }
            Some(h) if !h.is_finished() => {
                self.loading = Some(h);
                false
            }
            Some(h) => {
                let loaded = match h.join() {
                    Ok(Ok(l)) => Self::finish_load(gpu, l).map_err(|e| e.to_string()),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err("loader panicked".to_owned()),
                };
                match loaded {
                    Ok(f) => {
                        tracing::info!("NVIDIA RTX VSR (NGX Vulkan) ready");
                        self.feature = Some(f);
                        true
                    }
                    Err(e) => {
                        tracing::error!("RTX VSR unavailable: {e}");
                        self.failed = true;
                        false
                    }
                }
            }
        }
    }

    fn upscale(&mut self, ctx: &mut RecordCtx<'_>, src: &Texture, out: &Texture) -> Result<()> {
        let Some(feature) = self.feature.as_ref() else {
            return Err(VkError::Other("RTX VSR not prepared".into()));
        };
        let (src_size, dst_size) = ((src.width, src.height), (out.width, out.height));
        if self
            .surfaces
            .as_ref()
            .is_none_or(|s| (s.src, s.dst) != (src_size, dst_size))
        {
            let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
            let surface =
                |(w, h): Size| Texture::new(ctx.gpu, ctx.cmd(), w, h, SURFACE_FORMAT, usage);
            let new = Surfaces {
                src: src_size,
                dst: dst_size,
                input: surface(src_size)?,
                output: surface(dst_size)?,
            };
            if let Some(old) = self.surfaces.replace(new) {
                ctx.rec.retire.extend([old.input, old.output]);
            }
        }
        let Some(s) = self.surfaces.as_ref() else {
            return Err(VkError::Other("RTX VSR surfaces".into()));
        };
        let d = &ctx.gpu.device;
        let constants = |w: u32, h: u32| PackConstants {
            width: w,
            height: h,
            hdr: 1,
            ..Default::default()
        };
        let pass = |ctx: &mut RecordCtx<'_>, p: &Pipeline, from: &Texture, to: &Texture| {
            let c = constants(to.width, to.height);
            let cbuf = ctx
                .rec
                .arena
                .constants(&c, ctx.gpu.min_uniform_alignment)
                .ok_or_else(|| VkError::Other("frame arena exhausted".into()))?;
            let cbufs = [cbuf];
            let sampled = pipeline::sampled(from.view);
            let storage = pipeline::sampled(to.view);
            let sampler = ctx.layouts.sampler_info();
            let writes = [
                pipeline::image_write(
                    pipeline::BINDING_SRV,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    &sampled,
                ),
                pipeline::image_write(
                    pipeline::BINDING_UAV,
                    vk::DescriptorType::STORAGE_IMAGE,
                    &storage,
                ),
                pipeline::sampler_write(&sampler),
                pipeline::constants_write(&cbufs),
            ];
            barrier(&ctx.gpu.device, ctx.cmd());
            // SAFETY: recording into the frame's open command buffer;
            // descriptors reference live views and the frame arena.
            unsafe {
                ctx.gpu.device.cmd_bind_pipeline(
                    ctx.cmd(),
                    vk::PipelineBindPoint::COMPUTE,
                    p.pipeline,
                );
                ctx.gpu.push_fn.cmd_push_descriptor_set(
                    ctx.cmd(),
                    vk::PipelineBindPoint::COMPUTE,
                    ctx.layouts.compute,
                    0,
                    &writes,
                );
            }
            pipeline::dispatch_2d(&ctx.gpu.device, ctx.cmd(), to.width, to.height);
            Ok::<_, VkError>(())
        };

        // 1. Working frame → PQ RGB10A2 (the shader iterates the source size).
        pass(ctx, &self.encode, src, &s.input)?;
        // 2. VSR.
        barrier(d, ctx.cmd());
        feature
            .evaluate(
                ctx.cmd(),
                &image(&s.input),
                s.src,
                &image(&s.output),
                s.dst,
                self.quality,
            )
            .map_err(|e| VkError::Other(e.to_string()))?;
        // 3. PQ RGB10A2 → working frame of the target size. NGX left its own
        // pipeline bound; the pass binds ours again.
        pass(ctx, &self.decode, &s.output, out)?;
        barrier(d, ctx.cmd());
        Ok(())
    }
}
