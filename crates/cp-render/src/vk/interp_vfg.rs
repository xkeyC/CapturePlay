//! Frame interpolation with NVIDIA's AI Video Frame Generation (RTX 40/50)
//! on Vulkan.
//!
//! The Vulkan twin of the D3D12 `interp_vfg`. Per output frame: encode the
//! frames of the pair that are not cached yet into packed RGB10A2 buffers
//! exported to CUDA (opaque fds), submit with a signal of the exported
//! `producer` timeline semaphore, let VFG generate the frame at `t` on its
//! CUDA stream (which signals `completion`), and decode the result once the
//! rest of the frame has waited for `completion` — all GPU-side, no CPU
//! waits.

use std::os::fd::{FromRawFd, OwnedFd};
use std::path::PathBuf;

use ash::vk;
use cp_nvidia::vfg::{ExportedBuffer, VfgSession, VulkanShared};

use super::device::Gpu;
use super::pipeline::{self, ComputeLayout, Pipeline};
use super::renderer::{FrameInput, Interpolator, RecordCtx};
use super::texture::{Texture, barrier};
use super::{Result, VkError};
use crate::constants::PackConstants;

fn vfx_err(e: cp_nvidia::vfx::VfxError) -> VkError {
    VkError::Other(e.to_string())
}

/// A device-local storage buffer whose memory is exported to CUDA.
struct SharedBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // SAFETY: the owner waited for the GPU and the CUDA stream.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl SharedBuffer {
    /// Allocates `bytes` (dedicated, exportable) and exports an fd for CUDA.
    fn new(gpu: &Gpu, bytes: u64) -> Result<(Self, ExportedBuffer)> {
        let ext_fn = gpu
            .external_memory_fn
            .as_ref()
            .ok_or_else(|| VkError::Other("no external memory support".into()))?;
        let opaque = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(opaque);
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let d = &gpu.device;
        // SAFETY: valid create infos; cleanup on every error path (the
        // SharedBuffer drop covers everything after allocation).
        unsafe {
            let buffer = d.create_buffer(&info, None)?;
            let req = d.get_buffer_memory_requirements(buffer);
            let Some(index) = gpu.memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            ) else {
                d.destroy_buffer(buffer, None);
                return Err(VkError::NoMemoryType);
            };
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(opaque);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(index)
                .push_next(&mut export)
                .push_next(&mut dedicated);
            let memory = match d.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buffer, None);
                    return Err(e.into());
                }
            };
            let shared = Self {
                device: d.clone(),
                buffer,
                memory,
                size: bytes,
            };
            d.bind_buffer_memory(buffer, memory, 0)?;
            let fd = ext_fn.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(opaque),
            )?;
            let exported = ExportedBuffer {
                // SAFETY: a fresh fd exported for us.
                fd: OwnedFd::from_raw_fd(fd),
                size: req.size,
            };
            Ok((shared, exported))
        }
    }

    fn info(&self) -> [vk::DescriptorBufferInfo; 1] {
        [vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: 0,
            range: self.size,
        }]
    }
}

/// An exportable timeline semaphore.
struct SharedSemaphore {
    device: ash::Device,
    semaphore: vk::Semaphore,
}

impl Drop for SharedSemaphore {
    fn drop(&mut self) {
        // SAFETY: the owner waited for the GPU and the CUDA stream.
        unsafe { self.device.destroy_semaphore(self.semaphore, None) };
    }
}

impl SharedSemaphore {
    fn new(gpu: &Gpu) -> Result<(Self, OwnedFd)> {
        let ext_fn = gpu
            .external_semaphore_fn
            .as_ref()
            .ok_or_else(|| VkError::Other("no external semaphore support".into()))?;
        let opaque = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;
        let mut ty = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export = vk::ExportSemaphoreCreateInfo::default().handle_types(opaque);
        // SAFETY: valid create info; the semaphore is owned by `Self`.
        unsafe {
            let semaphore = gpu.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default()
                    .push_next(&mut ty)
                    .push_next(&mut export),
                None,
            )?;
            let s = Self {
                device: gpu.device.clone(),
                semaphore,
            };
            let fd = ext_fn.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(semaphore)
                    .handle_type(opaque),
            )?;
            // SAFETY: a fresh fd exported for us.
            Ok((s, OwnedFd::from_raw_fd(fd)))
        }
    }
}

struct Resources {
    // Drop order: the CUDA session (drains its stream) before the Vulkan
    // objects it imported.
    session: VfgSession,
    inputs: [SharedBuffer; 2],
    output: SharedBuffer,
    producer: SharedSemaphore,
    completion: SharedSemaphore,
    /// Frame id held by each input slot.
    slot_ids: [Option<u64>; 2],
    /// Last value of `producer` / `completion`.
    value: u64,
}

pub struct VfgInterpolator {
    encode: Pipeline,
    decode: Pipeline,
    encode_layout: ComputeLayout,
    decode_layout: ComputeLayout,
    runtime_dir: PathBuf,
    mode: u32,
    hdr: bool,
    sdr_white: f32,
    res: Option<Resources>,
}

impl std::fmt::Debug for VfgInterpolator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VfgInterpolator")
            .field("runtime_dir", &self.runtime_dir)
            .field("mode", &self.mode)
            .finish()
    }
}

impl VfgInterpolator {
    /// `mode`: 0 low, 1 medium, 2 high. Fails fast if the runtime or the
    /// interop extensions are absent.
    pub fn new(gpu: &Gpu, runtime_dir: PathBuf, mode: u32) -> Result<Self> {
        if gpu.vendor_id != super::device::VENDOR_NVIDIA || !gpu.features.external_fd {
            return Err(VkError::Other(
                "VFG needs an NVIDIA GPU with external memory/semaphore fds".into(),
            ));
        }
        if !cp_nvidia::vfg::runtime_present(&runtime_dir) {
            return Err(VkError::Other(format!(
                "VFG runtime not found in {}",
                runtime_dir.display()
            )));
        }
        use vk::DescriptorType as T;
        let encode_layout = ComputeLayout::new(
            gpu,
            &[
                (pipeline::BINDING_SRV, T::SAMPLED_IMAGE),
                (pipeline::BINDING_UAV, T::STORAGE_BUFFER),
            ],
        )?;
        let decode_layout = ComputeLayout::new(
            gpu,
            &[
                (pipeline::BINDING_SRV, T::STORAGE_BUFFER),
                (pipeline::BINDING_UAV, T::STORAGE_IMAGE),
            ],
        )?;
        Ok(Self {
            encode: Pipeline::compute_in(gpu, encode_layout.layout, pipeline::VFG_ENCODE)?,
            decode: Pipeline::compute_in(gpu, decode_layout.layout, pipeline::VFG_DECODE)?,
            encode_layout,
            decode_layout,
            runtime_dir,
            mode,
            hdr: false,
            sdr_white: 1.0,
            res: None,
        })
    }

    fn ensure(&mut self, gpu: &Gpu, w: u32, h: u32) -> Result<()> {
        if self
            .res
            .as_ref()
            .is_some_and(|r| r.session.width == w && r.session.height == h)
        {
            return Ok(());
        }
        if self.res.is_some() {
            // Source size changed: in-flight frames may still use the old
            // buffers (rare, so a full wait is fine).
            gpu.wait_idle()?;
            self.res = None;
        }
        let pitch = (w * 4).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(h);
        let (in0, e0) = SharedBuffer::new(gpu, bytes)?;
        let (in1, e1) = SharedBuffer::new(gpu, bytes)?;
        let (output, eo) = SharedBuffer::new(gpu, bytes)?;
        let (producer, pfd) = SharedSemaphore::new(gpu)?;
        let (completion, cfd) = SharedSemaphore::new(gpu)?;
        let session = VfgSession::new(
            VulkanShared {
                device_uuid: gpu.device_uuid,
                inputs: [e0, e1],
                output: eo,
                pitch,
                producer: pfd,
                completion: cfd,
            },
            &self.runtime_dir,
            w,
            h,
            self.mode,
        )
        .map_err(vfx_err)?;
        self.res = Some(Resources {
            session,
            inputs: [in0, in1],
            output,
            producer,
            completion,
            slot_ids: [None, None],
            value: 0,
        });
        Ok(())
    }

    fn constants(&self, res: &Resources) -> PackConstants {
        PackConstants {
            width: res.session.width,
            height: res.session.height,
            pitch: res.session.pitch,
            hdr: u32::from(self.hdr),
            sdr_white: self.sdr_white,
            pad: [0.0; 3],
        }
    }

    /// Records one compute pass of `p` with `writes` (constants added).
    fn dispatch(
        ctx: &mut RecordCtx<'_>,
        p: &Pipeline,
        layout: &ComputeLayout,
        c: &PackConstants,
        writes: &[vk::WriteDescriptorSet<'_>],
    ) -> Result<()> {
        let cbuf = ctx
            .rec
            .arena
            .constants(c, ctx.gpu.min_uniform_alignment)
            .ok_or_else(|| VkError::Other("frame arena exhausted".into()))?;
        let cbufs = [cbuf];
        let mut all = writes.to_vec();
        all.push(pipeline::constants_write(&cbufs));
        let cmd = ctx.cmd();
        barrier(&ctx.gpu.device, cmd);
        // SAFETY: recording into the frame's open command buffer;
        // descriptors reference live views/buffers and the frame arena.
        unsafe {
            ctx.gpu
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            ctx.gpu.push_fn.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout.layout,
                0,
                &all,
            );
        }
        pipeline::dispatch_2d(&ctx.gpu.device, cmd, c.width, c.height);
        barrier(&ctx.gpu.device, cmd);
        Ok(())
    }

    fn record_encode(
        &self,
        ctx: &mut RecordCtx<'_>,
        res: &Resources,
        frame: &Texture,
        slot: usize,
    ) -> Result<()> {
        let image = pipeline::sampled(frame.view);
        let buffer = res.inputs[slot].info();
        let writes = [
            pipeline::image_write(
                pipeline::BINDING_SRV,
                vk::DescriptorType::SAMPLED_IMAGE,
                &image,
            ),
            pipeline::buffer_write(pipeline::BINDING_UAV, &buffer),
        ];
        Self::dispatch(
            ctx,
            &self.encode,
            &self.encode_layout,
            &self.constants(res),
            &writes,
        )
    }
}

impl Interpolator for VfgInterpolator {
    fn name(&self) -> &'static str {
        "NVIDIA VFG (AI)"
    }

    fn set_content(&mut self, hdr: bool, sdr_white: f32) {
        if hdr != self.hdr || (sdr_white - self.sdr_white).abs() > 1e-3 {
            self.hdr = hdr;
            self.sdr_white = sdr_white;
            // Cached encodings used the old mapping.
            if let Some(r) = &mut self.res {
                r.slot_ids = [None, None];
                r.session.invalidate_inputs();
            }
        }
    }

    fn interpolate(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        prev: FrameInput<'_>,
        next: FrameInput<'_>,
        t: f32,
        out: &Texture,
    ) -> Result<()> {
        self.ensure(ctx.gpu, next.tex.width, next.tex.height)?;
        let mut res = self
            .res
            .take()
            .ok_or_else(|| VkError::Other("VFG not ready".into()))?;
        let result = (|| {
            let slot_of = |id: u64| res.slot_ids.iter().position(|s| *s == Some(id));
            let prev_slot =
                slot_of(prev.id).unwrap_or_else(|| if slot_of(next.id) == Some(0) { 1 } else { 0 });
            let next_slot = 1 - prev_slot;
            let mut encoded = false;
            if res.slot_ids[prev_slot] != Some(prev.id) {
                self.record_encode(ctx, &res, prev.tex, prev_slot)?;
                res.slot_ids[prev_slot] = Some(prev.id);
                encoded = true;
            }
            if res.slot_ids[next_slot] != Some(next.id) {
                self.record_encode(ctx, &res, next.tex, next_slot)?;
                res.slot_ids[next_slot] = Some(next.id);
                encoded = true;
            }
            if encoded {
                res.session.invalidate_inputs();
            }

            // Hand the encoded inputs to CUDA, and wait for its result.
            res.value += 1;
            let v = res.value;
            ctx.submit_and_continue(&[(res.producer.semaphore, v)])?;
            res.session
                .run(prev_slot, next_slot, t, false, v)
                .map_err(vfx_err)?;
            ctx.wait_next(res.completion.semaphore, v);

            // Decode into the working-format output.
            let buffer = res.output.info();
            let storage = pipeline::sampled(out.view);
            let writes = [
                pipeline::buffer_write(pipeline::BINDING_SRV, &buffer),
                pipeline::image_write(
                    pipeline::BINDING_UAV,
                    vk::DescriptorType::STORAGE_IMAGE,
                    &storage,
                ),
            ];
            Self::dispatch(
                ctx,
                &self.decode,
                &self.decode_layout,
                &self.constants(&res),
                &writes,
            )
        })();
        self.res = Some(res);
        result
    }
}
