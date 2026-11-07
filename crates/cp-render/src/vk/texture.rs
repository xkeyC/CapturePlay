//! 2D images and buffers with their memory.
//!
//! Every image except the swapchain's lives in `GENERAL` layout: the passes
//! use images as storage, sampled and copy targets in turn, and on the GPUs
//! this targets `GENERAL` costs nothing. Hazards between passes are covered
//! by [`barrier`].

use ash::vk;

use super::Result;
use super::device::Gpu;

/// A 2D image + view in `GENERAL` layout.
pub struct Texture {
    device: ash::Device,
    pub image: vk::Image,
    pub view: vk::ImageView,
    memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Texture")
            .field("format", &self.format)
            .field("size", &(self.width, self.height))
            .finish()
    }
}

/// Usage of the working frames (RGBA16F): written by compute, sampled by
/// the output pass, copied by interop encoders.
pub const WORKING_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::STORAGE.as_raw()
        | vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

impl Texture {
    /// Creates a device-local image and transitions it to `GENERAL` on
    /// `cmd` (which must execute before the first use).
    pub fn new(
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: width.max(1),
                height: height.max(1),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let device = gpu.device.clone();
        // SAFETY: valid create infos; on error everything created so far is
        // destroyed before returning.
        unsafe {
            let image = device.create_image(&info, None)?;
            let req = device.get_image_memory_requirements(image);
            let memory = match gpu.allocate(
                req,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            ) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_image(image, None);
                    return Err(e);
                }
            };
            if let Err(e) = device.bind_image_memory(image, memory, 0) {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
                return Err(e.into());
            }
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(COLOR_RANGE);
            let view = match device.create_image_view(&view_info, None) {
                Ok(v) => v,
                Err(e) => {
                    device.destroy_image(image, None);
                    device.free_memory(memory, None);
                    return Err(e.into());
                }
            };
            layout_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            Ok(Self {
                device,
                image,
                view,
                memory,
                format,
                width: width.max(1),
                height: height.max(1),
            })
        }
    }
}

impl Texture {
    /// An image on memory another process exported as an opaque fd
    /// (`allocation_size` bytes, dedicated to an identical image), moved to
    /// `GENERAL` on `cmd`. Takes ownership of `fd` (closed on failure).
    #[allow(clippy::too_many_arguments)]
    pub fn import_opaque_fd(
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        fd: std::os::fd::OwnedFd,
        allocation_size: u64,
    ) -> Result<Self> {
        use std::os::fd::IntoRawFd as _;
        let opaque = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(opaque);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external);
        let device = gpu.device.clone();
        // SAFETY: valid create infos; on error everything created so far is
        // destroyed. A successful import owns the fd, a failed one leaves it
        // to us (dropped by `fd` until handed over).
        unsafe {
            let image = device.create_image(&info, None)?;
            let req = device.get_image_memory_requirements(image);
            let Some(index) = gpu.memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            ) else {
                device.destroy_image(image, None);
                return Err(super::VkError::NoMemoryType);
            };
            let raw = fd.into_raw_fd();
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(opaque)
                .fd(raw);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size)
                .memory_type_index(index)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            let memory = match device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    libc::close(raw);
                    device.destroy_image(image, None);
                    return Err(e.into());
                }
            };
            if let Err(e) = device.bind_image_memory(image, memory, 0) {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
                return Err(e.into());
            }
            let view = match device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(COLOR_RANGE),
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    device.destroy_image(image, None);
                    device.free_memory(memory, None);
                    return Err(e.into());
                }
            };
            layout_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            Ok(Self {
                device,
                image,
                view,
                memory,
                format,
                width,
                height,
            })
        }
    }
}

/// Ownership hand-over of `image` (kept in `GENERAL`) with another process
/// on the same device: `acquire` takes it from the external owner,
/// otherwise it is released to it.
pub fn external_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    family: u32,
    image: vk::Image,
    acquire: bool,
) {
    let (src, dst) = if acquire {
        (vk::QUEUE_FAMILY_EXTERNAL, family)
    } else {
        (family, vk::QUEUE_FAMILY_EXTERNAL)
    };
    let ib = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(src)
        .dst_queue_family_index(dst)
        .image(image)
        .subresource_range(COLOR_RANGE)];
    // SAFETY: recording into an open command buffer.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&ib),
        );
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        // SAFETY: owners keep textures alive until the GPU is done with them.
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub const COLOR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// Full execution + memory dependency between everything recorded before
/// and after on the queue (between passes and across submissions).
pub fn barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    // SAFETY: recording into an open command buffer.
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&mb));
    }
}

/// Layout transition with a full dependency.
pub fn layout_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let ib = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(COLOR_RANGE)];
    // SAFETY: recording into an open command buffer.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&ib),
        );
    }
}

/// A buffer with its own memory, optionally persistently mapped.
pub struct Buffer {
    device: ash::Device,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
    /// Host pointer for host-visible buffers.
    pub ptr: *mut u8,
}

// SAFETY: the handles are plain Vulkan objects; `ptr` is a persistent
// mapping whose accesses are coordinated by the owner.
unsafe impl Send for Buffer {}
// SAFETY: as above; `&Buffer` exposes no interior mutability.
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("size", &self.size).finish()
    }
}

impl Buffer {
    /// Host-visible, coherent and mapped (system memory preferred: written
    /// once by the CPU, read once by the GPU).
    pub fn host(
        device: &ash::Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(1))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create infos; cleanup on every error path.
        unsafe {
            let buffer = device.create_buffer(&info, None)?;
            let req = device.get_buffer_memory_requirements(buffer);
            let flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let types = &props.memory_types[..props.memory_type_count as usize];
            let pick = |avoid: vk::MemoryPropertyFlags| {
                (0..types.len()).find(|&i| {
                    req.memory_type_bits & (1 << i) != 0
                        && types[i].property_flags.contains(flags)
                        && !types[i].property_flags.intersects(avoid)
                })
            };
            let Some(index) = pick(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .or_else(|| pick(vk::MemoryPropertyFlags::empty()))
            else {
                device.destroy_buffer(buffer, None);
                return Err(super::VkError::NoMemoryType);
            };
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(index as u32);
            let memory = match device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(e.into());
                }
            };
            let mapped = device.bind_buffer_memory(buffer, memory, 0).and_then(|()| {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            });
            match mapped {
                Ok(ptr) => Ok(Self {
                    device: device.clone(),
                    buffer,
                    memory,
                    size,
                    ptr: ptr.cast(),
                }),
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                    Err(e.into())
                }
            }
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: owners keep buffers alive until the GPU is done with them;
        // freeing the memory also unmaps it.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
