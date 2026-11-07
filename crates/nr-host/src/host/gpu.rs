//! Vulkan through winevulkan: the instance, the NVIDIA device with the
//! extensions NGX and the fd export need, images and one-shot submits.

use std::ffi::CStr;

use ash::vk;

pub type Result<T> = std::result::Result<T, String>;

fn vk_err(what: &'static str) -> impl Fn(vk::Result) -> String {
    move |e| format!("{what}: {e:?}")
}

/// Device extensions NGX's Vulkan path relies on.
const NGX_EXTENSIONS: [&CStr; 3] = [
    c"VK_NVX_binary_import",
    c"VK_NVX_image_view_handle",
    c"VK_KHR_push_descriptor",
];

/// Extensions that move memory and semaphores across processes, reported
/// and enabled when offered.
pub const SHARING_EXTENSIONS: [&CStr; 6] = [
    c"VK_KHR_external_memory_fd",
    c"VK_EXT_external_memory_dma_buf",
    c"VK_KHR_external_semaphore_fd",
    c"VK_KHR_external_memory_win32",
    c"VK_KHR_external_semaphore_win32",
    c"VK_EXT_external_memory_host",
];

pub struct Gpu {
    _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub family: u32,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    /// Sharing extensions the device offers (and that were enabled).
    pub sharing: Vec<&'static CStr>,
    pool: vk::CommandPool,
    pub name: String,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        // SAFETY: loads vulkan-1.dll (winevulkan under Wine).
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("vulkan-1.dll: {e}"))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"captureplay-nr-host")
            .api_version(vk::API_VERSION_1_3);
        // SAFETY: plain instance creation.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(vk_err("vkCreateInstance"))?;
        // SAFETY: valid instance.
        let physicals = unsafe { instance.enumerate_physical_devices() }
            .map_err(vk_err("vkEnumeratePhysicalDevices"))?;
        let has = |p: vk::PhysicalDevice, name: &CStr| {
            // SAFETY: valid physical device.
            unsafe { instance.enumerate_device_extension_properties(p) }
                .unwrap_or_default()
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let physical = physicals
            .iter()
            .copied()
            .find(|&p| {
                // SAFETY: valid physical device.
                let props = unsafe { instance.get_physical_device_properties(p) };
                props.vendor_id == 0x10de && NGX_EXTENSIONS.iter().all(|e| has(p, e))
            })
            .ok_or("no NVIDIA device offering VK_NVX_binary_import / VK_NVX_image_view_handle")?;
        // SAFETY: valid physical device.
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = props
            .device_name_as_c_str()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let sharing: Vec<&'static CStr> = SHARING_EXTENSIONS
            .iter()
            .copied()
            .filter(|e| has(physical, e))
            .collect();

        // SAFETY: valid physical device.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|f| {
                f.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .ok_or("no graphics+compute queue")? as u32;

        let extensions: Vec<*const i8> = NGX_EXTENSIONS
            .iter()
            .chain(&sharing)
            .map(|e| e.as_ptr())
            .collect();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        // SAFETY: valid physical device; extensions were checked above.
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&extensions)
                    .push_next(&mut v12)
                    .push_next(&mut v13),
                None,
            )
        }
        .map_err(vk_err("vkCreateDevice"))?;
        // SAFETY: the queue family was requested above.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // SAFETY: valid physical device.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        // SAFETY: plain object creation.
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(family),
                None,
            )
        }
        .map_err(vk_err("vkCreateCommandPool"))?;
        Ok(Self {
            _entry: entry,
            instance,
            physical,
            device,
            queue,
            family,
            memory,
            sharing,
            pool,
            name,
        })
    }

    /// `VkPhysicalDeviceIDProperties::deviceUUID` as hex (the native side
    /// picks the same device by it).
    pub fn uuid_hex(&self) -> String {
        self.uuid().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// `VkPhysicalDeviceIDProperties::deviceUUID`.
    pub fn uuid(&self) -> [u8; 16] {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: valid physical device.
        unsafe {
            self.instance
                .get_physical_device_properties2(self.physical, &mut props);
        }
        id.device_uuid
    }

    pub fn has(&self, extension: &CStr) -> bool {
        self.sharing.contains(&extension)
    }

    pub fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        (0..self.memory.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && self.memory.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or_else(|| format!("no memory type {flags:?} in {bits:#x}"))
    }

    /// Records with `record` into a fresh command buffer, submits it and
    /// waits for completion.
    pub fn run(&self, record: impl FnOnce(vk::CommandBuffer) -> Result<()>) -> Result<()> {
        let d = &self.device;
        // SAFETY: our pool; the buffer is freed after the wait below.
        unsafe {
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .command_buffer_count(1),
                )
                .map_err(vk_err("vkAllocateCommandBuffers"))?[0];
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_err("vkBeginCommandBuffer"))?;
            let recorded = record(cmd);
            d.end_command_buffer(cmd)
                .map_err(vk_err("vkEndCommandBuffer"))?;
            recorded?;
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(vk_err("vkCreateFence"))?;
            let cmds = [cmd];
            let r = d
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cmds)],
                    fence,
                )
                .and_then(|()| d.wait_for_fences(&[fence], true, u64::MAX));
            d.destroy_fence(fence, None);
            d.free_command_buffers(self.pool, &cmds);
            r.map_err(vk_err("vkQueueSubmit / vkWaitForFences"))
        }
    }

    /// `n` command buffers with signalled fences, for [`Ring`].
    pub fn ring(&self, n: usize) -> Result<Ring> {
        let d = &self.device;
        // SAFETY: our pool; objects are destroyed by `Ring::drop`.
        unsafe {
            let cmds = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .command_buffer_count(n as u32),
                )
                .map_err(vk_err("vkAllocateCommandBuffers"))?;
            let mut slots = Vec::with_capacity(n);
            for cmd in cmds {
                let fence = d
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .map_err(vk_err("vkCreateFence"))?;
                slots.push((cmd, fence));
            }
            Ok(Ring {
                device: d.clone(),
                pool: self.pool,
                slots,
                next: 0,
            })
        }
    }

    /// Submits `cmd` waiting for `wait` (timeline value) and signalling
    /// `signal`, then `fence`.
    pub fn submit_timeline(
        &self,
        cmd: vk::CommandBuffer,
        wait: (vk::Semaphore, u64),
        signal: (vk::Semaphore, u64),
        fence: vk::Fence,
    ) -> Result<()> {
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(wait.0)
            .value(wait.1)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(signal.0)
            .value(signal.1)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        // SAFETY: a fully recorded command buffer and live semaphores.
        unsafe {
            self.device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&waits)
                    .command_buffer_infos(&cmds)
                    .signal_semaphore_infos(&signals)],
                fence,
            )
        }
        .map_err(vk_err("vkQueueSubmit2"))
    }

    /// Full barrier between whatever came before and after (every image
    /// stays in GENERAL).
    pub fn barrier(&self, cmd: vk::CommandBuffer) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        // SAFETY: recording command buffer.
        unsafe {
            self.device
                .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: owners dropped their objects; the device is idle.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Command buffers reused round-robin, each guarded by a fence.
pub struct Ring {
    device: ash::Device,
    pool: vk::CommandPool,
    slots: Vec<(vk::CommandBuffer, vk::Fence)>,
    next: usize,
}

impl Ring {
    /// The next command buffer, reset and begun, with its (unsignalled)
    /// fence for the submit.
    pub fn begin(&mut self) -> Result<(vk::CommandBuffer, vk::Fence)> {
        let (cmd, fence) = self.slots[self.next];
        self.next = (self.next + 1) % self.slots.len();
        let d = &self.device;
        // SAFETY: the fence guards the buffer's previous submission.
        unsafe {
            d.wait_for_fences(&[fence], true, u64::MAX)
                .map_err(vk_err("vkWaitForFences"))?;
            d.reset_fences(&[fence]).map_err(vk_err("vkResetFences"))?;
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(vk_err("vkResetCommandBuffer"))?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_err("vkBeginCommandBuffer"))?;
        }
        Ok((cmd, fence))
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // SAFETY: waits for every slot before freeing it.
        unsafe {
            for &(cmd, fence) in &self.slots {
                let _ = self.device.wait_for_fences(&[fence], true, u64::MAX);
                self.device.destroy_fence(fence, None);
                self.device.free_command_buffers(self.pool, &[cmd]);
            }
        }
    }
}

/// Ownership hand-over of `image` (kept in GENERAL) between this queue
/// family and the other process: `acquire` takes it from the external
/// owner, otherwise it is released to it.
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
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(src)
        .dst_queue_family_index(dst)
        .image(image)
        .subresource_range(color_range())];
    // SAFETY: recording command buffer.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&b),
        );
    }
}

/// A 2D image in GENERAL layout with its own memory, optionally exportable.
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// Allocation size (what an importer must allocate).
    pub size: u64,
    /// Linux fd of exportable memory (see [`super::share`]).
    pub fd: Option<i32>,
    device: ash::Device,
}

impl Image {
    pub fn new(
        gpu: &Gpu,
        format: vk::Format,
        (width, height): (u32, u32),
        export: Option<vk::ExternalMemoryHandleTypeFlags>,
    ) -> Result<Self> {
        let d = &gpu.device;
        let mut external = vk::ExternalMemoryImageCreateInfo::default();
        let mut info = vk::ImageCreateInfo::default()
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
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if let Some(types) = export {
            external = external.handle_types(types);
            info = info.push_next(&mut external);
        }
        // SAFETY: valid device and create info.
        let image = unsafe { d.create_image(&info, None) }.map_err(vk_err("vkCreateImage"))?;
        // SAFETY: valid image.
        let req = unsafe { d.get_image_memory_requirements(image) };
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut exported = vk::ExportMemoryAllocateInfo::default();
        let mut alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(
                gpu.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?,
            )
            .push_next(&mut dedicated);
        if let Some(types) = export {
            exported = exported.handle_types(types);
            alloc = alloc.push_next(&mut exported);
        }
        // SAFETY: valid allocation info.
        let allocate =
            || unsafe { d.allocate_memory(&alloc, None) }.map_err(vk_err("vkAllocateMemory"));
        let (memory, fd) = if export.is_some() {
            let (m, fd) = super::share::with_new_fd(allocate)?;
            (m, Some(fd))
        } else {
            (allocate()?, None)
        };
        // SAFETY: fresh image and memory.
        unsafe { d.bind_image_memory(image, memory, 0) }.map_err(vk_err("vkBindImageMemory"))?;
        // SAFETY: valid image.
        let view = unsafe {
            d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(color_range()),
                None,
            )
        }
        .map_err(vk_err("vkCreateImageView"))?;
        let img = Self {
            image,
            view,
            memory,
            format,
            width,
            height,
            size: req.size,
            fd,
            device: d.clone(),
        };
        gpu.run(|cmd| {
            let b = [vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(img.image)
                .subresource_range(color_range())];
            // SAFETY: recording command buffer.
            unsafe {
                d.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&b),
                );
            }
            Ok(())
        })?;
        Ok(img)
    }

    pub fn region(&self) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: owners wait for the GPU before dropping images.
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// A host-visible buffer for uploads and readbacks.
pub struct Staging {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub size: usize,
    device: ash::Device,
}

impl Staging {
    pub fn new(gpu: &Gpu, size: usize) -> Result<Self> {
        let d = &gpu.device;
        // SAFETY: valid device.
        let buffer = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
        }
        .map_err(vk_err("vkCreateBuffer"))?;
        // SAFETY: valid buffer.
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let ty = gpu.memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // SAFETY: valid allocation info; the mapping lives as long as the memory.
        unsafe {
            let memory = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .map_err(vk_err("vkAllocateMemory"))?;
            d.bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_err("vkBindBufferMemory"))?;
            let ptr = d
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_err("vkMapMemory"))?
                .cast();
            Ok(Self {
                buffer,
                memory,
                ptr,
                size,
                device: d.clone(),
            })
        }
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: mapped, coherent, `size` bytes long.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; exclusive borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // SAFETY: owners wait for the GPU before dropping buffers.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
