//! Vulkan instance, device, the single graphics/compute/present queue and a
//! timeline semaphore playing the role of the D3D12 device fence.

use std::cell::Cell;
use std::ffi::{CStr, c_char};

use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::{Result, VkError};

pub const VENDOR_NVIDIA: u32 = 0x10de;

/// Optional device capabilities that change how the renderer works.
#[derive(Debug, Clone, Copy, Default)]
pub struct Features {
    /// `VK_KHR_present_id` + `VK_KHR_present_wait`.
    pub present_wait: bool,
    /// `VK_EXT_swapchain_colorspace` (scRGB and HDR10 surface formats).
    pub swapchain_colorspace: bool,
}

/// The device and its queue. Owned by the render thread; the timeline
/// semaphore is also read by the capture thread (vkGetSemaphoreCounterValue
/// needs no external synchronization).
pub struct Gpu {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub adapter_name: String,
    pub vendor_id: u32,
    pub features: Features,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    pub min_uniform_alignment: u64,
    pub surface_fn: ash::khr::surface::Instance,
    pub swapchain_fn: ash::khr::swapchain::Device,
    pub push_fn: ash::khr::push_descriptor::Device,
    pub present_wait_fn: Option<ash::khr::present_wait::Device>,
    /// Signaled with increasing values by every submission.
    pub timeline: vk::Semaphore,
    next_value: Cell<u64>,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu")
            .field("adapter", &self.adapter_name)
            .field("features", &self.features)
            .finish()
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes a valid callback-data struct whose message
    // is a NUL-terminated string.
    let msg = unsafe { data.as_ref() }
        .and_then(|d| (!d.p_message.is_null()).then(|| unsafe { CStr::from_ptr(d.p_message) }))
        .map(|m| m.to_string_lossy())
        .unwrap_or_default();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!("vulkan: {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        tracing::warn!("vulkan: {msg}");
    } else {
        tracing::debug!("vulkan: {msg}");
    }
    vk::FALSE
}

fn has_ext(available: &[vk::ExtensionProperties], name: &CStr) -> bool {
    available
        .iter()
        .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
}

impl Gpu {
    /// Creates the instance and a surface for `window`, then a device on the
    /// best adapter that can present to it (discrete NVIDIA first).
    pub fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        debug_layer: bool,
    ) -> Result<(Self, vk::SurfaceKHR)> {
        let (gpu, surface) = Self::create(Some((display, window)), debug_layer)?;
        Ok((gpu, surface.ok_or(VkError::NoDevice)?))
    }

    /// A device without a window (tests, offline tools).
    pub fn headless(debug_layer: bool) -> Result<Self> {
        Ok(Self::create(None, debug_layer)?.0)
    }

    fn create(
        target: Option<(RawDisplayHandle, RawWindowHandle)>,
        debug_layer: bool,
    ) -> Result<(Self, Option<vk::SurfaceKHR>)> {
        // SAFETY: loads the system Vulkan loader; every create call below gets
        // fully initialized create-info structs that outlive the call.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| VkError::Load(e.to_string()))?;
            let available = entry.enumerate_instance_extension_properties(None)?;
            let mut exts: Vec<*const c_char> = match target {
                Some((display, _)) => ash_window::enumerate_required_extensions(display)?.to_vec(),
                None => Vec::new(),
            };
            // Requires VK_KHR_surface, i.e. a window.
            let swapchain_colorspace =
                target.is_some() && has_ext(&available, ash::ext::swapchain_colorspace::NAME);
            if swapchain_colorspace {
                exts.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
            }
            let debug_layer = debug_layer && has_ext(&available, ash::ext::debug_utils::NAME);
            let validation = c"VK_LAYER_KHRONOS_validation";
            let layers: Vec<*const c_char> = if debug_layer {
                exts.push(ash::ext::debug_utils::NAME.as_ptr());
                let present = entry
                    .enumerate_instance_layer_properties()?
                    .iter()
                    .any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == validation));
                if !present {
                    tracing::warn!("Vulkan validation layer not installed");
                }
                present.then_some(validation.as_ptr()).into_iter().collect()
            } else {
                Vec::new()
            };
            let app = vk::ApplicationInfo::default()
                .application_name(c"CapturePlay")
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(c"CapturePlay")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&exts)
                    .enabled_layer_names(&layers),
                None,
            )?;
            let debug = if debug_layer {
                let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(debug_callback));
                let m = du.create_debug_utils_messenger(&info, None)?;
                Some((du, m))
            } else {
                None
            };
            let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
            let surface = match target {
                Some((display, window)) => Some(ash_window::create_surface(
                    &entry, &instance, display, window, None,
                )?),
                None => None,
            };

            let (physical, queue_family, props) =
                Self::pick_physical(&instance, &surface_fn, surface)?;
            let adapter_name = props
                .device_name_as_c_str()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            let dev_exts = instance.enumerate_device_extension_properties(physical)?;
            let mut enable: Vec<*const c_char> = vec![ash::khr::push_descriptor::NAME.as_ptr()];
            if target.is_some() {
                enable.push(ash::khr::swapchain::NAME.as_ptr());
            }
            let present_wait = target.is_some()
                && has_ext(&dev_exts, ash::khr::present_id::NAME)
                && has_ext(&dev_exts, ash::khr::present_wait::NAME);
            if present_wait {
                enable.push(ash::khr::present_id::NAME.as_ptr());
                enable.push(ash::khr::present_wait::NAME.as_ptr());
            }

            let priorities = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let base = vk::PhysicalDeviceFeatures::default()
                .shader_storage_image_write_without_format(true)
                .shader_storage_image_read_without_format(true);
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true);
            let mut pid = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
            let mut pwait = vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&enable)
                .enabled_features(&base)
                .push_next(&mut v12)
                .push_next(&mut v13);
            if present_wait {
                info = info.push_next(&mut pid).push_next(&mut pwait);
            }
            let device = instance.create_device(physical, &info, None)?;
            let queue = device.get_device_queue(queue_family, 0);

            let mut ty = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let timeline = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)?;

            let gpu = Self {
                memory_properties: instance.get_physical_device_memory_properties(physical),
                timestamp_period: props.limits.timestamp_period,
                min_uniform_alignment: props.limits.min_uniform_buffer_offset_alignment,
                swapchain_fn: ash::khr::swapchain::Device::new(&instance, &device),
                push_fn: ash::khr::push_descriptor::Device::new(&instance, &device),
                present_wait_fn: present_wait
                    .then(|| ash::khr::present_wait::Device::new(&instance, &device)),
                surface_fn,
                features: Features {
                    present_wait,
                    swapchain_colorspace,
                },
                vendor_id: props.vendor_id,
                adapter_name,
                queue,
                queue_family,
                device,
                physical,
                instance,
                entry,
                timeline,
                next_value: Cell::new(1),
                debug,
            };
            tracing::info!(adapter = %gpu.adapter_name, features = ?gpu.features, "Vulkan device created");
            Ok((gpu, surface))
        }
    }

    /// Discrete before integrated before anything else, NVIDIA first within a
    /// class; the device needs Vulkan 1.3 and a queue family with graphics,
    /// compute and (with a surface) present support.
    unsafe fn pick_physical(
        instance: &ash::Instance,
        surface_fn: &ash::khr::surface::Instance,
        surface: Option<vk::SurfaceKHR>,
    ) -> Result<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> {
        let mut best: Option<(u32, vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> = None;
        // SAFETY: plain enumeration on a valid instance and surface.
        unsafe {
            for pd in instance.enumerate_physical_devices()? {
                let props = instance.get_physical_device_properties(pd);
                if props.api_version < vk::API_VERSION_1_3 {
                    continue;
                }
                let families = instance.get_physical_device_queue_family_properties(pd);
                let Some(family) = families.iter().enumerate().find_map(|(i, f)| {
                    let i = i as u32;
                    let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
                    let present = surface.is_none_or(|s| {
                        surface_fn
                            .get_physical_device_surface_support(pd, i, s)
                            .unwrap_or(false)
                    });
                    (f.queue_flags.contains(flags) && present).then_some(i)
                }) else {
                    continue;
                };
                let class = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 0,
                    _ => 1,
                };
                let score = class * 2 + u32::from(props.vendor_id == VENDOR_NVIDIA);
                if best.as_ref().is_none_or(|b| score > b.0) {
                    best = Some((score, pd, family, props));
                }
            }
        }
        best.map(|(_, pd, f, p)| (pd, f, p))
            .ok_or(VkError::NoDevice)
    }

    /// Memory type for `bits` with `flags`, preferring `preferred` on top.
    pub fn memory_type(
        &self,
        bits: u32,
        flags: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let types = &self.memory_properties.memory_types
            [..self.memory_properties.memory_type_count as usize];
        let ok = |i: usize, f: vk::MemoryPropertyFlags| {
            bits & (1 << i) != 0 && types[i].property_flags.contains(f)
        };
        (0..types.len())
            .find(|&i| ok(i, flags | preferred))
            .or_else(|| (0..types.len()).find(|&i| ok(i, flags)))
            .map(|i| i as u32)
    }

    /// Allocates memory for `req`.
    pub fn allocate(
        &self,
        req: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory> {
        let index = self
            .memory_type(req.memory_type_bits, flags, preferred)
            .ok_or(VkError::NoMemoryType)?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(index);
        // SAFETY: valid allocate info for this device.
        Ok(unsafe { self.device.allocate_memory(&info, None)? })
    }

    /// Value the next submission will signal.
    pub fn next_value(&self) -> u64 {
        self.next_value.get()
    }

    /// Submits `cmd`, waiting on `waits` (binary semaphores, at the given
    /// stages) and signaling `signal` (binary) plus the timeline. Returns the
    /// timeline value that marks completion.
    pub fn submit(
        &self,
        cmd: vk::CommandBuffer,
        waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        signal: &[vk::Semaphore],
    ) -> Result<u64> {
        let value = self.next_value.get();
        self.next_value.set(value + 1);
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let wait_infos: Vec<_> = waits
            .iter()
            .map(|&(s, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s)
                    .stage_mask(stage)
            })
            .collect();
        let mut signal_infos: Vec<_> = signal
            .iter()
            .map(|&s| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        signal_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmds)
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(&signal_infos);
        // SAFETY: the queue is only used by the render thread; the command
        // buffer is fully recorded and its resources outlive the submission
        // (callers wait on the returned value before releasing them).
        unsafe {
            self.device
                .queue_submit2(self.queue, &[submit], vk::Fence::null())?
        };
        Ok(value)
    }

    /// Latest completed timeline value.
    pub fn completed(&self) -> u64 {
        // SAFETY: valid timeline semaphore.
        unsafe { self.device.get_semaphore_counter_value(self.timeline) }.unwrap_or(0)
    }

    /// Blocks until the timeline reaches `value`.
    pub fn wait(&self, value: u64) -> Result<()> {
        if value == 0 || self.completed() >= value {
            return Ok(());
        }
        let semaphores = [self.timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: valid semaphore; the value was or will be signaled by a
        // submission already queued.
        unsafe { self.device.wait_semaphores(&info, u64::MAX)? };
        Ok(())
    }

    /// Waits for all submitted work.
    pub fn wait_idle(&self) -> Result<()> {
        self.wait(self.next_value.get() - 1)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: every child object was destroyed by its owner before the
        // GPU (the renderer drops its users first).
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_semaphore(self.timeline, None);
            self.device.destroy_device(None);
            if let Some((du, m)) = self.debug.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
