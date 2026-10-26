//! [`UploadMemory`] for the Vulkan backend: host-visible buffers the capture
//! thread writes into, completion from the device's timeline semaphore.

use std::sync::Arc;

use ash::vk;

use super::texture::Buffer;
use crate::upload::UploadMemory;

pub struct VkUploadMemory {
    pub(crate) device: ash::Device,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) timeline: vk::Semaphore,
}

impl std::fmt::Debug for VkUploadMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VkUploadMemory")
    }
}

impl UploadMemory for VkUploadMemory {
    type Buffer = Arc<Buffer>;

    fn allocate(&self, size: usize) -> Option<(Self::Buffer, *mut u8)> {
        let b = Buffer::host(
            &self.device,
            &self.memory_properties,
            size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .inspect_err(|e| tracing::warn!("upload buffer allocation failed: {e}"))
        .ok()?;
        let ptr = b.ptr;
        Some((Arc::new(b), ptr))
    }

    fn completed(&self) -> u64 {
        // SAFETY: valid timeline semaphore; the query needs no external
        // synchronization, so the capture thread may call it.
        unsafe { self.device.get_semaphore_counter_value(self.timeline) }.unwrap_or(0)
    }
}
