//! Per-frame linear allocator in a persistently mapped host buffer: shader
//! constants, egui geometry and texture uploads. Reset once the frame's
//! timeline value has completed.

use ash::vk;

use super::Result;
use super::texture::Buffer;

pub struct Arena {
    pub buffer: Buffer,
    head: u64,
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena")
            .field("used", &self.head)
            .field("size", &self.buffer.size)
            .finish()
    }
}

/// A region handed out by [`Arena::alloc`].
#[derive(Debug, Clone, Copy)]
pub struct Allocation {
    pub ptr: *mut u8,
    pub offset: u64,
    pub size: u64,
}

impl Arena {
    pub fn new(
        device: &ash::Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
    ) -> Result<Self> {
        let usage = vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC;
        Ok(Self {
            buffer: Buffer::host(device, props, size, usage)?,
            head: 0,
        })
    }

    pub fn reset(&mut self) {
        self.head = 0;
    }

    pub fn alloc(&mut self, size: u64, align: u64) -> Option<Allocation> {
        let offset = self.head.next_multiple_of(align.max(1));
        if offset + size > self.buffer.size {
            return None;
        }
        self.head = offset + size;
        Some(Allocation {
            // SAFETY: `offset + size` lies within the mapping.
            ptr: unsafe { self.buffer.ptr.add(offset as usize) },
            offset,
            size,
        })
    }

    /// Copies `data` into a fresh allocation.
    pub fn push<T: bytemuck::Pod>(&mut self, data: &[T], align: u64) -> Option<Allocation> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let a = self.alloc(bytes.len() as u64, align)?;
        // SAFETY: the allocation is `bytes.len()` bytes of mapped memory not
        // in use by the GPU (the frame's previous work completed).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), a.ptr, bytes.len()) };
        Some(a)
    }

    /// Shader constants as a uniform-buffer descriptor.
    pub fn constants<T: bytemuck::Pod>(
        &mut self,
        c: &T,
        align: u64,
    ) -> Option<vk::DescriptorBufferInfo> {
        let a = self.push(std::slice::from_ref(c), align.max(256))?;
        Some(vk::DescriptorBufferInfo {
            buffer: self.buffer.buffer,
            offset: a.offset,
            range: a.size,
        })
    }
}
