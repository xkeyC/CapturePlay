//! Per-frame linear upload allocator (vertex/index data, texture updates).

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

use crate::gpu::Gpu;

pub struct UploadArena {
    pub buffer: ID3D12Resource,
    ptr: *mut u8,
    gpu_va: u64,
    capacity: usize,
    cursor: usize,
}

impl std::fmt::Debug for UploadArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadArena")
            .field("capacity", &self.capacity)
            .field("cursor", &self.cursor)
            .finish()
    }
}

/// A sub-allocation inside the arena.
#[derive(Debug, Clone, Copy)]
pub struct ArenaAlloc {
    pub offset: usize,
    pub gpu_va: u64,
    pub ptr: *mut u8,
}

impl UploadArena {
    pub fn new(gpu: &Gpu, capacity: usize) -> Result<Self> {
        let buffer = gpu.create_buffer(
            capacity as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: upload heaps may stay persistently mapped.
        unsafe {
            buffer.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut ptr))?;
        }
        // SAFETY: valid buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        Ok(Self {
            buffer,
            ptr: ptr.cast(),
            gpu_va,
            capacity,
            cursor: 0,
        })
    }

    /// Recycles the whole arena; the frame's fence must have completed.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn alloc(&mut self, size: usize, align: usize) -> Option<ArenaAlloc> {
        let offset = self.cursor.next_multiple_of(align);
        if offset + size > self.capacity {
            return None;
        }
        self.cursor = offset + size;
        // SAFETY: offset + size is within the mapped range.
        let ptr = unsafe { self.ptr.add(offset) };
        Some(ArenaAlloc {
            offset,
            gpu_va: self.gpu_va + offset as u64,
            ptr,
        })
    }

    /// Copies a slice into the arena.
    pub fn push<T: bytemuck::Pod>(&mut self, data: &[T], align: usize) -> Option<ArenaAlloc> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let a = self.alloc(bytes.len(), align)?;
        // SAFETY: freshly reserved, non-overlapping region of the mapping.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), a.ptr, bytes.len()) };
        Some(a)
    }
}
