//! Descriptor management.
//!
//! Persistent views live in non-shader-visible "staging" heaps (one slot per
//! view, free list). Each draw/dispatch copies the views it needs into a
//! contiguous table in the shader-visible heap, which is split into one
//! segment per frame in flight and recycled when that frame's fence passes.

use std::cell::RefCell;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

#[derive(Debug, Clone, Copy)]
pub struct CpuDescriptor {
    pub index: u32,
    pub handle: D3D12_CPU_DESCRIPTOR_HANDLE,
}

pub struct StagingHeap {
    heap: ID3D12DescriptorHeap,
    start: D3D12_CPU_DESCRIPTOR_HANDLE,
    increment: usize,
    free: RefCell<Vec<u32>>,
}

impl std::fmt::Debug for StagingHeap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingHeap")
            .field("free", &self.free.borrow().len())
            .finish()
    }
}

impl StagingHeap {
    pub fn new(
        device: &ID3D12Device,
        ty: D3D12_DESCRIPTOR_HEAP_TYPE,
        capacity: u32,
    ) -> Result<Self> {
        // SAFETY: valid device and descriptor.
        unsafe {
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: ty,
                    NumDescriptors: capacity,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })?;
            Ok(Self {
                start: heap.GetCPUDescriptorHandleForHeapStart(),
                increment: device.GetDescriptorHandleIncrementSize(ty) as usize,
                heap,
                free: RefCell::new((0..capacity).rev().collect()),
            })
        }
    }

    pub fn alloc(&self) -> Option<CpuDescriptor> {
        let index = self.free.borrow_mut().pop()?;
        Some(CpuDescriptor {
            index,
            handle: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.start.ptr + index as usize * self.increment,
            },
        })
    }

    pub fn free(&self, d: CpuDescriptor) {
        self.free.borrow_mut().push(d.index);
    }

    pub fn heap(&self) -> &ID3D12DescriptorHeap {
        &self.heap
    }
}

/// Shader-visible CBV/SRV/UAV heap with per-frame linear segments.
pub struct GpuDescriptorRing {
    heap: ID3D12DescriptorHeap,
    cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    increment: u32,
    segment_size: u32,
    segment: u32,
    cursor: u32,
}

impl std::fmt::Debug for GpuDescriptorRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDescriptorRing")
            .field("segment", &self.segment)
            .field("cursor", &self.cursor)
            .finish()
    }
}

impl GpuDescriptorRing {
    pub fn new(device: &ID3D12Device, segments: u32, segment_size: u32) -> Result<Self> {
        // SAFETY: valid device and descriptor.
        unsafe {
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: segments * segment_size,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            Ok(Self {
                cpu_start: heap.GetCPUDescriptorHandleForHeapStart(),
                gpu_start: heap.GetGPUDescriptorHandleForHeapStart(),
                increment: device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                heap,
                segment_size,
                segment: 0,
                cursor: 0,
            })
        }
    }

    pub fn heap(&self) -> &ID3D12DescriptorHeap {
        &self.heap
    }

    /// Starts using `segment` (the caller guarantees the GPU is done with it).
    pub fn begin_segment(&mut self, segment: u32) {
        self.segment = segment;
        self.cursor = 0;
    }

    /// Copies `views` into a contiguous table and returns its GPU handle.
    pub fn table(
        &mut self,
        device: &ID3D12Device,
        views: &[D3D12_CPU_DESCRIPTOR_HANDLE],
    ) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let n = views.len() as u32;
        assert!(
            self.cursor + n <= self.segment_size,
            "descriptor segment overflow"
        );
        let index = self.segment * self.segment_size + self.cursor;
        self.cursor += n;
        for (i, v) in views.iter().enumerate() {
            let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.cpu_start.ptr + (index as usize + i) * self.increment as usize,
            };
            // SAFETY: both handles are valid descriptors of the same type.
            unsafe {
                device.CopyDescriptorsSimple(1, dst, *v, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            };
        }
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu_start.ptr + u64::from(index) * u64::from(self.increment),
        }
    }
}
