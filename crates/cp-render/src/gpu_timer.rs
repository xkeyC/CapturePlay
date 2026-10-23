//! GPU timestamp pairs per frame-in-flight, read back without stalling.
//!
//! Work on other engines (NVOFA, CUDA) is included in a measured span when
//! the direct queue waits for it inside the span.

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

use crate::gpu::Gpu;

pub struct GpuTimer {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    ptr: *const u64,
    ticks_per_ms: f64,
    pending: Vec<bool>,
}

impl std::fmt::Debug for GpuTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuTimer")
    }
}

impl GpuTimer {
    pub fn new(gpu: &Gpu, slots: usize) -> Result<Self> {
        // SAFETY: plain query heap / readback buffer creation; the readback
        // buffer stays mapped and is only read after its frame's fence.
        unsafe {
            let mut heap: Option<ID3D12QueryHeap> = None;
            gpu.device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: (slots * 2) as u32,
                    NodeMask: 0,
                },
                &mut heap,
            )?;
            let readback = gpu.create_buffer(
                (slots * 16) as u64,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            let mut ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr))?;
            let freq = gpu.queue.GetTimestampFrequency()?;
            Ok(Self {
                heap: heap.ok_or_else(windows::core::Error::empty)?,
                readback,
                ptr: ptr.cast(),
                ticks_per_ms: freq as f64 / 1000.0,
                pending: vec![false; slots],
            })
        }
    }

    pub fn begin(&self, list: &ID3D12GraphicsCommandList, slot: usize) {
        // SAFETY: index within the heap.
        unsafe { list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, (slot * 2) as u32) };
    }

    pub fn end(&mut self, list: &ID3D12GraphicsCommandList, slot: usize) {
        // SAFETY: indices/offsets within the heap and readback buffer.
        unsafe {
            list.EndQuery(
                &self.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                (slot * 2 + 1) as u32,
            );
            list.ResolveQueryData(
                &self.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                (slot * 2) as u32,
                2,
                &self.readback,
                (slot * 16) as u64,
            );
        }
        self.pending[slot] = true;
    }

    /// Elapsed ms of `slot`'s span. Call only after that frame's fence.
    pub fn read(&mut self, slot: usize) -> Option<f64> {
        if !std::mem::take(&mut self.pending[slot]) {
            return None;
        }
        // SAFETY: the GPU finished writing this slot (caller waited on the
        // frame fence); the mapping covers `slots * 16` bytes.
        let (a, b) = unsafe { (*self.ptr.add(slot * 2), *self.ptr.add(slot * 2 + 1)) };
        (b > a).then(|| (b - a) as f64 / self.ticks_per_ms)
    }
}
