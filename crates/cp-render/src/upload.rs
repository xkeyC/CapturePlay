//! Capture → GPU hand-off: a small ring of persistently mapped upload
//! buffers implementing [`FrameSink`].
//!
//! Slot life cycle: `Free → Writing (capture thread) → Ready → Taken (render
//! thread records the GPU copy) → InFlight(fence) → Free`. When the renderer
//! falls behind, the capture thread recycles the *oldest* ready frame rather
//! than dropping the newest one, so latency never accumulates.

use std::sync::{Arc, Mutex};

use cp_capture::{FrameDest, FrameMeta, FrameSink, PlaneDest};
use cp_core::{PixelFormat, VideoMode};
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::{buffer_desc, heap_props};

pub const SLOT_COUNT: usize = 4;
const PITCH_ALIGN: usize = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize;
const PLACEMENT_ALIGN: usize = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as usize;

/// Byte layout of one frame inside an upload slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotLayout {
    pub planes: [PlaneDest; 2],
    pub plane_count: usize,
    pub total: usize,
}

impl SlotLayout {
    pub fn for_mode(format: PixelFormat, width: u32, height: u32) -> Self {
        let sizes = format.planes(width, height);
        let mut planes = [PlaneDest {
            offset: 0,
            pitch: 0,
        }; 2];
        let mut offset = 0usize;
        for (i, (row_bytes, rows)) in sizes.iter().enumerate() {
            offset = offset.next_multiple_of(PLACEMENT_ALIGN);
            let pitch = row_bytes.next_multiple_of(PITCH_ALIGN);
            planes[i] = PlaneDest { offset, pitch };
            offset += pitch * rows;
        }
        Self {
            planes,
            plane_count: sizes.count,
            total: offset,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Writing,
    Ready,
    Taken,
    InFlight(u64),
}

struct Slot {
    buffer: Option<ID3D12Resource>,
    ptr: *mut u8,
    capacity: usize,
    state: SlotState,
    meta: Option<FrameMeta>,
    layout: SlotLayout,
}

// SAFETY: `ptr` is a persistent mapping of `buffer`, accessed only by the
// thread that currently owns the slot according to `state`.
unsafe impl Send for Slot {}

/// A frame ready for the renderer.
#[derive(Debug)]
pub struct ReadyFrame {
    pub slot: usize,
    pub meta: FrameMeta,
    pub layout: SlotLayout,
}

pub struct UploadRing {
    device: ID3D12Device,
    fence: ID3D12Fence,
    slots: Mutex<Vec<Slot>>,
    wake: Box<dyn Fn() + Send + Sync>,
    recycled: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for UploadRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadRing").finish_non_exhaustive()
    }
}

impl UploadRing {
    /// `wake` is called (on the capture thread) whenever a frame is ready.
    pub fn new(
        device: ID3D12Device,
        fence: ID3D12Fence,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Arc<Self> {
        let slots = (0..SLOT_COUNT)
            .map(|_| Slot {
                buffer: None,
                ptr: std::ptr::null_mut(),
                capacity: 0,
                state: SlotState::Free,
                meta: None,
                layout: SlotLayout {
                    planes: [PlaneDest {
                        offset: 0,
                        pitch: 0,
                    }; 2],
                    plane_count: 0,
                    total: 0,
                },
            })
            .collect();
        Arc::new(Self {
            device,
            fence,
            slots: Mutex::new(slots),
            wake: Box::new(wake),
            recycled: Default::default(),
        })
    }

    /// Frames overwritten before the renderer consumed them.
    pub fn recycled(&self) -> u64 {
        self.recycled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Takes every ready frame, oldest first.
    pub fn take_ready(&self) -> Vec<ReadyFrame> {
        let Ok(mut slots) = self.slots.lock() else {
            return Vec::new();
        };
        let mut out: Vec<ReadyFrame> = slots
            .iter_mut()
            .enumerate()
            .filter(|(_, s)| s.state == SlotState::Ready)
            .filter_map(|(i, s)| {
                s.state = SlotState::Taken;
                s.meta.take().map(|meta| ReadyFrame {
                    slot: i,
                    meta,
                    layout: s.layout,
                })
            })
            .collect();
        out.sort_by_key(|f| f.meta.seq);
        out
    }

    /// Upload buffer of a taken slot (for recording the GPU copy).
    pub fn buffer(&self, slot: usize) -> Option<ID3D12Resource> {
        self.slots.lock().ok()?.get(slot)?.buffer.clone()
    }

    /// Returns a taken slot; the GPU may read it until `fence_value`
    /// completes (`0` = not used by the GPU at all).
    pub fn release(&self, slot: usize, fence_value: u64) {
        if let Ok(mut slots) = self.slots.lock()
            && let Some(s) = slots.get_mut(slot)
        {
            s.state = if fence_value == 0 {
                SlotState::Free
            } else {
                SlotState::InFlight(fence_value)
            };
        }
    }

    fn ensure_capacity(&self, s: &mut Slot, size: usize) -> bool {
        if s.capacity >= size && s.buffer.is_some() {
            return true;
        }
        s.buffer = None;
        s.ptr = std::ptr::null_mut();
        s.capacity = 0;
        let desc = buffer_desc(size as u64, D3D12_RESOURCE_FLAG_NONE);
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: creating and mapping an upload buffer; the mapping stays
        // valid for the resource lifetime (upload heaps may stay mapped).
        unsafe {
            if self
                .device
                .CreateCommittedResource(
                    &heap_props(D3D12_HEAP_TYPE_UPLOAD),
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut res,
                )
                .is_err()
            {
                return false;
            }
            let Some(res) = res else { return false };
            let mut ptr = std::ptr::null_mut();
            let no_read = D3D12_RANGE { Begin: 0, End: 0 };
            if res.Map(0, Some(&no_read), Some(&mut ptr)).is_err() {
                return false;
            }
            s.ptr = ptr.cast();
            s.buffer = Some(res);
        }
        s.capacity = size;
        true
    }
}

impl FrameSink for UploadRing {
    fn acquire(&self, mode: &VideoMode) -> Option<FrameDest> {
        let layout = SlotLayout::for_mode(mode.format, mode.width, mode.height);
        // SAFETY: plain fence query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        let mut slots = self.slots.lock().ok()?;
        for s in slots.iter_mut() {
            if let SlotState::InFlight(v) = s.state
                && v <= completed
            {
                s.state = SlotState::Free;
            }
        }
        let index = slots
            .iter()
            .position(|s| s.state == SlotState::Free)
            .or_else(|| {
                // Renderer is behind: recycle the oldest unconsumed frame.
                let oldest = slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.state == SlotState::Ready)
                    .min_by_key(|(_, s)| s.meta.as_ref().map_or(0, |m| m.seq))
                    .map(|(i, _)| i);
                if oldest.is_some() {
                    self.recycled
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                oldest
            })?;
        let slot = &mut slots[index];
        if !self.ensure_capacity(slot, layout.total) {
            return None;
        }
        slot.state = SlotState::Writing;
        slot.meta = None;
        slot.layout = layout;
        Some(FrameDest {
            slot: index as u32,
            base: slot.ptr,
            len: slot.capacity,
            planes: layout.planes,
        })
    }

    fn publish(&self, dest: FrameDest, meta: FrameMeta) {
        if let Ok(mut slots) = self.slots.lock() {
            let s = &mut slots[dest.slot as usize];
            s.state = SlotState::Ready;
            s.meta = Some(meta);
        }
        (self.wake)();
    }

    fn cancel(&self, dest: FrameDest) {
        if let Ok(mut slots) = self.slots.lock() {
            slots[dest.slot as usize].state = SlotState::Free;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_d3d12_aligned() {
        let l = SlotLayout::for_mode(PixelFormat::Nv12, 1920, 1080);
        assert_eq!(l.plane_count, 2);
        assert_eq!(l.planes[0].pitch % 256, 0);
        assert_eq!(l.planes[1].offset % 512, 0);
        assert!(l.planes[1].offset >= 1920 * 1080);
        let p = SlotLayout::for_mode(PixelFormat::P010, 2560, 1440);
        assert_eq!(p.planes[0].pitch, 5120);
        assert_eq!(p.total, 5120 * 1440 + 5120 * 720);
        let y = SlotLayout::for_mode(PixelFormat::Yuy2, 1366, 768);
        assert_eq!(y.plane_count, 1);
        assert_eq!(y.planes[0].pitch, 2816);
    }
}
