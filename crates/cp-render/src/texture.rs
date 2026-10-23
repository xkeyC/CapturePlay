//! GPU textures with persistent views and tracked resource state.

use std::cell::Cell;
use std::rc::Rc;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Result;

use crate::descriptors::{CpuDescriptor, StagingHeap};
use crate::gpu::{Gpu, transition};

pub struct Texture {
    pub resource: ID3D12Resource,
    pub width: u32,
    pub height: u32,
    pub format: DXGI_FORMAT,
    pub srv: CpuDescriptor,
    pub uav: Option<CpuDescriptor>,
    state: Cell<D3D12_RESOURCE_STATES>,
    heap: Rc<StagingHeap>,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Texture({}x{} {:?})",
            self.width, self.height, self.format
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: DXGI_FORMAT,
    /// SRV format if different from `format` (e.g. an `_SRGB` view).
    pub srv_format: Option<DXGI_FORMAT>,
    pub uav: bool,
    pub initial: D3D12_RESOURCE_STATES,
}

impl TextureDesc {
    pub fn new(width: u32, height: u32, format: DXGI_FORMAT) -> Self {
        Self {
            width,
            height,
            format,
            srv_format: None,
            uav: false,
            initial: D3D12_RESOURCE_STATE_COMMON,
        }
    }

    pub fn uav(mut self) -> Self {
        self.uav = true;
        self
    }
}

impl Texture {
    pub fn new(gpu: &Gpu, heap: &Rc<StagingHeap>, d: TextureDesc) -> Result<Self> {
        let flags = if d.uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        };
        let resource = gpu.create_texture(d.width, d.height, d.format, flags, d.initial)?;
        let srv = heap.alloc().ok_or_else(windows::core::Error::empty)?;
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: d.srv_format.unwrap_or(d.format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        // SAFETY: descriptor handles come from our heap; resource is valid.
        unsafe {
            gpu.device
                .CreateShaderResourceView(&resource, Some(&srv_desc), srv.handle)
        };
        let uav = if d.uav {
            let u = heap.alloc().ok_or_else(windows::core::Error::empty)?;
            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: d.format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV::default(),
                },
            };
            // SAFETY: as above.
            unsafe {
                gpu.device
                    .CreateUnorderedAccessView(&resource, None, Some(&uav_desc), u.handle)
            };
            Some(u)
        } else {
            None
        };
        Ok(Self {
            resource,
            width: d.width,
            height: d.height,
            format: d.format,
            srv,
            uav,
            state: Cell::new(d.initial),
            heap: heap.clone(),
        })
    }

    pub fn state(&self) -> D3D12_RESOURCE_STATES {
        self.state.get()
    }

    /// Appends a transition to `state` if needed.
    pub fn transition(
        &self,
        barriers: &mut Vec<D3D12_RESOURCE_BARRIER>,
        state: D3D12_RESOURCE_STATES,
    ) {
        let cur = self.state.get();
        if cur != state {
            barriers.push(transition(&self.resource, cur, state));
            self.state.set(state);
        }
    }

    /// Records the state the resource was left in by someone else (e.g.
    /// decay to COMMON after an external queue used it).
    pub fn assume_state(&self, state: D3D12_RESOURCE_STATES) {
        self.state.set(state);
    }

    pub fn uav_handle(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.uav.expect("texture created without UAV").handle
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        self.heap.free(self.srv);
        if let Some(u) = self.uav {
            self.heap.free(u);
        }
    }
}

/// Records and flushes resource barriers in one call.
pub fn flush_barriers(
    list: &ID3D12GraphicsCommandList,
    barriers: &mut Vec<D3D12_RESOURCE_BARRIER>,
) {
    if !barriers.is_empty() {
        // SAFETY: barrier structs borrow resources that outlive the call.
        unsafe { list.ResourceBarrier(barriers) };
        barriers.clear();
    }
}
