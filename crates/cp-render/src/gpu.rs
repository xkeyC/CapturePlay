//! D3D12 device, queue, fences and small resource helpers.

use std::cell::Cell;
use std::mem::ManuallyDrop;

use windows::Win32::Foundation::{CloseHandle, HANDLE, LUID};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, Result};

/// The device and its direct queue. Owned by the render thread; the fence
/// is also read by the capture thread (fences are free-threaded).
pub struct Gpu {
    pub factory: IDXGIFactory6,
    pub adapter: IDXGIAdapter1,
    pub adapter_name: String,
    pub adapter_luid: LUID,
    pub vendor_id: u32,
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    pub fence: ID3D12Fence,
    next_value: Cell<u64>,
    event: HANDLE,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu")
            .field("adapter", &self.adapter_name)
            .finish()
    }
}

pub const VENDOR_NVIDIA: u32 = 0x10de;

impl Gpu {
    /// Creates a device on the highest-performance hardware adapter.
    pub fn new(debug_layer: bool) -> Result<Self> {
        // SAFETY: standard D3D12/DXGI initialization sequence; all out
        // pointers are valid locals.
        unsafe {
            if debug_layer {
                let mut dbg: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut dbg).is_ok()
                    && let Some(d) = dbg
                {
                    d.EnableDebugLayer();
                }
            }
            let flags = if debug_layer {
                DXGI_CREATE_FACTORY_DEBUG
            } else {
                DXGI_CREATE_FACTORY_FLAGS(0)
            };
            let factory: IDXGIFactory6 = CreateDXGIFactory2(flags)?;
            let mut chosen = None;
            for i in 0.. {
                let Ok(adapter) = factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    i,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ) else {
                    break;
                };
                let desc = adapter.GetDesc1()?;
                if (DXGI_ADAPTER_FLAG(desc.Flags as i32) & DXGI_ADAPTER_FLAG_SOFTWARE)
                    != DXGI_ADAPTER_FLAG_NONE
                {
                    continue;
                }
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok()
                    && let Some(dev) = dev
                {
                    chosen = Some((adapter, desc, dev));
                    break;
                }
            }
            let (adapter, desc, device) =
                chosen.ok_or_else(|| windows::core::Error::from_hresult(DXGI_ERROR_NOT_FOUND))?;
            let name_len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let adapter_name = String::from_utf16_lossy(&desc.Description[..name_len]);

            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                    ..Default::default()
                })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            tracing::info!(adapter = %adapter_name, "D3D12 device created");
            Ok(Self {
                factory,
                adapter,
                adapter_name,
                adapter_luid: desc.AdapterLuid,
                vendor_id: desc.VendorId,
                device,
                queue,
                fence,
                next_value: Cell::new(1),
                event,
            })
        }
    }

    pub fn is_nvidia(&self) -> bool {
        self.vendor_id == VENDOR_NVIDIA
    }

    /// Signals the fence on the queue and returns the value.
    pub fn signal(&self) -> Result<u64> {
        let v = self.next_value.get();
        self.next_value.set(v + 1);
        // SAFETY: valid queue and fence.
        unsafe { self.queue.Signal(&self.fence, v)? };
        Ok(v)
    }

    /// Reserves the next fence value without signaling (for external
    /// producers such as NvOF that signal the shared fence themselves).
    pub fn reserve_value(&self) -> u64 {
        let v = self.next_value.get();
        self.next_value.set(v + 1);
        v
    }

    pub fn completed(&self) -> u64 {
        // SAFETY: valid fence.
        unsafe { self.fence.GetCompletedValue() }
    }

    /// Blocks the CPU until the fence reaches `value`.
    pub fn wait(&self, value: u64) -> Result<()> {
        if self.completed() >= value {
            return Ok(());
        }
        // SAFETY: event handle owned by self.
        unsafe {
            self.fence.SetEventOnCompletion(value, self.event)?;
            WaitForSingleObject(self.event, INFINITE);
        }
        Ok(())
    }

    pub fn wait_idle(&self) -> Result<()> {
        let v = self.signal()?;
        self.wait(v)
    }

    /// `S_OK` while the device is healthy.
    pub fn removed_reason(&self) -> windows::core::HRESULT {
        // SAFETY: valid device.
        unsafe { self.device.GetDeviceRemovedReason() }
            .err()
            .map_or(windows::core::HRESULT(0), |e| e.code())
    }

    pub fn create_buffer(
        &self,
        size: u64,
        heap: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource> {
        self.create_buffer_with(size, heap, state, D3D12_RESOURCE_FLAG_NONE)
    }

    pub fn create_buffer_with(
        &self,
        size: u64,
        heap: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> Result<ID3D12Resource> {
        let desc = buffer_desc(size, flags);
        let mut res = None;
        // SAFETY: valid descriptors; out pointer is a local.
        unsafe {
            self.device.CreateCommittedResource(
                &heap_props(heap),
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut res,
            )?
        };
        res.ok_or_else(windows::core::Error::empty)
    }

    pub fn create_texture(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
            Alignment: 0,
        };
        let mut res = None;
        // SAFETY: valid descriptors; out pointer is a local.
        unsafe {
            self.device.CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut res,
            )?
        };
        res.ok_or_else(windows::core::Error::empty)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        // SAFETY: handle created in `new`.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

pub fn heap_props(t: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: t,
        ..Default::default()
    }
}

pub fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        Alignment: 0,
    }
}

/// Borrows a COM pointer into a `ManuallyDrop<Option<T>>` field without
/// touching the reference count. The caller keeps `r` alive for as long as
/// the returned value is used.
pub fn borrow_com<T: Interface>(r: &T) -> ManuallyDrop<Option<T>> {
    // SAFETY: `Option<T>` has the same layout as a nullable interface
    // pointer and ManuallyDrop prevents the matching Release.
    unsafe { std::mem::transmute_copy(r) }
}

pub fn transition(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: borrow_com(r),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

pub fn uav_barrier(r: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: borrow_com(r),
            }),
        },
    }
}

pub fn shader_bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}
