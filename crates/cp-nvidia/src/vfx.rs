//! NVIDIA Video Effects SDK plumbing shared by all effects (VFG, VSR, …):
//! runtime loading, the CUDA context/stream, D3D12↔CUDA shared buffers and
//! shared fences.
//!
//! Frames live in D3D12 buffers created with `D3D12_HEAP_FLAG_SHARED` and
//! imported into CUDA (`cuImportExternalMemory`); synchronization uses two
//! shared D3D12 fences imported as CUDA external semaphores:
//!
//! ```text
//! D3D12 queue: write inputs → Signal(producer, n) ─┐
//! CUDA stream:  Wait(producer, n) → NvVFX_Run → Signal(completion, n)
//! D3D12 queue:  Wait(completion, n) → read output ← ┘
//! ```
//!
//! The runtime DLLs come from NVIDIA's Video Effects SDK (`nvidia-vfx`
//! package) and are loaded from a user-configured directory; they are not
//! part of this repository. They are loaded once per process and never
//! unloaded: effect DLLs run internal worker threads.

use std::ffi::CString;
use std::path::Path;

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, HMODULE, LUID};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows::core::{HSTRING, PCSTR};

use crate::cuda::{
    self, CUDA_EXTERNAL_MEMORY_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    CUexternalSemaphore, Cuda, ExternalHandle, Win32Handle,
};
pub(crate) use crate::vfx_common::Effect;
use crate::vfx_common::{Api, Core, cu};
pub use crate::vfx_common::{Runtime, SharedImage, VfxError};

/// Core runtime DLLs in load order (dependencies first).
pub const CORE_DLLS: [&str; 4] = [
    "cudart64_12.dll",
    "NVCVImage.dll",
    "nvngxruntime.dll",
    "NVVideoEffects.dll",
];

fn sym<T>(module: HMODULE, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    // SAFETY: exported symbol of the expected signature; T is a fn pointer.
    unsafe {
        GetProcAddress(module, PCSTR(cname.as_ptr().cast())).map(|p| std::mem::transmute_copy(&p))
    }
}

fn load_dll(path: &Path) -> Result<HMODULE, VfxError> {
    // SAFETY: loads a DLL from the configured runtime directory; its own
    // dependencies resolve from the same directory or System32.
    unsafe {
        LoadLibraryExW(
            &HSTRING::from(path.as_os_str()),
            None,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    }
    .map_err(|_| VfxError::MissingRuntime(path.to_path_buf()))
}

/// Loads the core DLLs of `dir` and resolves their entry points.
pub(crate) fn load_runtime_libraries(dir: &Path) -> Result<Api, VfxError> {
    let modules = CORE_DLLS
        .iter()
        .map(|n| load_dll(&dir.join(n)))
        .collect::<Result<Vec<_>, _>>()?;
    let (core, image) = (modules[3], modules[1]);
    // SAFETY: the DLLs are the Video Effects SDK's.
    unsafe { Api::resolve(|n| sym(core, n), |n| sym(image, n)) }
        .ok_or_else(|| VfxError::MissingRuntime(dir.to_path_buf()))
}

/// Loads an effect DLL (never unloaded).
pub(crate) fn load_effect_library(path: &Path) -> Result<(), VfxError> {
    load_dll(path).map(|_| ())
}

/// What a [`SharedImage`] keeps alive besides the CUDA import: the shared
/// D3D12 buffer.
pub(crate) type ImageOwner = ID3D12Resource;

impl SharedImage {
    /// The D3D12 side (packed RGB10A2 rows of `pitch` bytes).
    pub fn resource(&self) -> &ID3D12Resource {
        &self.owner
    }
}

/// Whether the core DLLs and `effect_dlls` exist in `dir`.
pub fn runtime_present(dir: &Path, effect_dlls: &[&str]) -> bool {
    CORE_DLLS
        .iter()
        .chain(effect_dlls)
        .all(|n| dir.join(n).is_file())
}

fn shared_handle(
    device: &ID3D12Device,
    object: &impl windows::core::Interface,
) -> windows::core::Result<HANDLE> {
    // SAFETY: valid shareable object.
    unsafe {
        device.CreateSharedHandle(
            &object.cast::<ID3D12DeviceChild>()?,
            None,
            GENERIC_ALL.0,
            None,
        )
    }
}

/// CUDA side of a session: context, stream and the two shared fences (see
/// [`Core`], which it dereferences to).
#[derive(Debug)]
pub struct Bridge {
    core: Core,
    producer: ID3D12Fence,
    completion: ID3D12Fence,
}

impl std::ops::Deref for Bridge {
    type Target = Core;
    fn deref(&self) -> &Core {
        &self.core
    }
}

impl Bridge {
    /// Matches the CUDA device to `luid`, checks `min_compute`, loads the
    /// runtime + `effect_dll` and creates the stream and shared fences.
    pub fn new(
        device: &ID3D12Device,
        luid: LUID,
        runtime_dir: &Path,
        effect_dll: &str,
        min_compute: (i32, i32),
    ) -> Result<Self, VfxError> {
        let matches = |cuda: &Cuda, dev| {
            let (mut l, mut mask) = ([0u8; 8], 0u32);
            // SAFETY: plain driver query with valid out pointers.
            unsafe {
                cu(
                    "cuDeviceGetLuid",
                    (cuda.cuDeviceGetLuid)(l.as_mut_ptr(), &mut mask, dev),
                )?;
            }
            let low = u32::from_le_bytes([l[0], l[1], l[2], l[3]]);
            let high = i32::from_le_bytes([l[4], l[5], l[6], l[7]]);
            Ok(low == luid.LowPart && high == luid.HighPart)
        };
        let mut fences = None;
        let import = |cuda: &'static Cuda| {
            let make_fence = || -> Result<(ID3D12Fence, CUexternalSemaphore), VfxError> {
                // SAFETY: shared fence imported as a CUDA semaphore; handle closed after import.
                unsafe {
                    let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_SHARED)?;
                    let handle = shared_handle(device, &fence)?;
                    let mut sem = std::ptr::null_mut();
                    let desc = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
                        type_: cuda::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
                        handle: ExternalHandle {
                            win32: Win32Handle {
                                handle: handle.0,
                                name: std::ptr::null(),
                            },
                        },
                        flags: 0,
                        reserved: [0; 16],
                    };
                    let r = (cuda.cuImportExternalSemaphore)(&mut sem, &desc);
                    let _ = CloseHandle(handle);
                    cu("cuImportExternalSemaphore", r)?;
                    Ok((fence, sem))
                }
            };
            let (producer, producer_sem) = make_fence()?;
            let (completion, completion_sem) = make_fence()?;
            fences = Some((producer, completion));
            Ok((producer_sem, completion_sem))
        };
        let core = Core::new(runtime_dir, effect_dll, min_compute, matches, import)?;
        let (producer, completion) = fences.ok_or(VfxError::NoCuda)?;
        Ok(Self {
            core,
            producer,
            completion,
        })
    }

    /// Fence the D3D12 queue signals when inputs are ready.
    pub fn producer(&self) -> &ID3D12Fence {
        &self.producer
    }

    /// Fence CUDA signals when outputs are written.
    pub fn completion(&self) -> &ID3D12Fence {
        &self.completion
    }

    /// A packed RGB10A2 image of `width × height` shared between D3D12 and CUDA.
    pub fn image(
        &self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<SharedImage, VfxError> {
        let _g = self.push()?;
        let cuda = self.cuda;
        let pitch = (width * 4).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            Alignment: 0,
        };
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: shared committed buffer; handle closed after import.
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut res,
            )?;
            let resource = res.ok_or_else(windows::core::Error::empty)?;
            let handle = shared_handle(device, &resource)?;
            let alloc = device.GetResourceAllocationInfo(0, &[desc]);
            let mut memory = std::ptr::null_mut();
            let import = CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
                type_: cuda::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
                handle: ExternalHandle {
                    win32: Win32Handle {
                        handle: handle.0,
                        name: std::ptr::null(),
                    },
                },
                size: alloc.SizeInBytes,
                flags: cuda::CUDA_EXTERNAL_MEMORY_DEDICATED,
                reserved: [0; 16],
            };
            let r = (cuda.cuImportExternalMemory)(&mut memory, &import);
            let _ = CloseHandle(handle);
            cu("cuImportExternalMemory", r)?;
            self.wrap(memory, (width, height, pitch), resource)
        }
    }
}
