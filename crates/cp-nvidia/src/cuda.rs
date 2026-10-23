//! Minimal CUDA driver API (`nvcuda.dll`, ships with the display driver),
//! loaded dynamically. Only what D3D12 ↔ CUDA interop needs.

#![allow(non_camel_case_types, non_snake_case, missing_debug_implementations)]

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows::core::{PCSTR, w};

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUexternalMemory = *mut c_void;
pub type CUexternalSemaphore = *mut c_void;
pub type CUdeviceptr = u64;

pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: i32 = 5;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE: i32 = 4;
pub const CUDA_EXTERNAL_MEMORY_DEDICATED: u32 = 1;
pub const CU_STREAM_NON_BLOCKING: u32 = 1;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Win32Handle {
    pub handle: *mut c_void,
    pub name: *const c_void,
}

#[repr(C)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    pub type_: i32,
    pub handle: Win32Handle,
    pub size: u64,
    pub flags: u32,
    pub reserved: [u32; 16],
}

#[repr(C)]
pub struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    pub offset: u64,
    pub size: u64,
    pub flags: u32,
    pub reserved: [u32; 16],
}

#[repr(C)]
pub struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    pub type_: i32,
    pub handle: Win32Handle,
    pub flags: u32,
    pub reserved: [u32; 16],
}

/// Layout shared by the signal and wait parameter structs (both 144 bytes;
/// only the fence value is used for D3D12 fences).
#[repr(C)]
pub struct CUDA_EXTERNAL_SEMAPHORE_PARAMS {
    pub fence_value: u64,
    pub nv_sci_sync: u64,
    pub keyed_mutex: [u64; 2],
    pub reserved: [u32; 10],
    pub flags: u32,
    pub reserved2: [u32; 16],
}

impl CUDA_EXTERNAL_SEMAPHORE_PARAMS {
    pub fn fence(value: u64) -> Self {
        Self {
            fence_value: value,
            nv_sci_sync: 0,
            keyed_mutex: [0; 2],
            reserved: [0; 10],
            flags: 0,
            reserved2: [0; 16],
        }
    }
}

macro_rules! cuda_api {
    ($($field:ident = $export:literal : fn($($arg:ty),*);)*) => {
        /// Driver entry points.
        pub struct Cuda {
            _module: HMODULE,
            $(pub $field: unsafe extern "system" fn($($arg),*) -> CUresult,)*
        }

        impl Cuda {
            /// # Safety
            /// `module` must be `nvcuda.dll`.
            unsafe fn bind(module: HMODULE) -> Option<Self> {
                // SAFETY: exports have the declared driver-API signatures.
                unsafe {
                    Some(Self {
                        _module: module,
                        $($field: std::mem::transmute::<
                            unsafe extern "system" fn() -> isize,
                            unsafe extern "system" fn($($arg),*) -> CUresult,
                        >(GetProcAddress(module, PCSTR(concat!($export, "\0").as_ptr()))?),)*
                    })
                }
            }
        }
    };
}

cuda_api! {
    cuInit = "cuInit": fn(u32);
    cuDeviceGetCount = "cuDeviceGetCount": fn(*mut i32);
    cuDeviceGet = "cuDeviceGet": fn(*mut CUdevice, i32);
    cuDeviceGetLuid = "cuDeviceGetLuid": fn(*mut u8, *mut u32, CUdevice);
    cuDeviceGetAttribute = "cuDeviceGetAttribute": fn(*mut i32, i32, CUdevice);
    cuDevicePrimaryCtxRetain = "cuDevicePrimaryCtxRetain": fn(*mut CUcontext, CUdevice);
    cuDevicePrimaryCtxRelease = "cuDevicePrimaryCtxRelease_v2": fn(CUdevice);
    cuCtxPushCurrent = "cuCtxPushCurrent_v2": fn(CUcontext);
    cuCtxPopCurrent = "cuCtxPopCurrent_v2": fn(*mut CUcontext);
    cuStreamCreate = "cuStreamCreate": fn(*mut CUstream, u32);
    cuStreamDestroy = "cuStreamDestroy_v2": fn(CUstream);
    cuStreamSynchronize = "cuStreamSynchronize": fn(CUstream);
    cuImportExternalMemory = "cuImportExternalMemory": fn(*mut CUexternalMemory, *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC);
    cuExternalMemoryGetMappedBuffer = "cuExternalMemoryGetMappedBuffer": fn(*mut CUdeviceptr, CUexternalMemory, *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC);
    cuDestroyExternalMemory = "cuDestroyExternalMemory": fn(CUexternalMemory);
    cuMemFree = "cuMemFree_v2": fn(CUdeviceptr);
    cuImportExternalSemaphore = "cuImportExternalSemaphore": fn(*mut CUexternalSemaphore, *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC);
    cuDestroyExternalSemaphore = "cuDestroyExternalSemaphore": fn(CUexternalSemaphore);
    cuWaitExternalSemaphoresAsync = "cuWaitExternalSemaphoresAsync": fn(*const CUexternalSemaphore, *const CUDA_EXTERNAL_SEMAPHORE_PARAMS, u32, CUstream);
    cuSignalExternalSemaphoresAsync = "cuSignalExternalSemaphoresAsync": fn(*const CUexternalSemaphore, *const CUDA_EXTERNAL_SEMAPHORE_PARAMS, u32, CUstream);
}

impl std::fmt::Debug for Cuda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Cuda")
    }
}

// SAFETY: function pointers into a DLL that stays loaded for the process.
unsafe impl Send for Cuda {}
// SAFETY: as above.
unsafe impl Sync for Cuda {}

/// Loads and initializes the driver API once per process.
pub fn driver() -> Option<&'static Cuda> {
    static CUDA: OnceLock<Option<Cuda>> = OnceLock::new();
    CUDA.get_or_init(|| {
        // SAFETY: loading the system CUDA driver DLL from System32 only.
        unsafe {
            let module =
                LoadLibraryExW(w!("nvcuda.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
            let cuda = Cuda::bind(module)?;
            ((cuda.cuInit)(0) == 0).then_some(cuda)
        }
    })
    .as_ref()
}

/// Turns a `CUresult` into a `Result`.
pub fn check(call: &'static str, r: CUresult) -> Result<(), String> {
    if r == 0 {
        Ok(())
    } else {
        Err(format!("{call} failed with CUresult {r}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout checks against `cuda.h` (CUDA 13.3, x64).
    #[test]
    fn struct_layouts_match_cuda_h() {
        assert_eq!(size_of::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>(), 104);
        assert_eq!(
            std::mem::offset_of!(CUDA_EXTERNAL_MEMORY_HANDLE_DESC, size),
            24
        );
        assert_eq!(size_of::<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>(), 88);
        assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>(), 96);
        assert_eq!(
            std::mem::offset_of!(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, flags),
            24
        );
        assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_PARAMS>(), 144);
        assert_eq!(
            std::mem::offset_of!(CUDA_EXTERNAL_SEMAPHORE_PARAMS, flags),
            72
        );
    }
}
