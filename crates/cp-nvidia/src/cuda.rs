//! Minimal CUDA driver API (`nvcuda.dll` / `libcuda.so.1`, ships with the
//! display driver), loaded dynamically. Only what graphics ↔ CUDA interop
//! needs: D3D12 resources and fences on Windows, Vulkan opaque-fd memory and
//! timeline semaphores on Linux.

#![allow(non_camel_case_types, non_snake_case, missing_debug_implementations)]

use std::ffi::c_void;
use std::sync::OnceLock;

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUexternalMemory = *mut c_void;
pub type CUexternalSemaphore = *mut c_void;
pub type CUdeviceptr = u64;

pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: i32 = 1;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: i32 = 5;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE: i32 = 4;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD: i32 = 9;
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

/// The `handle` union of the external memory/semaphore descriptors.
#[repr(C)]
#[derive(Clone, Copy)]
pub union ExternalHandle {
    pub win32: Win32Handle,
    pub fd: i32,
}

#[repr(C)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    pub type_: i32,
    pub handle: ExternalHandle,
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
    pub handle: ExternalHandle,
    pub flags: u32,
    pub reserved: [u32; 16],
}

/// Layout shared by the signal and wait parameter structs (both 144 bytes;
/// only the fence value is used, for D3D12 fences and timeline semaphores).
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

/// The loaded driver library (never unloaded).
#[cfg(windows)]
type Module = windows::Win32::Foundation::HMODULE;
#[cfg(not(windows))]
type Module = *mut c_void;

/// Looks up `name` (NUL-terminated) in `module`.
///
/// # Safety
/// `module` must be a loaded library.
unsafe fn symbol(module: Module, name: &'static str) -> Option<*const c_void> {
    #[cfg(windows)]
    {
        use windows::Win32::System::LibraryLoader::GetProcAddress;
        // SAFETY: valid module and NUL-terminated name.
        unsafe { GetProcAddress(module, windows::core::PCSTR(name.as_ptr())) }
            .map(|f| f as *const c_void)
    }
    #[cfg(not(windows))]
    {
        // SAFETY: valid handle and NUL-terminated name.
        let p = unsafe { libc::dlsym(module, name.as_ptr().cast()) };
        (!p.is_null()).then_some(p.cast_const())
    }
}

macro_rules! cuda_api {
    ($($(#[$m:meta])* $field:ident = $export:literal : fn($($arg:ty),*);)*) => {
        /// Driver entry points.
        pub struct Cuda {
            _module: Module,
            $($(#[$m])* pub $field: unsafe extern "system" fn($($arg),*) -> CUresult,)*
        }

        impl Cuda {
            /// # Safety
            /// `module` must be the CUDA driver library.
            unsafe fn bind(module: Module) -> Option<Self> {
                // SAFETY: exports have the declared driver-API signatures.
                unsafe {
                    Some(Self {
                        _module: module,
                        $($(#[$m])* $field: std::mem::transmute::<
                            *const c_void,
                            unsafe extern "system" fn($($arg),*) -> CUresult,
                        >(symbol(module, concat!($export, "\0"))?),)*
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
    #[cfg(windows)]
    cuDeviceGetLuid = "cuDeviceGetLuid": fn(*mut u8, *mut u32, CUdevice);
    cuDeviceGetUuid = "cuDeviceGetUuid_v2": fn(*mut u8, CUdevice);
    cuDeviceGetAttribute = "cuDeviceGetAttribute": fn(*mut i32, i32, CUdevice);
    cuDevicePrimaryCtxRetain = "cuDevicePrimaryCtxRetain": fn(*mut CUcontext, CUdevice);
    cuDevicePrimaryCtxRelease = "cuDevicePrimaryCtxRelease_v2": fn(CUdevice);
    cuCtxPushCurrent = "cuCtxPushCurrent_v2": fn(CUcontext);
    cuCtxPopCurrent = "cuCtxPopCurrent_v2": fn(*mut CUcontext);
    cuStreamCreate = "cuStreamCreate": fn(*mut CUstream, u32);
    cuStreamDestroy = "cuStreamDestroy_v2": fn(CUstream);
    cuStreamSynchronize = "cuStreamSynchronize": fn(CUstream);
    cuCtxSynchronize = "cuCtxSynchronize": fn();
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

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Cuda {}
// SAFETY: as above.
unsafe impl Sync for Cuda {}

/// Loads and initializes the driver API once per process.
pub fn driver() -> Option<&'static Cuda> {
    static CUDA: OnceLock<Option<Cuda>> = OnceLock::new();
    CUDA.get_or_init(|| {
        // SAFETY: loading the system CUDA driver library.
        unsafe {
            let cuda = Cuda::bind(load_driver()?)?;
            ((cuda.cuInit)(0) == 0).then_some(cuda)
        }
    })
    .as_ref()
}

/// `nvcuda.dll` from System32 only.
#[cfg(windows)]
unsafe fn load_driver() -> Option<Module> {
    use windows::Win32::System::LibraryLoader::{LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW};
    // SAFETY: plain library load.
    unsafe {
        LoadLibraryExW(
            windows::core::w!("nvcuda.dll"),
            None,
            LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    }
    .ok()
}

/// `libcuda.so.1` from the driver (dynamic linker search path).
#[cfg(not(windows))]
unsafe fn load_driver() -> Option<Module> {
    // SAFETY: plain library load.
    let h = unsafe { libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
    (!h.is_null()).then_some(h)
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
