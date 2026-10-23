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

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, HMODULE, LUID};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows::core::{HSTRING, PCSTR};

use crate::cuda::{
    self, CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_PARAMS, CUcontext, CUdevice,
    CUdeviceptr, CUexternalMemory, CUexternalSemaphore, CUstream, Cuda, Win32Handle,
};

/// Core runtime DLLs in load order (dependencies first).
pub const CORE_DLLS: [&str; 4] = [
    "cudart64_12.dll",
    "NVCVImage.dll",
    "nvngxruntime.dll",
    "NVVideoEffects.dll",
];

/// NvCV packed RGB10A2 pixel format / 32-bit packed component type; bit
/// layout identical to `DXGI_FORMAT_R10G10B10A2_UNORM` (R in the low bits).
const NVCV_RGB10A2: i32 = 13;
const NVCV_P32: i32 = 11;
const NVCV_INTERLEAVED: u32 = 0;
const NVCV_CPU: u32 = 0;
const NVCV_CUDA: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum VfxError {
    #[error("Video Effects runtime not found: {0}")]
    MissingRuntime(PathBuf),
    #[error("CUDA driver unavailable")]
    NoCuda,
    #[error("no CUDA device matches the D3D12 adapter")]
    NoMatchingDevice,
    #[error("this effect requires a newer GPU (compute {0}.{1})")]
    Unsupported(i32, i32),
    #[error("{0}")]
    Cuda(String),
    #[error("{call}: NvCV status {status} ({detail})")]
    Vfx {
        call: &'static str,
        status: i32,
        detail: String,
    },
    #[error("D3D12: {0}")]
    D3d(#[from] windows::core::Error),
}

type NvStatus = i32;

struct Api {
    create: unsafe extern "C" fn(*const i8, *mut *mut c_void) -> NvStatus,
    destroy: unsafe extern "C" fn(*mut c_void),
    set_u32: unsafe extern "C" fn(*mut c_void, *const i8, u32) -> NvStatus,
    set_f32: unsafe extern "C" fn(*mut c_void, *const i8, f32) -> NvStatus,
    set_image: unsafe extern "C" fn(*mut c_void, *const i8, *mut c_void) -> NvStatus,
    set_stream: unsafe extern "C" fn(*mut c_void, *const i8, CUstream) -> NvStatus,
    load: unsafe extern "C" fn(*mut c_void) -> NvStatus,
    run: unsafe extern "C" fn(*mut c_void, i32) -> NvStatus,
    image_create:
        unsafe extern "C" fn(u32, u32, i32, i32, u32, u32, u32, *mut *mut c_void) -> NvStatus,
    image_dealloc: unsafe extern "C" fn(*mut c_void),
    image_destroy: unsafe extern "C" fn(*mut c_void),
    image_init: unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        i32,
        *mut c_void,
        i32,
        i32,
        u32,
        u32,
    ) -> NvStatus,
    error_string: unsafe extern "C" fn(NvStatus) -> *const i8,
}

/// A loaded runtime directory. Lives for the rest of the process.
pub struct Runtime {
    dir: PathBuf,
    api: Api,
    effects: Mutex<Vec<String>>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").field("dir", &self.dir).finish()
    }
}

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

impl Runtime {
    /// The runtime of `dir`, loading it on first use.
    pub fn get(dir: &Path) -> Result<&'static Runtime, VfxError> {
        static LOADED: OnceLock<Mutex<HashMap<PathBuf, &'static Runtime>>> = OnceLock::new();
        let mut map = LOADED
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| VfxError::MissingRuntime(dir.to_path_buf()))?;
        if let Some(r) = map.get(dir) {
            return Ok(r);
        }
        let modules = CORE_DLLS
            .iter()
            .map(|n| load_dll(&dir.join(n)))
            .collect::<Result<Vec<_>, _>>()?;
        let (core, image) = (modules[3], modules[1]);
        let missing = || VfxError::MissingRuntime(dir.to_path_buf());
        let api = Api {
            create: sym(core, "NvVFX_CreateEffect").ok_or_else(missing)?,
            destroy: sym(core, "NvVFX_DestroyEffect").ok_or_else(missing)?,
            set_u32: sym(core, "NvVFX_SetU32").ok_or_else(missing)?,
            set_f32: sym(core, "NvVFX_SetF32").ok_or_else(missing)?,
            set_image: sym(core, "NvVFX_SetImage").ok_or_else(missing)?,
            set_stream: sym(core, "NvVFX_SetCudaStream").ok_or_else(missing)?,
            load: sym(core, "NvVFX_Load").ok_or_else(missing)?,
            run: sym(core, "NvVFX_Run").ok_or_else(missing)?,
            image_create: sym(image, "NvCVImage_Create").ok_or_else(missing)?,
            image_dealloc: sym(image, "NvCVImage_Dealloc").ok_or_else(missing)?,
            image_destroy: sym(image, "NvCVImage_Destroy").ok_or_else(missing)?,
            image_init: sym(image, "NvCVImage_Init").ok_or_else(missing)?,
            error_string: sym(image, "NvCV_GetErrorStringFromCode").ok_or_else(missing)?,
        };
        let rt: &'static Runtime = Box::leak(Box::new(Runtime {
            dir: dir.to_path_buf(),
            api,
            effects: Mutex::new(Vec::new()),
        }));
        map.insert(dir.to_path_buf(), rt);
        Ok(rt)
    }

    /// Loads an effect DLL (e.g. `nvVFXVideoSuperRes.dll`) once.
    fn ensure_effect(&self, dll: &str) -> Result<(), VfxError> {
        let mut loaded = self
            .effects
            .lock()
            .map_err(|_| VfxError::MissingRuntime(self.dir.clone()))?;
        if !loaded.iter().any(|d| d == dll) {
            load_dll(&self.dir.join(dll))?;
            loaded.push(dll.to_string());
        }
        Ok(())
    }

    fn check(&self, call: &'static str, status: NvStatus) -> Result<(), VfxError> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: returns a static string for any code.
        let detail = unsafe {
            let p = (self.api.error_string)(status);
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(VfxError::Vfx {
            call,
            status,
            detail,
        })
    }
}

/// Whether the core DLLs and `effect_dlls` exist in `dir`.
pub fn runtime_present(dir: &Path, effect_dlls: &[&str]) -> bool {
    CORE_DLLS
        .iter()
        .chain(effect_dlls)
        .all(|n| dir.join(n).is_file())
}

/// Pushes the CUDA context for the scope.
pub(crate) struct CtxGuard<'a>(&'a Cuda);

impl<'a> CtxGuard<'a> {
    fn push(cuda: &'a Cuda, ctx: CUcontext) -> Result<Self, VfxError> {
        // SAFETY: valid retained primary context.
        cuda::check("cuCtxPushCurrent", unsafe { (cuda.cuCtxPushCurrent)(ctx) })
            .map_err(VfxError::Cuda)?;
        Ok(Self(cuda))
    }
}

impl Drop for CtxGuard<'_> {
    fn drop(&mut self) {
        let mut prev = std::ptr::null_mut();
        // SAFETY: pops the context pushed in `push`.
        unsafe { (self.0.cuCtxPopCurrent)(&mut prev) };
    }
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

/// CUDA side of a session: context, stream and the two shared fences.
pub struct Bridge {
    cuda: &'static Cuda,
    device: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
    producer: ID3D12Fence,
    completion: ID3D12Fence,
    producer_sem: CUexternalSemaphore,
    completion_sem: CUexternalSemaphore,
    pub(crate) runtime: &'static Runtime,
}

impl std::fmt::Debug for Bridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bridge")
            .field("runtime", &self.runtime)
            .finish()
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
        let cuda = cuda::driver().ok_or(VfxError::NoCuda)?;
        if !runtime_present(runtime_dir, &[effect_dll]) {
            return Err(VfxError::MissingRuntime(runtime_dir.to_path_buf()));
        }
        let cu = |call, r| cuda::check(call, r).map_err(VfxError::Cuda);
        let mut count = 0;
        // SAFETY: plain driver queries with valid out pointers.
        unsafe { cu("cuDeviceGetCount", (cuda.cuDeviceGetCount)(&mut count))? };
        let mut found = None;
        for i in 0..count {
            let (mut dev, mut l, mut mask) = (0, [0u8; 8], 0u32);
            // SAFETY: as above.
            unsafe {
                cu("cuDeviceGet", (cuda.cuDeviceGet)(&mut dev, i))?;
                cu(
                    "cuDeviceGetLuid",
                    (cuda.cuDeviceGetLuid)(l.as_mut_ptr(), &mut mask, dev),
                )?;
            }
            let low = u32::from_le_bytes([l[0], l[1], l[2], l[3]]);
            let high = i32::from_le_bytes([l[4], l[5], l[6], l[7]]);
            if low == luid.LowPart && high == luid.HighPart {
                found = Some(dev);
                break;
            }
        }
        let dev = found.ok_or(VfxError::NoMatchingDevice)?;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: as above.
        unsafe {
            cu(
                "compute major",
                (cuda.cuDeviceGetAttribute)(
                    &mut major,
                    cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                ),
            )?;
            cu(
                "compute minor",
                (cuda.cuDeviceGetAttribute)(
                    &mut minor,
                    cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                ),
            )?;
        }
        if (major, minor) < min_compute {
            return Err(VfxError::Unsupported(major, minor));
        }

        let runtime = Runtime::get(runtime_dir)?;
        runtime.ensure_effect(effect_dll)?;
        let mut ctx = std::ptr::null_mut();
        // SAFETY: retain the primary context (released in Drop).
        unsafe {
            cu(
                "cuDevicePrimaryCtxRetain",
                (cuda.cuDevicePrimaryCtxRetain)(&mut ctx, dev),
            )?
        };
        let guard = CtxGuard::push(cuda, ctx)?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: context is current.
        unsafe {
            cu(
                "cuStreamCreate",
                (cuda.cuStreamCreate)(&mut stream, cuda::CU_STREAM_NON_BLOCKING),
            )?
        };
        let make_fence = || -> Result<(ID3D12Fence, CUexternalSemaphore), VfxError> {
            // SAFETY: shared fence imported as a CUDA semaphore; handle closed after import.
            unsafe {
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_SHARED)?;
                let handle = shared_handle(device, &fence)?;
                let mut sem = std::ptr::null_mut();
                let desc = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
                    type_: cuda::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
                    handle: Win32Handle {
                        handle: handle.0,
                        name: std::ptr::null(),
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
        drop(guard);
        tracing::debug!(
            effect_dll,
            compute = format!("{major}.{minor}"),
            "VFX bridge ready"
        );
        Ok(Self {
            cuda,
            device: dev,
            ctx,
            stream,
            producer,
            completion,
            producer_sem,
            completion_sem,
            runtime,
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

    pub(crate) fn push(&self) -> Result<CtxGuard<'static>, VfxError> {
        CtxGuard::push(self.cuda, self.ctx)
    }

    /// Enqueues a GPU wait for `producer >= value` on the stream.
    pub(crate) fn wait(&self, value: u64) -> Result<(), VfxError> {
        let p = CUDA_EXTERNAL_SEMAPHORE_PARAMS::fence(value);
        // SAFETY: semaphore and stream owned by this bridge; context pushed by caller.
        let r = unsafe {
            (self.cuda.cuWaitExternalSemaphoresAsync)(&self.producer_sem, &p, 1, self.stream)
        };
        cuda::check("cuWaitExternalSemaphoresAsync", r).map_err(VfxError::Cuda)
    }

    /// Enqueues `completion = value` on the stream.
    pub(crate) fn signal(&self, value: u64) -> Result<(), VfxError> {
        let p = CUDA_EXTERNAL_SEMAPHORE_PARAMS::fence(value);
        // SAFETY: as above.
        let r = unsafe {
            (self.cuda.cuSignalExternalSemaphoresAsync)(&self.completion_sem, &p, 1, self.stream)
        };
        cuda::check("cuSignalExternalSemaphoresAsync", r).map_err(VfxError::Cuda)
    }

    /// Blocks until the stream is idle (teardown and tests only).
    pub fn synchronize(&self) {
        if let Ok(_g) = self.push() {
            // SAFETY: valid stream; context current.
            unsafe { (self.cuda.cuStreamSynchronize)(self.stream) };
        }
    }

    /// Blocks until *all* work in the CUDA context is idle, including
    /// streams the SDK may use internally (benchmarks only).
    pub fn synchronize_context(&self) {
        if let Ok(_g) = self.push() {
            // SAFETY: context current.
            unsafe { (self.cuda.cuCtxSynchronize)() };
        }
    }

    /// Creates an effect bound to this bridge's stream.
    pub(crate) fn effect(&self, name: &str) -> Result<Effect, VfxError> {
        let _g = self.push()?;
        let mut handle = std::ptr::null_mut();
        let api = &self.runtime.api;
        let n = CString::new(name).unwrap_or_default();
        let s = CString::new("CudaStream").unwrap_or_default();
        // SAFETY: documented VFX calls; strings outlive the calls.
        unsafe {
            self.runtime
                .check("NvVFX_CreateEffect", (api.create)(n.as_ptr(), &mut handle))?;
            let e = Effect {
                handle,
                runtime: self.runtime,
                cuda: self.cuda,
                ctx: self.ctx,
            };
            self.runtime.check(
                "CudaStream",
                (api.set_stream)(handle, s.as_ptr(), self.stream),
            )?;
            Ok(e)
        }
    }

    /// A packed RGB10A2 image of `width × height` shared between D3D12 and CUDA.
    pub fn image(
        &self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<SharedImage, VfxError> {
        let _g = self.push()?;
        let cu = |call, r| cuda::check(call, r).map_err(VfxError::Cuda);
        let cuda = self.cuda;
        let runtime = self.runtime;
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
                handle: Win32Handle {
                    handle: handle.0,
                    name: std::ptr::null(),
                },
                size: alloc.SizeInBytes,
                flags: cuda::CUDA_EXTERNAL_MEMORY_DEDICATED,
                reserved: [0; 16],
            };
            let r = (cuda.cuImportExternalMemory)(&mut memory, &import);
            let _ = CloseHandle(handle);
            cu("cuImportExternalMemory", r)?;
            let mut pointer = 0;
            let map = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
                offset: 0,
                size: bytes,
                flags: 0,
                reserved: [0; 16],
            };
            cu(
                "cuExternalMemoryGetMappedBuffer",
                (cuda.cuExternalMemoryGetMappedBuffer)(&mut pointer, memory, &map),
            )?;
            // Let NvCV allocate its descriptor, then point it at our memory.
            let mut image = std::ptr::null_mut();
            runtime.check(
                "NvCVImage_Create",
                (runtime.api.image_create)(
                    1,
                    1,
                    NVCV_RGB10A2,
                    NVCV_P32,
                    NVCV_INTERLEAVED,
                    NVCV_CPU,
                    1,
                    &mut image,
                ),
            )?;
            (runtime.api.image_dealloc)(image);
            let img = SharedImage {
                resource,
                memory,
                pointer,
                image,
                width,
                height,
                pitch,
                cuda,
                ctx: self.ctx,
                runtime,
            };
            runtime.check(
                "NvCVImage_Init",
                (runtime.api.image_init)(
                    image,
                    width,
                    height,
                    pitch as i32,
                    pointer as *mut c_void,
                    NVCV_RGB10A2,
                    NVCV_P32,
                    NVCV_INTERLEAVED,
                    NVCV_CUDA,
                ),
            )?;
            Ok(img)
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let cuda = self.cuda;
        if let Ok(_g) = CtxGuard::push(cuda, self.ctx) {
            // SAFETY: drain the stream, then destroy what this bridge created.
            unsafe {
                (cuda.cuStreamSynchronize)(self.stream);
                (cuda.cuDestroyExternalSemaphore)(self.producer_sem);
                (cuda.cuDestroyExternalSemaphore)(self.completion_sem);
                (cuda.cuStreamDestroy)(self.stream);
            }
        }
        // SAFETY: balances cuDevicePrimaryCtxRetain.
        unsafe { (cuda.cuDevicePrimaryCtxRelease)(self.device) };
    }
}

/// A D3D12 buffer shared with CUDA, wrapped as an NvCV RGB10A2 image.
pub struct SharedImage {
    resource: ID3D12Resource,
    memory: CUexternalMemory,
    pointer: CUdeviceptr,
    image: *mut c_void,
    pub width: u32,
    pub height: u32,
    /// Bytes per row.
    pub pitch: u32,
    cuda: &'static Cuda,
    ctx: CUcontext,
    runtime: &'static Runtime,
}

impl std::fmt::Debug for SharedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedImage({}x{})", self.width, self.height)
    }
}

impl SharedImage {
    /// The D3D12 side (packed RGB10A2 rows of `pitch` bytes).
    pub fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    pub(crate) fn nvcv(&self) -> *mut c_void {
        self.image
    }
}

impl Drop for SharedImage {
    fn drop(&mut self) {
        if let Ok(_g) = CtxGuard::push(self.cuda, self.ctx) {
            // SAFETY: created in `Bridge::image`; the owning session drained
            // the stream before dropping images.
            unsafe {
                (self.runtime.api.image_destroy)(self.image);
                (self.cuda.cuMemFree)(self.pointer);
                (self.cuda.cuDestroyExternalMemory)(self.memory);
            }
        }
    }
}

/// An effect instance (`NvVFX_Handle`).
pub(crate) struct Effect {
    handle: *mut c_void,
    runtime: &'static Runtime,
    cuda: &'static Cuda,
    ctx: CUcontext,
}

impl Effect {
    fn name(s: &str) -> CString {
        CString::new(s).unwrap_or_default()
    }

    pub(crate) fn set_u32(&self, key: &'static str, v: u32) -> Result<(), VfxError> {
        let k = Self::name(key);
        // SAFETY: valid handle; key outlives the call.
        self.runtime.check(key, unsafe {
            (self.runtime.api.set_u32)(self.handle, k.as_ptr(), v)
        })
    }

    pub(crate) fn set_f32(&self, key: &'static str, v: f32) -> Result<(), VfxError> {
        let k = Self::name(key);
        // SAFETY: as above.
        self.runtime.check(key, unsafe {
            (self.runtime.api.set_f32)(self.handle, k.as_ptr(), v)
        })
    }

    pub(crate) fn set_image(&self, key: &'static str, img: &SharedImage) -> Result<(), VfxError> {
        let k = Self::name(key);
        // SAFETY: as above; the image outlives its binding (owned by the session).
        self.runtime.check(key, unsafe {
            (self.runtime.api.set_image)(self.handle, k.as_ptr(), img.nvcv())
        })
    }

    pub(crate) fn load(&self) -> Result<(), VfxError> {
        // SAFETY: valid handle.
        self.runtime.check("NvVFX_Load", unsafe {
            (self.runtime.api.load)(self.handle)
        })
    }

    /// Asynchronous run on the bridge's stream.
    pub(crate) fn run(&self) -> Result<(), VfxError> {
        // SAFETY: valid handle.
        self.runtime.check("NvVFX_Run", unsafe {
            (self.runtime.api.run)(self.handle, 1)
        })
    }
}

impl Drop for Effect {
    fn drop(&mut self) {
        let _g = CtxGuard::push(self.cuda, self.ctx);
        // SAFETY: created by NvVFX_CreateEffect; the session drained the stream.
        unsafe { (self.runtime.api.destroy)(self.handle) };
    }
}
