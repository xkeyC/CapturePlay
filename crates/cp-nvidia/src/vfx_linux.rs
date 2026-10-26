//! NVIDIA Video Effects SDK plumbing on Linux: runtime loading, the CUDA
//! context/stream, and Vulkan memory/semaphores imported into CUDA.
//!
//! The same model as the Windows module (`vfx.rs`), with the graphics side
//! owned by the caller: frames live in Vulkan buffers exported as opaque
//! fds (`cuImportExternalMemory`), synchronization uses two exported
//! Vulkan timeline semaphores:
//!
//! ```text
//! Vulkan queue: write inputs → signal(producer, n) ─┐
//! CUDA stream:  wait(producer, n) → NvVFX_Run → signal(completion, n)
//! Vulkan queue: wait(completion, n) → read output ← ┘
//! ```
//!
//! The runtime libraries come from NVIDIA's Video Effects SDK (`nvidia-vfx`
//! wheel: `nvvfx/libs` + `nvidia_vfx.libs`) and are loaded once per process
//! from a user-configured directory; they are not part of this repository.

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::os::fd::{IntoRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::cuda::{
    self, CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_PARAMS, CUcontext, CUdevice,
    CUdeviceptr, CUexternalMemory, CUexternalSemaphore, CUstream, Cuda, ExternalHandle,
};

/// Libraries preloaded before the core, in dependency order (prefixes; the
/// wheel's RUNPATH only covers `nvidia_vfx.libs`, so NPP and the core
/// libraries are resolved from what is already loaded).
const PRELOAD: [&str; 11] = [
    "libcudart",
    "libnppc.so",
    "libnppial.so",
    "libnppicc.so",
    "libnppidei.so",
    "libnppif.so",
    "libnppig.so",
    "libnppim.so",
    "libnppist.so",
    "libnppitc.so",
    "libcudnn.so",
];

/// Core libraries in load order.
pub const CORE_LIBS: [&str; 4] = [
    "libNVCVImage.so",
    "libnvngxruntime.so",
    "libVideoFXLocal.so",
    "libVideoFX.so",
];

/// NvCV packed RGB10A2 pixel format / 32-bit packed component type (R in
/// the low bits, like Vulkan `A2B10G10R10_UNORM_PACK32`).
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
    #[error("no CUDA device matches the Vulkan device")]
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

type Library = *mut c_void;

fn sym<T>(lib: Library, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    // SAFETY: valid handle and name.
    let p = unsafe { libc::dlsym(lib, cname.as_ptr()) };
    // SAFETY: exported symbol of the expected signature; T is a fn pointer.
    (!p.is_null()).then(|| unsafe { std::mem::transmute_copy(&p) })
}

fn load_lib(path: &Path) -> Result<Library, VfxError> {
    let c = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| VfxError::MissingRuntime(path.to_path_buf()))?;
    // SAFETY: loads a library from the configured runtime directory; global
    // so that later libraries resolve their NEEDED entries against it.
    let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
    if h.is_null() {
        // SAFETY: dlerror returns a thread-local message or null.
        let msg = unsafe { libc::dlerror() };
        let detail = if msg.is_null() {
            String::new()
        } else {
            // SAFETY: NUL-terminated message valid until the next dl call.
            unsafe { std::ffi::CStr::from_ptr(msg) }
                .to_string_lossy()
                .into_owned()
        };
        tracing::warn!(path = %path.display(), detail, "cannot load VFX library");
        return Err(VfxError::MissingRuntime(path.to_path_buf()));
    }
    Ok(h)
}

/// Folders a runtime directory's libraries may be in: the folder itself and
/// the wheel's `nvidia_vfx.libs` (two levels up from `nvvfx/libs`).
fn lib_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut d = vec![dir.to_path_buf()];
    if let Some(root) = dir.parent().and_then(Path::parent) {
        d.push(root.join("nvidia_vfx.libs"));
    }
    d
}

/// The first library whose file name starts with `prefix`.
fn find_lib(dir: &Path, prefix: &str) -> Option<PathBuf> {
    lib_dirs(dir).into_iter().find_map(|d| {
        let mut names: Vec<PathBuf> = std::fs::read_dir(&d)
            .ok()?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
            })
            .collect();
        names.sort();
        names.into_iter().next()
    })
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
        for p in PRELOAD.iter().filter_map(|n| find_lib(dir, n)) {
            load_lib(&p)?;
        }
        let modules = CORE_LIBS
            .iter()
            .map(|n| load_lib(&dir.join(n)))
            .collect::<Result<Vec<_>, _>>()?;
        let (core, image) = (modules[3], modules[0]);
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

    /// Loads an effect library (e.g. `libnvVFXVideoFrameGeneration.so`) once.
    fn ensure_effect(&self, lib: &str) -> Result<(), VfxError> {
        let mut loaded = self
            .effects
            .lock()
            .map_err(|_| VfxError::MissingRuntime(self.dir.clone()))?;
        if !loaded.iter().any(|d| d == lib) {
            load_lib(&self.dir.join(lib))?;
            loaded.push(lib.to_string());
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

/// Whether the core libraries and `effect_libs` exist in `dir`.
pub fn runtime_present(dir: &Path, effect_libs: &[&str]) -> bool {
    CORE_LIBS
        .iter()
        .chain(effect_libs)
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

/// Imports an exported Vulkan timeline semaphore; CUDA owns `fd` after a
/// successful import.
fn import_semaphore(cuda: &Cuda, fd: OwnedFd) -> Result<CUexternalSemaphore, VfxError> {
    let raw = fd.into_raw_fd();
    let desc = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
        type_: cuda::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
        handle: ExternalHandle { fd: raw },
        flags: 0,
        reserved: [0; 16],
    };
    let mut sem = std::ptr::null_mut();
    // SAFETY: valid descriptor; context pushed by the caller.
    let r = unsafe { (cuda.cuImportExternalSemaphore)(&mut sem, &desc) };
    if r != 0 {
        // SAFETY: still ours after a failed import.
        unsafe { libc::close(raw) };
    }
    cuda::check("cuImportExternalSemaphore", r).map_err(VfxError::Cuda)?;
    Ok(sem)
}

/// CUDA side of a session: context, stream and the two imported timeline
/// semaphores.
pub struct Bridge {
    cuda: &'static Cuda,
    device: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
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
    /// Matches the CUDA device to the Vulkan device `uuid`, checks
    /// `min_compute`, loads the runtime + `effect_lib`, creates the stream
    /// and imports the `producer` (signaled by Vulkan) and `completion`
    /// (signaled by CUDA) timeline semaphores.
    pub fn new(
        uuid: [u8; 16],
        runtime_dir: &Path,
        effect_lib: &str,
        min_compute: (i32, i32),
        producer: OwnedFd,
        completion: OwnedFd,
    ) -> Result<Self, VfxError> {
        let cuda = cuda::driver().ok_or(VfxError::NoCuda)?;
        if !runtime_present(runtime_dir, &[effect_lib]) {
            return Err(VfxError::MissingRuntime(runtime_dir.to_path_buf()));
        }
        let cu = |call, r| cuda::check(call, r).map_err(VfxError::Cuda);
        let mut count = 0;
        // SAFETY: plain driver queries with valid out pointers.
        unsafe { cu("cuDeviceGetCount", (cuda.cuDeviceGetCount)(&mut count))? };
        let mut found = None;
        for i in 0..count {
            let (mut dev, mut id) = (0, [0u8; 16]);
            // SAFETY: as above.
            unsafe {
                cu("cuDeviceGet", (cuda.cuDeviceGet)(&mut dev, i))?;
                cu(
                    "cuDeviceGetUuid",
                    (cuda.cuDeviceGetUuid)(id.as_mut_ptr(), dev),
                )?;
            }
            if id == uuid {
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
        runtime.ensure_effect(effect_lib)?;
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
        let producer_sem = import_semaphore(cuda, producer)?;
        let completion_sem = import_semaphore(cuda, completion)?;
        drop(guard);
        tracing::debug!(
            effect_lib,
            compute = format!("{major}.{minor}"),
            "VFX bridge ready"
        );
        Ok(Self {
            cuda,
            device: dev,
            ctx,
            stream,
            producer_sem,
            completion_sem,
            runtime,
        })
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

    /// Wraps an exported Vulkan buffer (`fd` of an `alloc_size`-byte
    /// dedicated allocation) as an NvCV RGB10A2 image of `width × height`
    /// with rows of `pitch` bytes. CUDA owns `fd` after a successful import.
    pub fn import_image(
        &self,
        fd: OwnedFd,
        alloc_size: u64,
        width: u32,
        height: u32,
        pitch: u32,
    ) -> Result<SharedImage, VfxError> {
        let _g = self.push()?;
        let cu = |call, r| cuda::check(call, r).map_err(VfxError::Cuda);
        let cuda = self.cuda;
        let runtime = self.runtime;
        let raw = fd.into_raw_fd();
        let import = CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: cuda::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: ExternalHandle { fd: raw },
            size: alloc_size,
            flags: cuda::CUDA_EXTERNAL_MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        let mut memory = std::ptr::null_mut();
        // SAFETY: valid descriptor; context pushed above.
        let r = unsafe { (cuda.cuImportExternalMemory)(&mut memory, &import) };
        if r != 0 {
            // SAFETY: still ours after a failed import.
            unsafe { libc::close(raw) };
        }
        cu("cuImportExternalMemory", r)?;
        let mut pointer = 0;
        let map = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset: 0,
            size: u64::from(pitch) * u64::from(height),
            flags: 0,
            reserved: [0; 16],
        };
        // SAFETY: imported memory covers the mapped range.
        unsafe {
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

/// Vulkan memory imported into CUDA, wrapped as an NvCV RGB10A2 image.
pub struct SharedImage {
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
    pub(crate) fn nvcv(&self) -> *mut c_void {
        self.image
    }
}

impl Drop for SharedImage {
    fn drop(&mut self) {
        if let Ok(_g) = CtxGuard::push(self.cuda, self.ctx) {
            // SAFETY: created in `Bridge::import_image`; the owning session
            // drained the stream before dropping images.
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
