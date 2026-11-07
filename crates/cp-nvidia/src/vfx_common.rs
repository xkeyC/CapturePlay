//! The platform-independent part of the Video Effects plumbing: the NvVFX /
//! NvCV entry points, the loaded runtime, effects, the CUDA side of a
//! session (context, stream, the two external semaphores) and images
//! wrapped around imported memory. The platform modules (`vfx.rs` on
//! Windows, `vfx_linux.rs` on Linux) load the libraries and import the
//! graphics API's memory and semaphores.

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::cuda::{
    self, CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_SEMAPHORE_PARAMS, CUcontext, CUdevice,
    CUdeviceptr, CUexternalMemory, CUexternalSemaphore, CUstream, Cuda,
};
use crate::vfx::{ImageOwner, load_effect_library, load_runtime_libraries};

/// NvCV packed RGB10A2 pixel format / 32-bit packed component type (R in
/// the low bits, like `DXGI_FORMAT_R10G10B10A2_UNORM` and Vulkan
/// `A2B10G10R10_UNORM_PACK32`).
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
    #[error("no CUDA device matches the rendering device")]
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
    #[cfg(windows)]
    #[error("D3D12: {0}")]
    D3d(#[from] windows::core::Error),
}

pub(crate) type NvStatus = i32;

/// Entry points of the core (NvVFX) and image (NvCV) libraries.
pub(crate) struct Api {
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

impl Api {
    /// Resolves every entry point: `core` looks up symbols of the NvVFX
    /// library, `image` those of the NvCV library.
    ///
    /// # Safety
    /// The resolvers must return symbols of the Video Effects SDK, whose
    /// signatures the fields declare.
    pub(crate) unsafe fn resolve(
        core: impl Fn(&str) -> Option<*mut c_void>,
        image: impl Fn(&str) -> Option<*mut c_void>,
    ) -> Option<Self> {
        /// # Safety
        /// `p` is the named SDK function, of type `T`.
        unsafe fn f<T>(p: *mut c_void) -> T {
            // SAFETY: per the function contract; `T` is a fn pointer.
            unsafe { std::mem::transmute_copy(&p) }
        }
        // SAFETY: per the contract, every symbol is the named SDK function.
        unsafe {
            Some(Self {
                create: f(core("NvVFX_CreateEffect")?),
                destroy: f(core("NvVFX_DestroyEffect")?),
                set_u32: f(core("NvVFX_SetU32")?),
                set_f32: f(core("NvVFX_SetF32")?),
                set_image: f(core("NvVFX_SetImage")?),
                set_stream: f(core("NvVFX_SetCudaStream")?),
                load: f(core("NvVFX_Load")?),
                run: f(core("NvVFX_Run")?),
                image_create: f(image("NvCVImage_Create")?),
                image_dealloc: f(image("NvCVImage_Dealloc")?),
                image_destroy: f(image("NvCVImage_Destroy")?),
                image_init: f(image("NvCVImage_Init")?),
                error_string: f(image("NvCV_GetErrorStringFromCode")?),
            })
        }
    }
}

/// A loaded runtime directory. Lives for the rest of the process: its
/// libraries are never unloaded (effects run internal worker threads).
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
        let api = load_runtime_libraries(dir)?;
        let rt: &'static Runtime = Box::leak(Box::new(Runtime {
            dir: dir.to_path_buf(),
            api,
            effects: Mutex::new(Vec::new()),
        }));
        map.insert(dir.to_path_buf(), rt);
        Ok(rt)
    }

    /// Loads an effect library (e.g. the VFG one) once.
    fn ensure_effect(&self, name: &str) -> Result<(), VfxError> {
        let mut loaded = self
            .effects
            .lock()
            .map_err(|_| VfxError::MissingRuntime(self.dir.clone()))?;
        if !loaded.iter().any(|d| d == name) {
            load_effect_library(&self.dir.join(name))?;
            loaded.push(name.to_string());
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

pub(crate) fn cu(call: &'static str, r: i32) -> Result<(), VfxError> {
    cuda::check(call, r).map_err(VfxError::Cuda)
}

/// The CUDA side of a session: device, retained primary context, stream and
/// the two imported semaphores (`producer`: the graphics queue signals that
/// inputs are written; `completion`: CUDA signals that outputs are).
/// Platform bridges wrap it and dereference to it.
pub struct Core {
    pub(crate) cuda: &'static Cuda,
    device: CUdevice,
    pub(crate) ctx: CUcontext,
    stream: CUstream,
    producer_sem: CUexternalSemaphore,
    completion_sem: CUexternalSemaphore,
    pub(crate) runtime: &'static Runtime,
}

impl std::fmt::Debug for Core {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bridge")
            .field("runtime", &self.runtime)
            .finish()
    }
}

impl Core {
    /// Finds the CUDA device `matches` accepts, checks `min_compute`, loads
    /// the runtime + `effect`, retains the device's primary context, creates
    /// the stream and imports the semaphores with `import` (context current;
    /// returns `(producer, completion)`).
    pub(crate) fn new(
        runtime_dir: &Path,
        effect: &str,
        min_compute: (i32, i32),
        matches: impl Fn(&Cuda, CUdevice) -> Result<bool, VfxError>,
        import: impl FnOnce(
            &'static Cuda,
        ) -> Result<(CUexternalSemaphore, CUexternalSemaphore), VfxError>,
    ) -> Result<Self, VfxError> {
        let cuda = cuda::driver().ok_or(VfxError::NoCuda)?;
        if !crate::vfx::runtime_present(runtime_dir, &[effect]) {
            return Err(VfxError::MissingRuntime(runtime_dir.to_path_buf()));
        }
        let mut count = 0;
        // SAFETY: plain driver queries with valid out pointers.
        unsafe { cu("cuDeviceGetCount", (cuda.cuDeviceGetCount)(&mut count))? };
        let mut found = None;
        for i in 0..count {
            let mut dev = 0;
            // SAFETY: as above.
            unsafe { cu("cuDeviceGet", (cuda.cuDeviceGet)(&mut dev, i))? };
            if matches(cuda, dev)? {
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
        runtime.ensure_effect(effect)?;
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
        let (producer_sem, completion_sem) = import(cuda)?;
        drop(guard);
        tracing::debug!(
            effect,
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
        cu("cuWaitExternalSemaphoresAsync", r)
    }

    /// Enqueues `completion = value` on the stream.
    pub(crate) fn signal(&self, value: u64) -> Result<(), VfxError> {
        let p = CUDA_EXTERNAL_SEMAPHORE_PARAMS::fence(value);
        // SAFETY: as above.
        let r = unsafe {
            (self.cuda.cuSignalExternalSemaphoresAsync)(&self.completion_sem, &p, 1, self.stream)
        };
        cu("cuSignalExternalSemaphoresAsync", r)
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

    /// Wraps imported `memory` (rows of `pitch` bytes) as an NvCV RGB10A2
    /// image of `width × height`; the image owns `memory` and `owner` (the
    /// graphics API's object, kept alive with it). Context current.
    pub(crate) fn wrap(
        &self,
        memory: CUexternalMemory,
        (width, height, pitch): (u32, u32, u32),
        owner: ImageOwner,
    ) -> Result<SharedImage, VfxError> {
        let (cuda, runtime) = (self.cuda, self.runtime);
        let mut pointer = 0;
        let map = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset: 0,
            size: u64::from(pitch) * u64::from(height),
            flags: 0,
            reserved: [0; 16],
        };
        // SAFETY: imported memory covers the mapped range.
        unsafe {
            let r = (cuda.cuExternalMemoryGetMappedBuffer)(&mut pointer, memory, &map);
            if r != 0 {
                (cuda.cuDestroyExternalMemory)(memory);
            }
            cu("cuExternalMemoryGetMappedBuffer", r)?;
            // Let NvCV allocate its descriptor, then point it at our memory.
            let mut image = std::ptr::null_mut();
            let created = runtime.check(
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
            );
            if let Err(e) = created {
                (cuda.cuMemFree)(pointer);
                (cuda.cuDestroyExternalMemory)(memory);
                return Err(e);
            }
            (runtime.api.image_dealloc)(image);
            let img = SharedImage {
                owner,
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

impl Drop for Core {
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

/// Memory shared with the graphics API, wrapped as an NvCV RGB10A2 image.
pub struct SharedImage {
    /// The graphics API's side (the D3D12 buffer on Windows).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) owner: ImageOwner,
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
            // SAFETY: created in `Core::wrap`; the owning session drained
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
