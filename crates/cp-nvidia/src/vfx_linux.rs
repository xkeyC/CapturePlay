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

use std::ffi::{CString, c_void};
use std::os::fd::{IntoRawFd, OwnedFd};
use std::path::{Path, PathBuf};

use crate::cuda::{
    self, CUDA_EXTERNAL_MEMORY_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    CUexternalSemaphore, Cuda, ExternalHandle,
};
pub(crate) use crate::vfx_common::Effect;
use crate::vfx_common::{Api, Core, cu};
pub use crate::vfx_common::{Runtime, SharedImage, VfxError};

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

/// Loads the runtime of `dir` (preloads, then the core libraries) and
/// resolves its entry points.
pub(crate) fn load_runtime_libraries(dir: &Path) -> Result<Api, VfxError> {
    for p in PRELOAD.iter().filter_map(|n| find_lib(dir, n)) {
        load_lib(&p)?;
    }
    let modules = CORE_LIBS
        .iter()
        .map(|n| load_lib(&dir.join(n)))
        .collect::<Result<Vec<_>, _>>()?;
    let (core, image) = (modules[3], modules[0]);
    // SAFETY: the libraries are the Video Effects SDK's.
    unsafe { Api::resolve(|n| sym(core, n), |n| sym(image, n)) }
        .ok_or_else(|| VfxError::MissingRuntime(dir.to_path_buf()))
}

/// Loads an effect library (never unloaded).
pub(crate) fn load_effect_library(path: &Path) -> Result<(), VfxError> {
    load_lib(path).map(|_| ())
}

/// What a [`SharedImage`] keeps alive besides the CUDA import: nothing, the
/// Vulkan side is the caller's.
pub(crate) type ImageOwner = ();

/// Whether the core libraries and `effect_libs` exist in `dir`.
pub fn runtime_present(dir: &Path, effect_libs: &[&str]) -> bool {
    CORE_LIBS
        .iter()
        .chain(effect_libs)
        .all(|n| dir.join(n).is_file())
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
/// semaphores (see [`Core`], which it dereferences to).
#[derive(Debug)]
pub struct Bridge {
    core: Core,
}

impl std::ops::Deref for Bridge {
    type Target = Core;
    fn deref(&self) -> &Core {
        &self.core
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
        let matches = |cuda: &Cuda, dev| {
            let mut id = [0u8; 16];
            // SAFETY: plain driver query with a valid out pointer.
            unsafe {
                cu(
                    "cuDeviceGetUuid",
                    (cuda.cuDeviceGetUuid)(id.as_mut_ptr(), dev),
                )?
            };
            Ok(id == uuid)
        };
        let import = |cuda: &'static Cuda| {
            Ok((
                import_semaphore(cuda, producer)?,
                import_semaphore(cuda, completion)?,
            ))
        };
        Ok(Self {
            core: Core::new(runtime_dir, effect_lib, min_compute, matches, import)?,
        })
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
        let r = unsafe { (self.cuda.cuImportExternalMemory)(&mut memory, &import) };
        if r != 0 {
            // SAFETY: still ours after a failed import.
            unsafe { libc::close(raw) };
        }
        cu("cuImportExternalMemory", r)?;
        self.wrap(memory, (width, height, pitch), ())
    }
}
