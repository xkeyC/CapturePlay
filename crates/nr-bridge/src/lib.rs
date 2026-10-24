//! Forwarding host for community `nvngx_dlssnr.dll` builds.
//!
//! NVIDIA's DLSS Neural Rendering snippet only accepts calls from a module
//! whose path contains `nvngx.dll` and returns `FAIL_PlatformError`
//! otherwise. Community builds keep that check, so they cannot be driven from
//! an application executable directly. This DLL is CapturePlay's own code (it
//! contains no NVIDIA code): its file name contains the accepted string and it
//! forwards the NGX D3D12 entry points to the snippet loaded into this module,
//! which makes the snippet see an accepted caller.
//!
//! This is an unofficial, unsupported compatibility path for community
//! runtimes; see README 6.4. Nothing here is used unless the user explicitly
//! selects the community NR backend.

use std::ffi::c_void;
use std::sync::Mutex;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW,
};
use windows::core::{PCSTR, PCWSTR, s};

type InitExt = unsafe extern "system" fn(u64, *const u16, *mut c_void, u32, *const c_void) -> i32;
type Create = unsafe extern "system" fn(*mut c_void, i32, *const c_void, *mut *mut c_void) -> i32;
type Evaluate =
    unsafe extern "system" fn(*mut c_void, *const c_void, *const c_void, *const c_void) -> i32;
type Release = unsafe extern "system" fn(*mut c_void) -> i32;

struct Api {
    _module: isize,
    init: InitExt,
    create: Create,
    evaluate: Evaluate,
    release: Release,
}

// SAFETY: the snippet is loaded once and calls into it are serialized by the
// caller (cp-nvidia's NGX lock).
unsafe impl Send for Api {}

static API: Mutex<Option<Api>> = Mutex::new(None);

/// Forces the result of an indirect call through a volatile read.
///
/// The snippet resolves its caller from the return address of the call. With
/// optimizations the compiler turns a plain `return f(...)` into a tail jump,
/// which makes the snippet see the application as its caller and reject it
/// with FAIL_PlatformError. The volatile read after the call keeps a real call
/// frame in this module.
#[inline(never)]
fn keep_frame(r: i32) -> i32 {
    // SAFETY: reading a local value; volatile cannot be optimized away.
    unsafe { std::ptr::read_volatile(&r) }
}

unsafe fn symbol<T>(module: isize, name: PCSTR) -> Option<T> {
    // SAFETY: the module handle comes from LoadLibraryExW and stays loaded.
    let f = unsafe { GetProcAddress(HMODULE(module as *mut _), name) }?;
    // SAFETY: the caller picks the signature matching the export.
    Some(unsafe { std::mem::transmute_copy::<_, T>(&f) })
}

/// Loads and resolves `path` (UTF-16). Returns 1 on success.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-16 string valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_load(path: *const u16) -> i32 {
    let mut api = match API.lock() {
        Ok(a) => a,
        Err(e) => e.into_inner(),
    };
    if api.is_some() {
        return 1;
    }
    // SAFETY: path is a valid NUL-terminated wide string per the contract.
    let module = match unsafe { LoadLibraryExW(PCWSTR(path), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
    {
        Ok(m) => m,
        Err(_) => return 0,
    };
    let raw = module.0 as isize;
    // SAFETY: names match the snippet's documented NGX exports.
    let (Some(init), Some(create), Some(evaluate), Some(release)) = (unsafe {
        (
            symbol::<InitExt>(raw, s!("NVSDK_NGX_D3D12_Init_Ext")),
            symbol::<Create>(raw, s!("NVSDK_NGX_D3D12_CreateFeature")),
            symbol::<Evaluate>(raw, s!("NVSDK_NGX_D3D12_EvaluateFeature")),
            symbol::<Release>(raw, s!("NVSDK_NGX_D3D12_ReleaseFeature")),
        )
    }) else {
        return 0;
    };
    *api = Some(Api {
        _module: raw,
        init,
        create,
        evaluate,
        release,
    });
    1
}

/// Forwards Init_Ext. Returns 1 (NVSDK_NGX_Result_Success) on success.
///
/// # Safety
/// Pointers must be valid for the snippet's NGX contract.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_init(
    app_id: u64,
    data_path: *const u16,
    device: *mut c_void,
    version: u32,
    params: *const c_void,
) -> i32 {
    let api = API.lock().ok().and_then(|g| g.as_ref().map(|a| a.init));
    match api {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(init) => keep_frame(unsafe { init(app_id, data_path, device, version, params) }),
        None => -1,
    }
}

/// Forwards CreateFeature.
///
/// # Safety
/// Pointers must be valid for the snippet's NGX contract.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_create(
    list: *mut c_void,
    feature: i32,
    params: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    let api = API.lock().ok().and_then(|g| g.as_ref().map(|a| a.create));
    match api {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(create) => keep_frame(unsafe { create(list, feature, params, out) }),
        None => -1,
    }
}

/// Forwards EvaluateFeature.
///
/// # Safety
/// Pointers must be valid for the snippet's NGX contract.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_evaluate(
    list: *mut c_void,
    handle: *const c_void,
    params: *const c_void,
    callback: *const c_void,
) -> i32 {
    let api = API.lock().ok().and_then(|g| g.as_ref().map(|a| a.evaluate));
    match api {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(evaluate) => keep_frame(unsafe { evaluate(list, handle, params, callback) }),
        None => -1,
    }
}

/// Forwards ReleaseFeature.
///
/// # Safety
/// `handle` must come from this snippet.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_release(handle: *mut c_void) -> i32 {
    let api = API.lock().ok().and_then(|g| g.as_ref().map(|a| a.release));
    match api {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(release) => keep_frame(unsafe { release(handle) }),
        None => -1,
    }
}
