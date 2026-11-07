//! The snippet's NGX Vulkan entry points, forwarded the same way as the
//! D3D12 ones.
//!
//! Used by `nr-host`, the helper that runs the snippet under Wine/Proton
//! for the Linux build: there the snippet drives the host's Vulkan driver
//! through winevulkan, and no NGX core is involved: snippets that export a
//! parameter allocator provide the blocks, otherwise the host brings its
//! own `NVSDK_NGX_Parameter`.

use std::ffi::c_void;
use std::sync::Mutex;

use windows::Win32::System::LibraryLoader::{LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW};
use windows::core::{PCWSTR, s};

use super::{keep_frame, symbol};

/// `NVSDK_NGX_VULKAN_Init_Ext(app id, data path, instance, physical device,
/// device, SDK version, feature info)`.
type InitExt = unsafe extern "system" fn(
    u64,
    *const u16,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u32,
    *const c_void,
) -> i32;
type Create = unsafe extern "system" fn(*mut c_void, i32, *mut c_void, *mut *mut c_void) -> i32;
type Evaluate =
    unsafe extern "system" fn(*mut c_void, *const c_void, *const c_void, *const c_void) -> i32;
type HandleCall = unsafe extern "system" fn(*mut c_void) -> i32;
type Allocate = unsafe extern "system" fn(*mut *mut c_void) -> i32;

#[derive(Clone, Copy)]
struct Api {
    init: InitExt,
    create: Create,
    evaluate: Evaluate,
    release: HandleCall,
    allocate: Option<Allocate>,
    destroy: Option<HandleCall>,
    shutdown: Option<HandleCall>,
}

static API: Mutex<Option<Api>> = Mutex::new(None);

fn api() -> Option<Api> {
    *API.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Loads `path` (UTF-16) and resolves its Vulkan entry points. Returns 1 on
/// success.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-16 string valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_load(path: *const u16) -> i32 {
    let mut slot = API
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.is_some() {
        return 1;
    }
    // SAFETY: path is a valid NUL-terminated wide string per the contract.
    let Ok(module) = (unsafe { LoadLibraryExW(PCWSTR(path), None, LOAD_WITH_ALTERED_SEARCH_PATH) })
    else {
        return 0;
    };
    let raw = module.0 as isize;
    // SAFETY: names match the snippet's NGX Vulkan exports.
    let resolved = unsafe {
        (
            symbol::<InitExt>(raw, s!("NVSDK_NGX_VULKAN_Init_Ext")),
            symbol::<Create>(raw, s!("NVSDK_NGX_VULKAN_CreateFeature")),
            symbol::<Evaluate>(raw, s!("NVSDK_NGX_VULKAN_EvaluateFeature")),
            symbol::<HandleCall>(raw, s!("NVSDK_NGX_VULKAN_ReleaseFeature")),
            symbol::<Allocate>(raw, s!("NVSDK_NGX_VULKAN_AllocateParameters")),
            symbol::<HandleCall>(raw, s!("NVSDK_NGX_VULKAN_DestroyParameters")),
            symbol::<HandleCall>(raw, s!("NVSDK_NGX_VULKAN_Shutdown1")),
        )
    };
    let (Some(init), Some(create), Some(evaluate), Some(release)) =
        (resolved.0, resolved.1, resolved.2, resolved.3)
    else {
        return 0;
    };
    let (allocate, destroy) = (resolved.4, resolved.5);
    *slot = Some(Api {
        init,
        create,
        evaluate,
        release,
        allocate,
        destroy,
        shutdown: resolved.6,
    });
    1
}

/// Forwards `NVSDK_NGX_VULKAN_Init_Ext`.
///
/// # Safety
/// Pointers must be valid for the snippet's NGX contract.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_init(
    app_id: u64,
    data_path: *const u16,
    instance: *mut c_void,
    physical_device: *mut c_void,
    device: *mut c_void,
    version: u32,
    feature_info: *const c_void,
) -> i32 {
    match api() {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(a) => keep_frame(unsafe {
            (a.init)(
                app_id,
                data_path,
                instance,
                physical_device,
                device,
                version,
                feature_info,
            )
        }),
        None => -1,
    }
}

/// Forwards `NVSDK_NGX_VULKAN_AllocateParameters` (-1 when the snippet
/// lacks it).
///
/// # Safety
/// `out` must be valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_allocate_parameters(out: *mut *mut c_void) -> i32 {
    match api().and_then(|a| a.allocate) {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(allocate) => keep_frame(unsafe { allocate(out) }),
        None => -1,
    }
}

/// Forwards `NVSDK_NGX_VULKAN_DestroyParameters`.
///
/// # Safety
/// `params` must come from [`cp_nr_vk_allocate_parameters`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_destroy_parameters(params: *mut c_void) -> i32 {
    match api().and_then(|a| a.destroy) {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(destroy) => keep_frame(unsafe { destroy(params) }),
        None => -1,
    }
}

/// Forwards `NVSDK_NGX_VULKAN_CreateFeature`.
///
/// # Safety
/// Pointers must be valid for the snippet's NGX contract.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_create(
    cmd: *mut c_void,
    feature: i32,
    params: *mut c_void,
    out: *mut *mut c_void,
) -> i32 {
    match api() {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(a) => keep_frame(unsafe { (a.create)(cmd, feature, params, out) }),
        None => -1,
    }
}

/// Forwards `NVSDK_NGX_VULKAN_EvaluateFeature`.
///
/// # Safety
/// Pointers must be valid for the snippet's NGX contract.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_evaluate(
    cmd: *mut c_void,
    handle: *const c_void,
    params: *const c_void,
    callback: *const c_void,
) -> i32 {
    match api() {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(a) => keep_frame(unsafe { (a.evaluate)(cmd, handle, params, callback) }),
        None => -1,
    }
}

/// Forwards `NVSDK_NGX_VULKAN_ReleaseFeature`.
///
/// # Safety
/// `handle` must come from this snippet.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_release(handle: *mut c_void) -> i32 {
    match api() {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(a) => keep_frame(unsafe { (a.release)(handle) }),
        None => -1,
    }
}

/// Forwards `NVSDK_NGX_VULKAN_Shutdown1` (0 when the snippet lacks it).
///
/// # Safety
/// `device` must be the device passed to [`cp_nr_vk_init`], idle.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cp_nr_vk_shutdown(device: *mut c_void) -> i32 {
    match api().and_then(|a| a.shutdown) {
        // SAFETY: forwarded verbatim under the caller's contract.
        Some(shutdown) => keep_frame(unsafe { shutdown(device) }),
        None => 0,
    }
}
