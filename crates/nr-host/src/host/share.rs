//! Handing Vulkan memory and semaphores to the native process.
//!
//! winevulkan offers only the Win32 flavours of external memory and
//! semaphores. It implements them on top of the host driver's opaque file
//! descriptors: allocating exportable memory (or creating an exportable
//! semaphore) sends the host fd to wineserver, which keeps it in a D3DKMT
//! object, and win32u keeps a server handle to that object in this process.
//! The NT handle `vkGet*Win32HandleKHR` returns only names an outer
//! "shared resource" wrapper without an fd, so instead we find the inner
//! object's handle — one that appeared while the object was created, whose
//! object has no type name (wineserver's D3DKMT objects are untyped, unlike
//! the events, files and sections other threads may create meanwhile) and
//! that ntdll's `wine_server_handle_to_fd` accepts — and get the fd from
//! it. The fd number is valid in this (Linux) process; the native side
//! takes its own reference with `pidfd_getfd`.
//!
//! This relies on Wine internals (win32u's D3DKMT objects, Wine 10 and
//! later); the native side verifies what it imports.

use std::ffi::c_void;

use ash::vk;
use windows::Win32::Foundation::{GetHandleInformation, HANDLE};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::core::{s, w};

use super::gpu::{Gpu, Result};

/// `NTSTATUS wine_server_handle_to_fd(HANDLE, unsigned access, int *fd,
/// unsigned *options)`.
type HandleToFd = unsafe extern "C" fn(*mut c_void, u32, *mut i32, *mut u32) -> i32;

/// The Linux fd behind a wineserver handle (a new descriptor of this
/// process; it stays open for the native side to adopt).
pub fn handle_to_fd(handle: HANDLE) -> Result<i32> {
    // Wine's server helper (absent on real Windows).
    let f: HandleToFd = ntdll(s!("wine_server_handle_to_fd"))?;
    let (mut fd, mut options) = (-1, 0);
    // SAFETY: valid handle and out-pointers.
    let status = unsafe { f(handle.0, 0, &mut fd, &mut options) };
    if status != 0 || fd < 0 {
        return Err(format!("wine_server_handle_to_fd: status {status:#x}"));
    }
    Ok(fd)
}

/// `NTSTATUS NtQueryObject(HANDLE, OBJECT_INFORMATION_CLASS, void *info,
/// ULONG size, ULONG *returned)`.
type QueryObject = unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32, *mut u32) -> i32;
const OBJECT_TYPE_INFORMATION: u32 = 2;

fn ntdll<T>(name: windows::core::PCSTR) -> Result<T> {
    // SAFETY: ntdll is always loaded; the caller names the export's type.
    unsafe {
        let ntdll = GetModuleHandleW(w!("ntdll.dll")).map_err(|e| format!("ntdll: {e}"))?;
        let p =
            GetProcAddress(ntdll, name).ok_or_else(|| format!("ntdll lacks {}", name.display()))?;
        Ok(std::mem::transmute_copy(&p))
    }
}

/// The object type name behind `handle` ("" for untyped objects).
fn type_name(query: QueryObject, handle: usize) -> Option<String> {
    // PUBLIC_OBJECT_TYPE_INFORMATION: UNICODE_STRING (u16 len, u16 max,
    // pointer) followed by reserved words and the name's characters.
    let mut buf = [0u64; 128];
    let mut returned = 0;
    // SAFETY: the buffer is large enough for any type name.
    let status = unsafe {
        query(
            handle as *mut c_void,
            OBJECT_TYPE_INFORMATION,
            buf.as_mut_ptr().cast(),
            std::mem::size_of_val(&buf) as u32,
            &mut returned,
        )
    };
    if status != 0 {
        return None;
    }
    let len = (buf[0] & 0xffff) as usize / 2;
    let ptr = buf[1] as *const u16;
    if len == 0 || ptr.is_null() {
        return Some(String::new());
    }
    // SAFETY: the string lives inside `buf` (or wherever NtQueryObject put it).
    Some(String::from_utf16_lossy(unsafe {
        std::slice::from_raw_parts(ptr, len)
    }))
}

/// Handle values scanned for new objects (Wine hands out the lowest free
/// slot, in steps of 4).
const HANDLE_SCAN_LIMIT: usize = 0x4000;

fn live_handles() -> Vec<usize> {
    (4..HANDLE_SCAN_LIMIT)
        .step_by(4)
        .filter(|&h| {
            let mut flags = 0;
            // SAFETY: querying an arbitrary handle value is harmless.
            unsafe { GetHandleInformation(HANDLE(h as *mut c_void), &mut flags) }.is_ok()
        })
        .collect()
}

/// Runs `create` (which makes one exportable Vulkan object) and returns its
/// result with the Linux fd of the D3DKMT object it left behind.
pub fn with_new_fd<T>(create: impl FnOnce() -> Result<T>) -> Result<(T, i32)> {
    let query: QueryObject = ntdll(s!("NtQueryObject"))?;
    let before = live_handles();
    let value = create()?;
    let fresh: Vec<(usize, String)> = live_handles()
        .into_iter()
        .filter(|h| before.binary_search(h).is_err())
        .map(|h| (h, type_name(query, h).unwrap_or_else(|| "?".into())))
        .collect();
    log!("nr-host: new handles {fresh:x?}");
    for (h, ty) in &fresh {
        if !ty.is_empty() {
            continue;
        }
        if let Ok(fd) = handle_to_fd(HANDLE(*h as *mut c_void)) {
            log!("nr-host: handle {h:#x} -> fd {fd}");
            return Ok((value, fd));
        }
    }
    Err(format!(
        "no new untyped handle with an fd among {fresh:x?} (not a Wine 10+ winevulkan?)"
    ))
}

/// An exportable timeline semaphore and its Linux fd.
pub fn timeline_semaphore(gpu: &Gpu) -> Result<(vk::Semaphore, i32)> {
    with_new_fd(|| create_timeline(gpu))
}

fn create_timeline(gpu: &Gpu) -> Result<vk::Semaphore> {
    let mut timeline = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let mut export = vk::ExportSemaphoreCreateInfo::default()
        .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
    // SAFETY: valid device and create info.
    let semaphore = unsafe {
        gpu.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default()
                .push_next(&mut timeline)
                .push_next(&mut export),
            None,
        )
    }
    .map_err(|e| format!("vkCreateSemaphore(timeline, exportable): {e:?}"))?;
    Ok(semaphore)
}
