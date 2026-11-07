//! Feature 18 through `captureplay_nvngx.dll`, with our own parameter
//! blocks ([`super::params`]): the snippet exports no allocator.

use std::ffi::c_void;
use std::path::Path;

use ash::vk;
use ash::vk::Handle as _;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW,
};
use windows::core::{PCSTR, PCWSTR, s};

use super::gpu::{Gpu, Image, Result};
use super::params::Params;

/// `NVSDK_NGX_Result_Success`.
const SUCCESS: i32 = 1;
const FEATURE_NR: i32 = 18;
/// Application id the snippet accepts for an unregistered caller, with the
/// SDK versions to try (newest first). The Windows build passes 0 / 1.5
/// with a core-allocated block; without a core the snippet wants these.
const INIT_ATTEMPTS: [(u64, u32); 3] = [(0x0876_232C, 0x14), (0x0876_232C, 0x13), (0, 0x15)];

pub fn wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}

type Load = unsafe extern "system" fn(*const u16) -> i32;
type Init = unsafe extern "system" fn(
    u64,
    *const u16,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u32,
    *const c_void,
) -> i32;
type HandleCall = unsafe extern "system" fn(*mut c_void) -> i32;
type Create = unsafe extern "system" fn(*mut c_void, i32, *mut c_void, *mut *mut c_void) -> i32;
type Evaluate =
    unsafe extern "system" fn(*mut c_void, *const c_void, *const c_void, *const c_void) -> i32;

/// The forwarding host and its Vulkan exports.
pub struct Bridge {
    init: Init,
    create: Create,
    evaluate: Evaluate,
    release: HandleCall,
    shutdown: HandleCall,
}

unsafe fn export<T>(module: HMODULE, name: PCSTR) -> Result<T> {
    // SAFETY: module comes from LoadLibraryExW and stays loaded.
    let f = unsafe { GetProcAddress(module, name) }.ok_or_else(|| {
        // SAFETY: the names below are NUL-terminated literals.
        format!("export {} missing", unsafe { name.display() })
    })?;
    // SAFETY: the caller picks the signature matching the export.
    Ok(unsafe { std::mem::transmute_copy::<_, T>(&f) })
}

impl Bridge {
    /// Loads `bridge` (captureplay_nvngx.dll) and through it `snippet`.
    pub fn load(bridge: &Path, snippet: &Path) -> Result<Self> {
        let path = wide(bridge);
        // SAFETY: NUL-terminated path to our own forwarding DLL.
        let module =
            unsafe { LoadLibraryExW(PCWSTR(path.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
                .map_err(|e| format!("{}: {e}", bridge.display()))?;
        // SAFETY: names and signatures of nr-bridge's `vulkan` module.
        let b = unsafe {
            let load: Load = export(module, s!("cp_nr_vk_load"))?;
            let snippet_path = wide(snippet);
            if load(snippet_path.as_ptr()) != 1 {
                return Err(format!(
                    "{}: not loadable or no NGX Vulkan exports",
                    snippet.display()
                ));
            }
            Self {
                init: export(module, s!("cp_nr_vk_init"))?,
                create: export(module, s!("cp_nr_vk_create"))?,
                evaluate: export(module, s!("cp_nr_vk_evaluate"))?,
                release: export(module, s!("cp_nr_vk_release"))?,
                shutdown: export(module, s!("cp_nr_vk_shutdown"))?,
            }
        };
        Ok(b)
    }

    /// `Init_Ext` on the device; returns the (app id, version) that worked.
    pub fn init(&self, gpu: &Gpu, data_dir: &Path) -> Result<(u64, u32)> {
        let data = wide(data_dir);
        let mut last = 0;
        for (app, version) in INIT_ATTEMPTS {
            // SAFETY: live instance/device handles and a NUL-terminated path.
            let r = unsafe {
                (self.init)(
                    app,
                    data.as_ptr(),
                    gpu.instance.handle().as_raw() as *mut c_void,
                    gpu.physical.as_raw() as *mut c_void,
                    gpu.device.handle().as_raw() as *mut c_void,
                    version,
                    std::ptr::null(),
                )
            };
            log!("nr-host: Init_Ext(app {app:#x}, version {version:#x}) -> {r:#x}");
            if r == SUCCESS {
                return Ok((app, version));
            }
            last = r;
        }
        Err(format!("Init_Ext failed ({last:#x})"))
    }

    pub fn shutdown(&self, gpu: &Gpu) {
        // SAFETY: the device is idle and every feature was released.
        unsafe {
            (self.shutdown)(gpu.device.handle().as_raw() as *mut c_void);
        }
    }
}

/// `NVSDK_NGX_ImageViewInfo_VK`.
#[repr(C)]
struct ImageViewInfo {
    view: vk::ImageView,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
}

/// `NVSDK_NGX_Resource_VK` holding an image view.
#[repr(C)]
pub struct ResourceVk {
    info: ImageViewInfo,
    kind: i32,
    read_write: bool,
}

pub fn resource(img: &Image, read_write: bool) -> ResourceVk {
    ResourceVk {
        info: ImageViewInfo {
            view: img.view,
            image: img.image,
            range: super::gpu::color_range(),
            format: img.format,
            width: img.width,
            height: img.height,
        },
        kind: 0,
        read_write,
    }
}

/// Live model tuning (the shared DLSSNR parameter list).
pub use nr_proto::dlssnr::Tuning;

/// Sets one `nr_proto::dlssnr` value.
fn set_value(p: &Params, key: &std::ffi::CStr, value: nr_proto::dlssnr::Value) {
    use nr_proto::dlssnr::Value;
    match value {
        Value::U32(v) => p.set_u32(key, v),
        Value::I32(v) => p.set_i32(key, v),
        Value::F32(v) => p.set_f32(key, v),
    }
}

/// The inputs of one evaluation.
pub struct Frame<'a> {
    pub color: &'a Image,
    pub depth: &'a Image,
    pub motion: &'a Image,
    pub output: &'a Image,
    pub reset: bool,
    pub tuning: Tuning,
}

pub struct Feature<'b> {
    handle: *mut c_void,
    params: Params,
    bridge: &'b Bridge,
}

impl<'b> Feature<'b> {
    /// Records creation into `cmd` (submit it before evaluating).
    pub fn create(
        bridge: &'b Bridge,
        cmd: vk::CommandBuffer,
        (width, height): (u32, u32),
        preset: u32,
    ) -> Result<Self> {
        let p = Params::new();
        nr_proto::dlssnr::create(width, height, preset, &mut |k, v| set_value(&p, k, v));
        let mut handle = std::ptr::null_mut();
        // SAFETY: recording command buffer, live parameter block.
        let r = unsafe {
            (bridge.create)(
                cmd.as_raw() as *mut c_void,
                FEATURE_NR,
                p.raw(),
                &mut handle,
            )
        };
        if r != SUCCESS || handle.is_null() {
            return Err(format!(
                "CreateFeature(18) -> {r:#x} (DLSSNR.Available = {:?})",
                p.get_u32(c"DLSSNR.Available")
            ));
        }
        Ok(Self {
            handle,
            params: p,
            bridge,
        })
    }

    /// Records one evaluation into `cmd`. The resources must stay alive
    /// until the command buffer completed.
    pub fn evaluate(&self, cmd: vk::CommandBuffer, f: &Frame<'_>) -> Result<()> {
        let p = &self.params;
        let mut color = resource(f.color, false);
        let mut depth = resource(f.depth, false);
        let mut motion = resource(f.motion, false);
        let mut output = resource(f.output, true);
        p.set_ptr(c"DLSSNR.Color", (&raw mut color).cast());
        p.set_ptr(c"DLSSNR.Depth", (&raw mut depth).cast());
        p.set_ptr(c"DLSSNR.MVec", (&raw mut motion).cast());
        p.set_ptr(c"DLSSNR.Output", (&raw mut output).cast());
        nr_proto::dlssnr::evaluate(
            (f.color.width, f.color.height),
            (f.output.width, f.output.height),
            f.reset,
            &f.tuning,
            &mut |k, v| set_value(p, k, v),
        );
        // SAFETY: recording command buffer; the resource structs are read
        // during the call, the images outlive the submission.
        let r = unsafe {
            (self.bridge.evaluate)(
                cmd.as_raw() as *mut c_void,
                self.handle,
                p.raw(),
                std::ptr::null(),
            )
        };
        if r == SUCCESS {
            Ok(())
        } else {
            Err(format!("EvaluateFeature(18) -> {r:#x}"))
        }
    }
}

impl Drop for Feature<'_> {
    fn drop(&mut self) {
        // SAFETY: the owner waited for the GPU.
        unsafe {
            (self.bridge.release)(self.handle);
        }
    }
}
