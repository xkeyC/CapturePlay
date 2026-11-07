//! NVIDIA RTX Video Super Resolution through NGX on Vulkan (Linux).
//!
//! The Vulkan counterpart of [`crate::ngx`] on D3D12: the upscale is
//! recorded into the application's own command buffer. The loader is the
//! static NGX library of NVIDIA's DLSS SDK (linked when present at build
//! time, `cfg(ngx)`); the feature itself is the `libnvidia-ngx-vsr.so`
//! snippet shipped with NVIDIA's Video Effects SDK, found through the
//! feature search path. NGX is not thread-safe: every call goes through one
//! process-wide lock.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use ash::vk;

pub use crate::ngx_common::{NgxError, NrOptions, NrTuning, VsrQuality};

/// Whether the NGX backend was compiled in.
pub const fn built() -> bool {
    cfg!(ngx)
}

/// The VSR feature snippet (`libnvidia-ngx-vsr.so.*`) is in `dir`.
pub fn runtime_present(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.filter_map(Result::ok).any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("libnvidia-ngx-vsr.so"))
        })
    })
}

/// Vulkan instance and device extensions NGX needs, to enable when the
/// instance/device are created. `None` when NGX is not built in.
pub fn required_extensions() -> Option<(Vec<CString>, Vec<CString>)> {
    #[cfg(ngx)]
    {
        imp::required_extensions()
    }
    #[cfg(not(ngx))]
    {
        None
    }
}

/// The Vulkan objects NGX is initialized with.
#[derive(Clone, Copy)]
pub struct VkContext {
    pub instance: vk::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: vk::Device,
    pub get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

impl std::fmt::Debug for VkContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkContext")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// An image NGX reads or writes, in `GENERAL` layout.
#[derive(Debug, Clone, Copy)]
pub struct VkImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

/// An RTX VSR feature instance. Sizes and quality are given per call, so one
/// instance serves any resolution.
pub struct VsrFeature {
    #[cfg(ngx)]
    inner: imp::Feature,
}

impl std::fmt::Debug for VsrFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VsrFeature")
    }
}

impl VsrFeature {
    /// Initializes NGX for `ctx.device` (once per process) and creates the
    /// feature. Creation work is recorded on `cmd`, which the caller must
    /// submit before the first [`Self::evaluate`]. `feature_dirs` are
    /// searched for the snippet; `data_dir` must be writable.
    #[cfg_attr(not(ngx), allow(unused_variables))]
    pub fn create(
        ctx: &VkContext,
        cmd: vk::CommandBuffer,
        feature_dirs: &[PathBuf],
        data_dir: &Path,
    ) -> Result<Self, NgxError> {
        #[cfg(ngx)]
        {
            imp::Feature::create(ctx, cmd, feature_dirs, data_dir).map(|inner| Self { inner })
        }
        #[cfg(not(ngx))]
        {
            Err(NgxError::NotBuilt)
        }
    }

    /// Records an upscale of `input` (`in_size` at the origin) into
    /// `output` (`out_size` at the origin); both `A2B10G10R10_UNORM` images
    /// in `GENERAL` layout, `output` with storage usage. NGX changes the
    /// bound pipeline and descriptors: callers re-bind afterwards.
    #[cfg_attr(not(ngx), allow(unused_variables))]
    pub fn evaluate(
        &self,
        cmd: vk::CommandBuffer,
        input: &VkImage,
        in_size: (u32, u32),
        output: &VkImage,
        out_size: (u32, u32),
        quality: VsrQuality,
    ) -> Result<(), NgxError> {
        #[cfg(ngx)]
        {
            self.inner
                .evaluate(cmd, input, in_size, output, out_size, quality)
        }
        #[cfg(not(ngx))]
        {
            Err(NgxError::NotBuilt)
        }
    }
}

#[cfg(ngx)]
mod imp {
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    use ash::vk;

    use super::{NgxError, VkContext, VkImage, VsrQuality};

    // ---- FFI (nvsdk_ngx_vk.h, nvsdk_ngx_defs*.h, nvsdk_ngx_params.h) ----

    #[repr(C)]
    struct Parameter {
        _p: [u8; 0],
    }
    #[repr(C)]
    struct Handle {
        _p: [u8; 0],
    }

    /// `wchar_t` is UTF-32 on Linux.
    type WChar = u32;

    #[repr(C)]
    struct PathListInfo {
        path: *const *const WChar,
        length: u32,
    }

    type LogCallback = unsafe extern "C" fn(*const c_char, i32, i32);

    #[repr(C)]
    struct LoggingInfo {
        callback: Option<LogCallback>,
        minimum_level: i32,
        disable_other_sinks: bool,
    }

    #[repr(C)]
    struct FeatureCommonInfo {
        path_list: PathListInfo,
        internal: *mut c_void,
        logging: LoggingInfo,
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

    /// `NVSDK_NGX_Resource_VK` with its union holding the image-view
    /// member (the larger one).
    #[repr(C)]
    struct ResourceVk {
        info: ImageViewInfo,
        kind: i32,
        read_write: bool,
    }

    const RESOURCE_VK_TYPE_IMAGEVIEW: i32 = 0;
    /// `NVSDK_NGX_Version_API` of the SDK the declarations follow (1.5).
    const VERSION_API: u32 = 0x15;
    /// `NVSDK_NGX_Feature_VSR` (`NVSDK_NGX_Feature_Reserved16`).
    const FEATURE_VSR: i32 = 16;
    const ENGINE_TYPE_CUSTOM: i32 = 0;
    const LOGGING_LEVEL_ON: i32 = 1;
    /// CapturePlay's NGX project id (`NVSDK_NGX_ENGINE_TYPE_CUSTOM`).
    const PROJECT_ID: &CStr = c"9a5c3e2f-4b1d-4c8e-a7f6-3d2b1c0e9f84";

    unsafe extern "C" {
        fn NVSDK_NGX_VULKAN_RequiredExtensions(
            instance_count: *mut u32,
            instance_exts: *mut *const *const c_char,
            device_count: *mut u32,
            device_exts: *mut *const *const c_char,
        ) -> u32;
        #[allow(clippy::too_many_arguments)]
        fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
            project_id: *const c_char,
            engine: i32,
            engine_version: *const c_char,
            data_path: *const WChar,
            instance: vk::Instance,
            physical: vk::PhysicalDevice,
            device: vk::Device,
            gipa: vk::PFN_vkGetInstanceProcAddr,
            gdpa: vk::PFN_vkGetDeviceProcAddr,
            info: *const FeatureCommonInfo,
            version: u32,
        ) -> u32;
        fn NVSDK_NGX_VULKAN_GetCapabilityParameters(out: *mut *mut Parameter) -> u32;
        fn NVSDK_NGX_VULKAN_CreateFeature(
            cmd: vk::CommandBuffer,
            feature: i32,
            params: *mut Parameter,
            out: *mut *mut Handle,
        ) -> u32;
        fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
            cmd: vk::CommandBuffer,
            handle: *const Handle,
            params: *const Parameter,
            callback: *const c_void,
        ) -> u32;
        fn NVSDK_NGX_Parameter_SetUI(p: *mut Parameter, name: *const c_char, value: u32);
        fn NVSDK_NGX_Parameter_SetVoidPointer(
            p: *mut Parameter,
            name: *const c_char,
            value: *mut c_void,
        );
        fn NVSDK_NGX_Parameter_GetI(p: *mut Parameter, name: *const c_char, out: *mut i32) -> u32;
    }

    fn succeeded(r: u32) -> bool {
        (r & 0xFFF0_0000) != 0xBAD0_0000
    }

    fn check(call: &'static str, r: u32) -> Result<(), NgxError> {
        if succeeded(r) {
            Ok(())
        } else {
            Err(NgxError::Call { call, code: r })
        }
    }

    unsafe extern "C" fn log_line(message: *const c_char, _level: i32, _feature: i32) {
        if message.is_null() {
            return;
        }
        // SAFETY: NGX passes a NUL-terminated string valid for this call.
        let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
        tracing::debug!(target: "ngx", "{}", text.trim_end());
    }

    /// Serializes every NGX call.
    static LOCK: Mutex<()> = Mutex::new(());

    /// The process-wide NGX instance: capability parameters (owned by NGX)
    /// for the device it was initialized with. NGX stays initialized until
    /// the process exits, like the other NVIDIA runtimes.
    struct Instance {
        device: vk::Device,
        params: usize,
    }

    static INSTANCE: OnceLock<Result<Instance, String>> = OnceLock::new();

    fn wide(s: &Path) -> Vec<WChar> {
        s.to_string_lossy()
            .chars()
            .map(|c| c as WChar)
            .chain(std::iter::once(0))
            .collect()
    }

    pub(super) fn required_extensions() -> Option<(Vec<CString>, Vec<CString>)> {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut ni, mut nd) = (0u32, 0u32);
        let (mut pi, mut pd): (*const *const c_char, *const *const c_char) =
            (std::ptr::null(), std::ptr::null());
        // SAFETY: NGX returns static arrays of NUL-terminated names.
        unsafe {
            let r = NVSDK_NGX_VULKAN_RequiredExtensions(&mut ni, &mut pi, &mut nd, &mut pd);
            if !succeeded(r) {
                tracing::warn!("NVSDK_NGX_VULKAN_RequiredExtensions failed: 0x{r:08X}");
                return None;
            }
            let read = |p: *const *const c_char, n: u32| -> Vec<CString> {
                if p.is_null() {
                    return Vec::new();
                }
                (0..n as usize)
                    .map(|i| *p.add(i))
                    .filter(|s| !s.is_null())
                    .map(|s| CStr::from_ptr(s).to_owned())
                    .collect()
            };
            Some((read(pi, ni), read(pd, nd)))
        }
    }

    fn instance(
        ctx: &VkContext,
        feature_dirs: &[PathBuf],
        data_dir: &Path,
    ) -> Result<&'static Instance, NgxError> {
        let init = INSTANCE.get_or_init(|| {
            let dirs: Vec<Vec<WChar>> = feature_dirs.iter().map(|d| wide(d)).collect();
            let ptrs: Vec<*const WChar> = dirs.iter().map(|d| d.as_ptr()).collect();
            let info = FeatureCommonInfo {
                path_list: PathListInfo {
                    path: ptrs.as_ptr(),
                    length: ptrs.len() as u32,
                },
                internal: std::ptr::null_mut(),
                logging: LoggingInfo {
                    callback: Some(log_line),
                    minimum_level: LOGGING_LEVEL_ON,
                    disable_other_sinks: true,
                },
            };
            let data = wide(data_dir);
            let mut params: *mut Parameter = std::ptr::null_mut();
            // SAFETY: valid Vulkan objects; strings and arrays outlive the
            // calls.
            unsafe {
                check(
                    "NVSDK_NGX_VULKAN_Init_with_ProjectID",
                    NVSDK_NGX_VULKAN_Init_with_ProjectID(
                        PROJECT_ID.as_ptr(),
                        ENGINE_TYPE_CUSTOM,
                        c"0.1".as_ptr(),
                        data.as_ptr(),
                        ctx.instance,
                        ctx.physical,
                        ctx.device,
                        ctx.get_instance_proc_addr,
                        ctx.get_device_proc_addr,
                        &info,
                        VERSION_API,
                    ),
                )
                .map_err(|e| e.to_string())?;
                check(
                    "NVSDK_NGX_VULKAN_GetCapabilityParameters",
                    NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut params),
                )
                .map_err(|e| e.to_string())?;
            }
            Ok(Instance {
                device: ctx.device,
                params: params as usize,
            })
        });
        match init {
            Ok(i) if i.device == ctx.device => Ok(i),
            Ok(_) => Err(NgxError::Call {
                call: "NGX already initialized for another device",
                code: 0xBAD0_0002,
            }),
            Err(e) => {
                tracing::warn!("NGX unavailable: {e}");
                Err(NgxError::Unavailable)
            }
        }
    }

    pub(super) struct Feature {
        handle: *mut Handle,
        params: *mut Parameter,
    }

    // SAFETY: the handle is used under `LOCK` only, from one thread at a time.
    unsafe impl Send for Feature {}

    fn resource(img: &VkImage, read_write: bool) -> ResourceVk {
        ResourceVk {
            info: ImageViewInfo {
                view: img.view,
                image: img.image,
                range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                format: img.format,
                width: img.width,
                height: img.height,
            },
            kind: RESOURCE_VK_TYPE_IMAGEVIEW,
            read_write,
        }
    }

    impl Feature {
        pub(super) fn create(
            ctx: &VkContext,
            cmd: vk::CommandBuffer,
            feature_dirs: &[PathBuf],
            data_dir: &Path,
        ) -> Result<Self, NgxError> {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let inst = instance(ctx, feature_dirs, data_dir)?;
            let params = inst.params as *mut Parameter;
            let mut available = 0i32;
            // SAFETY: NGX-owned parameter block; valid names and pointers.
            unsafe {
                let r = NVSDK_NGX_Parameter_GetI(params, c"VSR.Available".as_ptr(), &mut available);
                if !succeeded(r) || available == 0 {
                    return Err(NgxError::Unavailable);
                }
                NVSDK_NGX_Parameter_SetUI(params, c"CreationNodeMask".as_ptr(), 1);
                NVSDK_NGX_Parameter_SetUI(params, c"VisibilityNodeMask".as_ptr(), 1);
                let mut handle: *mut Handle = std::ptr::null_mut();
                check(
                    "NVSDK_NGX_VULKAN_CreateFeature(VSR)",
                    NVSDK_NGX_VULKAN_CreateFeature(cmd, FEATURE_VSR, params, &mut handle),
                )?;
                Ok(Self { handle, params })
            }
        }

        pub(super) fn evaluate(
            &self,
            cmd: vk::CommandBuffer,
            input: &VkImage,
            in_size: (u32, u32),
            output: &VkImage,
            out_size: (u32, u32),
            quality: VsrQuality,
        ) -> Result<(), NgxError> {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let p = self.params;
            let mut input = resource(input, false);
            let mut output = resource(output, true);
            // SAFETY: NGX-owned parameter block and feature handle; the
            // resource descriptions outlive the evaluate call (NGX reads
            // them while recording) and the images outlive the command
            // buffer (caller).
            unsafe {
                NVSDK_NGX_Parameter_SetVoidPointer(
                    p,
                    c"Input1".as_ptr(),
                    std::ptr::from_mut(&mut input).cast(),
                );
                NVSDK_NGX_Parameter_SetVoidPointer(
                    p,
                    c"Output".as_ptr(),
                    std::ptr::from_mut(&mut output).cast(),
                );
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.X".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.Y".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.W".as_ptr(), in_size.0);
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.H".as_ptr(), in_size.1);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.X".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.Y".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.W".as_ptr(), out_size.0);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.H".as_ptr(), out_size.1);
                NVSDK_NGX_Parameter_SetUI(p, c"VSR.QualityLevel".as_ptr(), quality as u32);
                let r = NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd, self.handle, p, std::ptr::null());
                // The descriptions die with this call.
                NVSDK_NGX_Parameter_SetVoidPointer(p, c"Input1".as_ptr(), std::ptr::null_mut());
                NVSDK_NGX_Parameter_SetVoidPointer(p, c"Output".as_ptr(), std::ptr::null_mut());
                check("NVSDK_NGX_VULKAN_EvaluateFeature(VSR)", r)
            }
        }
    }
}
