//! NVIDIA RTX Video Super Resolution through NGX on D3D12 (RTX Video SDK).
//!
//! Unlike the Video Effects SDK route (CUDA interop, an SDK-internal stream
//! and a result that lags one call behind), NGX records the upscale into
//! the application's own D3D12 command list: it is ordered with the rest of
//! the frame on the GPU and returns the frame it was given.
//!
//! The SDK (`nvsdk_ngx_d.lib`, `nvngx_vsr.dll`) is proprietary and linked
//! only when present at build time (`cfg(ngx)`, see `build.rs`); the FFI
//! below is declared from its public headers. NGX is not thread-safe: every
//! call goes through one process-wide lock.

use std::path::{Path, PathBuf};

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

/// `NVSDK_NGX_VSR_QualityLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsrQuality {
    Bicubic = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Ultra = 4,
}

#[derive(Debug, thiserror::Error)]
pub enum NgxError {
    #[error("built without the NVIDIA RTX Video SDK (see runtime/README.md)")]
    NotBuilt,
    #[error("{call} failed: 0x{code:08X} ({})", result_name(*.code))]
    Call { call: &'static str, code: u32 },
    #[error("RTX Video Super Resolution is not available (GPU or driver)")]
    Unavailable,
}

#[cfg_attr(not(ngx), allow(dead_code))]
fn result_name(code: u32) -> &'static str {
    match code {
        0xBAD0_0001 => "feature not supported",
        0xBAD0_0002 => "platform error",
        0xBAD0_0003 => "feature already exists",
        0xBAD0_0004 => "feature not found",
        0xBAD0_0005 => "invalid parameter",
        0xBAD0_0007 => "not initialized",
        0xBAD0_0008 => "unsupported input format",
        0xBAD0_000B => "unable to initialize feature",
        0xBAD0_000C => "out of date (driver too old)",
        0xBAD0_000E => "unsupported format",
        _ => "error",
    }
}

/// Whether the NGX backend was compiled in.
pub const fn built() -> bool {
    cfg!(ngx)
}

/// Feature DLLs (`nvngx_vsr.dll`) found in `dir`.
pub fn runtime_present(dir: &Path) -> bool {
    dir.join("nvngx_vsr.dll").is_file()
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
    /// Initializes NGX for `device` (once per process) and creates the
    /// feature. Creation work is recorded on `list`, which the caller must
    /// execute before the first [`Self::evaluate`]. `feature_dirs` are
    /// searched for `nvngx_vsr.dll` (after the executable's folder);
    /// `data_dir` must be writable.
    #[cfg_attr(not(ngx), allow(unused_variables))]
    pub fn create(
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        feature_dirs: &[PathBuf],
        data_dir: &Path,
    ) -> Result<Self, NgxError> {
        #[cfg(ngx)]
        {
            imp::Feature::create(device, list, feature_dirs, data_dir).map(|inner| Self { inner })
        }
        #[cfg(not(ngx))]
        {
            Err(NgxError::NotBuilt)
        }
    }

    /// Records an upscale of `input` (`in_size` at the origin) into
    /// `output` (`out_size` at the origin). Both are `R10G10B10A2_UNORM`
    /// textures; `input` must be in a shader-resource state and `output`
    /// (created with UAV access) in `UNORDERED_ACCESS`. NGX changes the
    /// list's state: callers re-bind descriptor heaps afterwards.
    #[cfg_attr(not(ngx), allow(unused_variables))]
    pub fn evaluate(
        &self,
        list: &ID3D12GraphicsCommandList,
        input: &ID3D12Resource,
        in_size: (u32, u32),
        output: &ID3D12Resource,
        out_size: (u32, u32),
        quality: VsrQuality,
    ) -> Result<(), NgxError> {
        #[cfg(ngx)]
        {
            self.inner
                .evaluate(list, input, in_size, output, out_size, quality)
        }
        #[cfg(not(ngx))]
        {
            Err(NgxError::NotBuilt)
        }
    }
}

#[cfg(ngx)]
mod imp {
    use std::ffi::{CStr, c_char, c_void};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    use windows::Win32::Graphics::Direct3D12::{
        ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
    };
    use windows::core::Interface;

    use super::{NgxError, VsrQuality};

    // ---- FFI (nvsdk_ngx.h, nvsdk_ngx_params.h, nvsdk_ngx_defs*.h) ----

    #[repr(C)]
    struct Parameter {
        _p: [u8; 0],
    }
    #[repr(C)]
    struct Handle {
        _p: [u8; 0],
    }

    #[repr(C)]
    struct PathListInfo {
        path: *const *const u16,
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

    /// `NVSDK_NGX_Version_API` of the SDK the declarations follow (1.5).
    const VERSION_API: u32 = 0x15;
    /// `NVSDK_NGX_Feature_VSR` (`NVSDK_NGX_Feature_Reserved16`).
    const FEATURE_VSR: i32 = 16;
    const LOGGING_LEVEL_ON: i32 = 1;

    unsafe extern "C" {
        fn NVSDK_NGX_D3D12_Init(
            app_id: u64,
            data_path: *const u16,
            device: *mut c_void,
            info: *const FeatureCommonInfo,
            version: u32,
        ) -> u32;
        fn NVSDK_NGX_D3D12_GetCapabilityParameters(out: *mut *mut Parameter) -> u32;
        fn NVSDK_NGX_D3D12_CreateFeature(
            list: *mut c_void,
            feature: i32,
            params: *mut Parameter,
            out: *mut *mut Handle,
        ) -> u32;
        fn NVSDK_NGX_D3D12_EvaluateFeature_C(
            list: *mut c_void,
            handle: *const Handle,
            params: *const Parameter,
            callback: *const c_void,
        ) -> u32;
        fn NVSDK_NGX_D3D12_ReleaseFeature(handle: *mut Handle) -> u32;
        fn NVSDK_NGX_Parameter_SetUI(p: *mut Parameter, name: *const c_char, value: u32);
        fn NVSDK_NGX_Parameter_SetD3d12Resource(
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
        device: usize,
        params: usize,
    }

    static INSTANCE: OnceLock<Result<Instance, String>> = OnceLock::new();

    fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    fn instance(
        device: &ID3D12Device,
        feature_dirs: &[PathBuf],
        data_dir: &Path,
    ) -> Result<&'static Instance, NgxError> {
        let init = INSTANCE.get_or_init(|| {
            let dirs: Vec<Vec<u16>> = feature_dirs.iter().map(|d| wide(d.as_os_str())).collect();
            let ptrs: Vec<*const u16> = dirs.iter().map(|d| d.as_ptr()).collect();
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
            let data = wide(data_dir.as_os_str());
            let mut params: *mut Parameter = std::ptr::null_mut();
            // SAFETY: valid device; strings and arrays outlive the calls.
            unsafe {
                check(
                    "NVSDK_NGX_D3D12_Init",
                    NVSDK_NGX_D3D12_Init(0, data.as_ptr(), device.as_raw(), &info, VERSION_API),
                )
                .map_err(|e| e.to_string())?;
                check(
                    "NVSDK_NGX_D3D12_GetCapabilityParameters",
                    NVSDK_NGX_D3D12_GetCapabilityParameters(&mut params),
                )
                .map_err(|e| e.to_string())?;
            }
            Ok(Instance {
                device: device.as_raw() as usize,
                params: params as usize,
            })
        });
        match init {
            Ok(i) if i.device == device.as_raw() as usize => Ok(i),
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

    impl Feature {
        pub(super) fn create(
            device: &ID3D12Device,
            list: &ID3D12GraphicsCommandList,
            feature_dirs: &[PathBuf],
            data_dir: &Path,
        ) -> Result<Self, NgxError> {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let inst = instance(device, feature_dirs, data_dir)?;
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
                    "NVSDK_NGX_D3D12_CreateFeature(VSR)",
                    NVSDK_NGX_D3D12_CreateFeature(list.as_raw(), FEATURE_VSR, params, &mut handle),
                )?;
                Ok(Self { handle, params })
            }
        }

        pub(super) fn evaluate(
            &self,
            list: &ID3D12GraphicsCommandList,
            input: &ID3D12Resource,
            in_size: (u32, u32),
            output: &ID3D12Resource,
            out_size: (u32, u32),
            quality: VsrQuality,
        ) -> Result<(), NgxError> {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let p = self.params;
            // SAFETY: NGX-owned parameter block and feature handle; the
            // resources outlive the recorded command list (caller).
            unsafe {
                NVSDK_NGX_Parameter_SetD3d12Resource(p, c"Input1".as_ptr(), input.as_raw());
                NVSDK_NGX_Parameter_SetD3d12Resource(p, c"Output".as_ptr(), output.as_raw());
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.X".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.Y".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.W".as_ptr(), in_size.0);
                NVSDK_NGX_Parameter_SetUI(p, c"Rect.H".as_ptr(), in_size.1);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.X".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.Y".as_ptr(), 0);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.W".as_ptr(), out_size.0);
                NVSDK_NGX_Parameter_SetUI(p, c"OutRect.H".as_ptr(), out_size.1);
                NVSDK_NGX_Parameter_SetUI(p, c"VSR.QualityLevel".as_ptr(), quality as u32);
                check(
                    "NVSDK_NGX_D3D12_EvaluateFeature(VSR)",
                    NVSDK_NGX_D3D12_EvaluateFeature_C(
                        list.as_raw(),
                        self.handle,
                        p,
                        std::ptr::null(),
                    ),
                )
            }
        }
    }

    impl Drop for Feature {
        fn drop(&mut self) {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // SAFETY: handle created by CreateFeature; the GPU must be idle
            // with respect to it (owner's responsibility).
            unsafe {
                let _ = NVSDK_NGX_D3D12_ReleaseFeature(self.handle);
            }
        }
    }
}
