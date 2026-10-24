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
    #[error("NVIDIA NGX feature is not available (GPU or driver)")]
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

/// Feature DLL (nvngx_dlssnr.dll) found in dir.
pub fn nr_runtime_present(dir: &Path) -> bool {
    dir.join("nvngx_dlssnr.dll").is_file()
}

/// Create-time options for DLSS Neural Rendering (NGX feature 18).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NrOptions {
    /// DLSSNR.Hint.Render.Preset: 0 leaves the model's default (or auto),
    /// 1 selects the quality model and 2 the performance model on the
    /// 310.9 builds. Earlier builds only know 0.
    pub preset: u32,
}

/// Live model tuning. These keys are read at evaluation time, so changing
/// them does not recreate the feature (the model preset does).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NrTuning {
    /// DLSSNR.Style: 0 default, 1 natural, 2 cinematic.
    pub style: u32,
    /// DLSSNR.Intensity: overall strength of the model's edit.
    pub intensity: f32,
    /// DLSSNR.LocalStructureStrength.
    pub local_structure: f32,
    /// DLSSNR.LocalToneStrength.
    pub local_tone: f32,
    /// DLSSNR.SkinStructureStrength.
    pub skin_structure: f32,
    /// DLSSNR.UseAutoMask.
    pub auto_mask: bool,
    /// DLSSNR.UICorrection.
    pub ui_correction: bool,
}

impl Default for NrTuning {
    fn default() -> Self {
        Self {
            style: 0,
            intensity: 1.0,
            local_structure: 1.0,
            local_tone: 1.0,
            skin_structure: 0.0,
            auto_mask: false,
            ui_correction: false,
        }
    }
}

/// One same-resolution NR evaluation. All four resources have the size the
/// feature was created for; color is display-referred sRGB in FP16, depth is
/// R32_FLOAT, motion is R16G16_FLOAT in render pixels and output is FP16 like
/// color.
#[derive(Debug, Clone, Copy)]
pub struct NrFrame<'a> {
    pub color: &'a ID3D12Resource,
    pub depth: &'a ID3D12Resource,
    pub motion: &'a ID3D12Resource,
    pub output: &'a ID3D12Resource,
    /// Discards temporal history (first frame, source change, cut).
    pub reset: bool,
    pub tuning: NrTuning,
}

/// How the DLSS Neural Rendering feature is obtained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NrRoute {
    /// Through the installed NGX runtime. Feature 18 is not in the public
    /// feature table, so this only works if the driver exposes it.
    Official,
    /// Through the forwarding host `captureplay_nvngx.dll` and an unsigned
    /// community `nvngx_dlssnr.dll`. Unsupported and experimental; see
    /// README 6.4.
    Community { bridge: PathBuf, snippet: PathBuf },
}

/// A DLSS Neural Rendering feature instance, created for one size.
pub struct NrFeature {
    #[cfg(ngx)]
    inner: imp::NrFeature,
}

impl std::fmt::Debug for NrFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NrFeature")
    }
}

impl NrFeature {
    /// Whether the installed NGX runtime advertises feature 18.
    ///
    /// The feature is not part of the public SDK: an NGX runtime that does
    /// not know it reports "not implemented" and feature creation would fail
    /// (or, on some drivers, fault), so callers must ask first.
    #[cfg_attr(not(ngx), allow(unused_variables))]
    pub fn supported(device: &ID3D12Device) -> Result<bool, NgxError> {
        #[cfg(ngx)]
        {
            imp::nr_supported(device)
        }
        #[cfg(not(ngx))]
        {
            Ok(false)
        }
    }

    /// Initializes NGX for device and creates the feature for width x height.
    /// Creation work is recorded on list, which the caller must execute before
    /// the first evaluate call.
    #[cfg_attr(not(ngx), allow(unused_variables))]
    #[cfg_attr(not(ngx), allow(unused_variables))]
    pub fn create(
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        feature_dirs: &[PathBuf],
        data_dir: &Path,
        size: (u32, u32),
        options: NrOptions,
        route: &NrRoute,
    ) -> Result<Self, NgxError> {
        #[cfg(ngx)]
        {
            imp::NrFeature::create(device, list, feature_dirs, data_dir, size, options, route)
                .map(|inner| Self { inner })
        }
        #[cfg(not(ngx))]
        {
            Err(NgxError::NotBuilt)
        }
    }

    /// Records one evaluation. NGX changes the command list's state and
    /// bindings: callers re-bind their descriptor heap afterwards.
    #[cfg_attr(not(ngx), allow(unused_variables))]
    pub fn evaluate(
        &self,
        list: &ID3D12GraphicsCommandList,
        frame: &NrFrame<'_>,
    ) -> Result<(), NgxError> {
        #[cfg(ngx)]
        {
            self.inner.evaluate(list, frame)
        }
        #[cfg(not(ngx))]
        {
            Err(NgxError::NotBuilt)
        }
    }
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
    use std::sync::{Arc, Mutex, OnceLock};

    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D12::{
        ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
    };
    use windows::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW,
    };
    use windows::core::{Interface, PCSTR, PCWSTR, s};

    use super::{NgxError, NrFrame, NrOptions, NrRoute, VsrQuality};

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
        fn NVSDK_NGX_Parameter_SetI(p: *mut Parameter, name: *const c_char, value: i32);
        fn NVSDK_NGX_Parameter_SetF(p: *mut Parameter, name: *const c_char, value: f32);
        fn NVSDK_NGX_D3D12_AllocateParameters(out: *mut *mut Parameter) -> u32;
        fn NVSDK_NGX_D3D12_DestroyParameters(p: *mut Parameter) -> u32;
        fn NVSDK_NGX_D3D12_GetFeatureRequirements(
            adapter: *mut c_void,
            info: *const FeatureDiscoveryInfo,
            out: *mut FeatureRequirement,
        ) -> u32;
    }

    /// NVSDK_NGX_Application_Identifier: a type tag and an 8-byte union
    /// (ProjectDesc is three pointers; only ApplicationId is ever set here).
    #[repr(C)]
    struct AppIdentifier {
        kind: i32,
        _reserved: i32,
        payload: [u64; 3],
    }

    #[repr(C)]
    struct FeatureDiscoveryInfo {
        version: u32,
        feature: i32,
        identifier: AppIdentifier,
        data_path: *const u16,
        feature_info: *const FeatureCommonInfo,
    }

    #[repr(C)]
    struct FeatureRequirement {
        supported: i32,
        min_arch: u32,
        min_os: [u8; 255],
    }

    /// NVSDK_NGX_Feature_Reserved18: DLSS Neural Rendering.
    const FEATURE_NR: i32 = 18;

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

    fn nr_adapter(device: &ID3D12Device) -> Option<windows::Win32::Graphics::Dxgi::IDXGIAdapter> {
        use windows::Win32::Graphics::Dxgi::IDXGIDevice;
        let dxgi: IDXGIDevice = device.cast().ok()?;
        // SAFETY: valid COM object owned by the device.
        unsafe { dxgi.GetAdapter().ok() }
    }

    /// Feature 18 support from the NGX feature table. The caller holds LOCK.
    fn nr_supported_locked(device: &ID3D12Device) -> Result<bool, NgxError> {
        let Some(adapter) = nr_adapter(device) else {
            return Ok(false);
        };
        let info = FeatureDiscoveryInfo {
            version: VERSION_API,
            feature: FEATURE_NR,
            identifier: AppIdentifier {
                kind: 0,
                _reserved: 0,
                payload: [0; 3],
            },
            data_path: std::ptr::null(),
            feature_info: std::ptr::null(),
        };
        let mut requirement = FeatureRequirement {
            supported: 0,
            min_arch: 0,
            min_os: [0; 255],
        };
        // SAFETY: valid adapter and output struct; the query only reads them.
        let r = unsafe {
            NVSDK_NGX_D3D12_GetFeatureRequirements(adapter.as_raw(), &info, &mut requirement)
        };
        if !succeeded(r) {
            // NotImplemented is the normal answer on runtimes without NR.
            return Ok(false);
        }
        Ok(requirement.supported == 0)
    }

    pub(super) fn nr_supported(device: &ID3D12Device) -> Result<bool, NgxError> {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        nr_supported_locked(device)
    }

    type BridgeLoad = unsafe extern "system" fn(*const u16) -> i32;
    type BridgeInit =
        unsafe extern "system" fn(u64, *const u16, *mut c_void, u32, *const c_void) -> i32;
    type BridgeCreate =
        unsafe extern "system" fn(*mut c_void, i32, *const c_void, *mut *mut c_void) -> i32;
    type BridgeEvaluate =
        unsafe extern "system" fn(*mut c_void, *const c_void, *const c_void, *const c_void) -> i32;
    type BridgeRelease = unsafe extern "system" fn(*mut c_void) -> i32;

    /// The CapturePlay forwarding host (captureplay_nvngx.dll) and its
    /// exported entry points. Loading that host rather than the snippet
    /// directly is what makes community builds accept the caller; see the
    /// crate docs of nr-bridge.
    struct Bridge {
        _module: isize,
        load: BridgeLoad,
        init: BridgeInit,
        create: BridgeCreate,
        evaluate: BridgeEvaluate,
        release: BridgeRelease,
    }

    // SAFETY: the DLL stays loaded for the process and calls are serialized
    // by LOCK.
    unsafe impl Send for Bridge {}
    // SAFETY: same as above; the function pointers are immutable after load.
    unsafe impl Sync for Bridge {}

    static BRIDGE: Mutex<Option<(PathBuf, Arc<Bridge>)>> = Mutex::new(None);
    /// Result of the snippet's one-shot Init_Ext.
    static COMMUNITY_INIT: Mutex<Option<std::result::Result<(), u32>>> = Mutex::new(None);

    unsafe fn symbol<T>(module: HMODULE, name: PCSTR) -> Option<T> {
        // SAFETY: module comes from LoadLibraryExW and stays loaded.
        let f = unsafe { GetProcAddress(module, name) }?;
        // SAFETY: the caller picks the signature matching the export.
        Some(unsafe { std::mem::transmute_copy::<_, T>(&f) })
    }

    fn bridge(path: &Path) -> Result<Arc<Bridge>, NgxError> {
        let mut cache = BRIDGE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((known, found)) = cache.as_ref() {
            return if known == path {
                Ok(found.clone())
            } else {
                Err(NgxError::Unavailable)
            };
        }
        let wide_path = wide(path.as_os_str());
        // SAFETY: user-supplied path to our own forwarding DLL.
        let module = unsafe {
            LoadLibraryExW(
                PCWSTR(wide_path.as_ptr()),
                None,
                LOAD_WITH_ALTERED_SEARCH_PATH,
            )
        }
        .map_err(|_| NgxError::Unavailable)?;
        let raw = module.0 as isize;
        // SAFETY: names are the host's documented exports.
        let (Some(load), Some(init), Some(create), Some(evaluate), Some(release)) = (unsafe {
            (
                symbol::<BridgeLoad>(module, s!("cp_nr_load")),
                symbol::<BridgeInit>(module, s!("cp_nr_init")),
                symbol::<BridgeCreate>(module, s!("cp_nr_create")),
                symbol::<BridgeEvaluate>(module, s!("cp_nr_evaluate")),
                symbol::<BridgeRelease>(module, s!("cp_nr_release")),
            )
        }) else {
            return Err(NgxError::Unavailable);
        };
        let found = Arc::new(Bridge {
            _module: raw,
            load,
            init,
            create,
            evaluate,
            release,
        });
        *cache = Some((path.to_owned(), found.clone()));
        Ok(found)
    }

    fn alloc_params() -> Result<*mut Parameter, NgxError> {
        let mut params: *mut Parameter = std::ptr::null_mut();
        // SAFETY: NGX allocates and owns the parameter block.
        check("NVSDK_NGX_D3D12_AllocateParameters", unsafe {
            NVSDK_NGX_D3D12_AllocateParameters(&mut params)
        })?;
        if params.is_null() {
            Err(NgxError::Unavailable)
        } else {
            Ok(params)
        }
    }

    fn fill_create(params: *mut Parameter, width: u32, height: u32, options: NrOptions) {
        // SAFETY: fresh NGX-owned parameter block; neither the generic nor
        // the DLSSNR-prefixed size keys are documented, so both are set.
        unsafe {
            NVSDK_NGX_Parameter_SetUI(params, c"CreationNodeMask".as_ptr(), 1);
            NVSDK_NGX_Parameter_SetUI(params, c"VisibilityNodeMask".as_ptr(), 1);
            NVSDK_NGX_Parameter_SetUI(params, c"Width".as_ptr(), width);
            NVSDK_NGX_Parameter_SetUI(params, c"Height".as_ptr(), height);
            NVSDK_NGX_Parameter_SetUI(params, c"OutWidth".as_ptr(), width);
            NVSDK_NGX_Parameter_SetUI(params, c"OutHeight".as_ptr(), height);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.InputWidth".as_ptr(), width);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.InputHeight".as_ptr(), height);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.OutputWidth".as_ptr(), width);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.OutputHeight".as_ptr(), height);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.Width".as_ptr(), width);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.Height".as_ptr(), height);
            NVSDK_NGX_Parameter_SetI(params, c"DLSSNR.Enabled".as_ptr(), 1);
            NVSDK_NGX_Parameter_SetI(params, c"DLSSNR.Upscaling".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetF(params, c"DLSSNR.Scale".as_ptr(), 1.0);
            NVSDK_NGX_Parameter_SetF(params, c"DLSSNR.ScalingRatio".as_ptr(), 1.0);
            NVSDK_NGX_Parameter_SetI(
                params,
                c"DLSSNR.Hint.Render.Preset".as_ptr(),
                options.preset as i32,
            );
        }
    }

    fn fill_eval(params: *mut Parameter, frame: &NrFrame<'_>) {
        let size = |t: &ID3D12Resource| {
            // NGX wants subrect sizes; every resource passed here is the size
            // the feature was created for, read from its descriptor.
            // SAFETY: valid resource; GetDesc only reads it.
            let desc = unsafe { t.GetDesc() };
            (desc.Width as u32, desc.Height)
        };
        let (w, h) = size(frame.color);
        let out = size(frame.output);
        // SAFETY: NGX-owned parameter block; the resources outlive the
        // recorded command list (owned by the caller's frame).
        unsafe {
            NVSDK_NGX_Parameter_SetD3d12Resource(
                params,
                c"DLSSNR.Color".as_ptr(),
                frame.color.as_raw(),
            );
            NVSDK_NGX_Parameter_SetD3d12Resource(
                params,
                c"DLSSNR.Depth".as_ptr(),
                frame.depth.as_raw(),
            );
            NVSDK_NGX_Parameter_SetD3d12Resource(
                params,
                c"DLSSNR.MVec".as_ptr(),
                frame.motion.as_raw(),
            );
            NVSDK_NGX_Parameter_SetD3d12Resource(
                params,
                c"DLSSNR.Output".as_ptr(),
                frame.output.as_raw(),
            );
            NVSDK_NGX_Parameter_SetI(params, c"DLSSNR.Enabled".as_ptr(), 1);
            NVSDK_NGX_Parameter_SetI(params, c"DLSSNR.Reset".as_ptr(), i32::from(frame.reset));
            NVSDK_NGX_Parameter_SetI(params, c"DLSSNR.DepthInverted".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.ColorSubrectBaseX".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.ColorSubrectBaseY".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.ColorSubrectWidth".as_ptr(), w);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.ColorSubrectHeight".as_ptr(), h);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.DepthSubrectBaseX".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.DepthSubrectBaseY".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.DepthSubrectWidth".as_ptr(), w);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.DepthSubrectHeight".as_ptr(), h);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.MVecSubrectBaseX".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.MVecSubrectBaseY".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.MVecSubrectWidth".as_ptr(), w);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.MVecSubrectHeight".as_ptr(), h);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.OutputSubrectBaseX".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.OutputSubrectBaseY".as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.OutputSubrectWidth".as_ptr(), out.0);
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.OutputSubrectHeight".as_ptr(), out.1);
            NVSDK_NGX_Parameter_SetF(params, c"DLSSNR.MVecScaleX".as_ptr(), 1.0);
            NVSDK_NGX_Parameter_SetF(params, c"DLSSNR.MVecScaleY".as_ptr(), 1.0);
            let t = frame.tuning;
            NVSDK_NGX_Parameter_SetUI(params, c"DLSSNR.Style".as_ptr(), t.style);
            NVSDK_NGX_Parameter_SetF(params, c"DLSSNR.Intensity".as_ptr(), t.intensity);
            NVSDK_NGX_Parameter_SetF(
                params,
                c"DLSSNR.LocalStructureStrength".as_ptr(),
                t.local_structure,
            );
            NVSDK_NGX_Parameter_SetF(params, c"DLSSNR.LocalToneStrength".as_ptr(), t.local_tone);
            NVSDK_NGX_Parameter_SetF(
                params,
                c"DLSSNR.SkinStructureStrength".as_ptr(),
                t.skin_structure,
            );
            NVSDK_NGX_Parameter_SetUI(
                params,
                c"DLSSNR.UseAutoMask".as_ptr(),
                u32::from(t.auto_mask),
            );
            NVSDK_NGX_Parameter_SetUI(
                params,
                c"DLSSNR.UICorrection".as_ptr(),
                u32::from(t.ui_correction),
            );
        }
    }

    enum Route {
        Core,
        Community(Arc<Bridge>),
    }

    pub(super) struct NrFeature {
        handle: *mut Handle,
        params: *mut Parameter,
        route: Route,
    }

    // SAFETY: the handle and parameter block are used under LOCK only.
    unsafe impl Send for NrFeature {}

    impl NrFeature {
        pub(super) fn create(
            device: &ID3D12Device,
            list: &ID3D12GraphicsCommandList,
            feature_dirs: &[PathBuf],
            data_dir: &Path,
            (width, height): (u32, u32),
            options: NrOptions,
            route: &NrRoute,
        ) -> Result<Self, NgxError> {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            instance(device, feature_dirs, data_dir)?;
            match route {
                NrRoute::Official => {
                    if !nr_supported_locked(device)? {
                        tracing::info!(
                            "NGX feature 18 (DLSS Neural Rendering) is not offered by this runtime"
                        );
                        return Err(NgxError::Unavailable);
                    }
                    let params = alloc_params()?;
                    fill_create(params, width, height, options);
                    let mut handle: *mut Handle = std::ptr::null_mut();
                    // SAFETY: valid list, parameter block and out-pointer.
                    let r = unsafe {
                        NVSDK_NGX_D3D12_CreateFeature(
                            list.as_raw(),
                            FEATURE_NR,
                            params,
                            &mut handle,
                        )
                    };
                    if !succeeded(r) || handle.is_null() {
                        // SAFETY: the block is ours and no feature exists.
                        unsafe {
                            let _ = NVSDK_NGX_D3D12_DestroyParameters(params);
                        }
                        return Err(NgxError::Call {
                            call: "NVSDK_NGX_D3D12_CreateFeature(NR)",
                            code: r,
                        });
                    }
                    Ok(Self {
                        handle,
                        params,
                        route: Route::Core,
                    })
                }
                NrRoute::Community {
                    bridge: bridge_path,
                    snippet,
                } => {
                    let bridge = bridge(bridge_path)?;
                    let wide_snippet = wide(snippet.as_os_str());
                    // SAFETY: path is a valid NUL-terminated wide string.
                    if unsafe { (bridge.load)(wide_snippet.as_ptr()) } != 1 {
                        tracing::warn!(?snippet, "cannot load the community NR snippet");
                        return Err(NgxError::Unavailable);
                    }
                    let params = alloc_params()?;
                    let init = {
                        let mut slot = COMMUNITY_INIT
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        match *slot {
                            Some(result) => result,
                            None => {
                                let data = wide(data_dir.as_os_str());
                                // SAFETY: valid device, data path and block.
                                let r = unsafe {
                                    (bridge.init)(
                                        0,
                                        data.as_ptr(),
                                        device.as_raw(),
                                        VERSION_API,
                                        params.cast(),
                                    )
                                };
                                let result = if r == 1 { Ok(()) } else { Err(r as u32) };
                                *slot = Some(result);
                                result
                            }
                        }
                    };
                    if let Err(code) = init {
                        // SAFETY: the block is ours and was not used.
                        unsafe {
                            let _ = NVSDK_NGX_D3D12_DestroyParameters(params);
                        }
                        return Err(NgxError::Call {
                            call: "community snippet Init_Ext",
                            code,
                        });
                    }
                    fill_create(params, width, height, options);
                    let mut handle: *mut Handle = std::ptr::null_mut();
                    let out: *mut *mut c_void = (&mut handle as *mut *mut Handle).cast();
                    // SAFETY: forwarded to the snippet under LOCK.
                    let r =
                        unsafe { (bridge.create)(list.as_raw(), FEATURE_NR, params.cast(), out) };
                    if r != 1 || handle.is_null() {
                        // SAFETY: the block is ours and no feature exists.
                        unsafe {
                            let _ = NVSDK_NGX_D3D12_DestroyParameters(params);
                        }
                        return Err(NgxError::Call {
                            call: "community snippet CreateFeature(NR)",
                            code: r as u32,
                        });
                    }
                    tracing::info!(?snippet, "community DLSS NR feature created");
                    Ok(Self {
                        handle,
                        params,
                        route: Route::Community(bridge),
                    })
                }
            }
        }

        pub(super) fn evaluate(
            &self,
            list: &ID3D12GraphicsCommandList,
            frame: &NrFrame<'_>,
        ) -> Result<(), NgxError> {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            fill_eval(self.params, frame);
            match &self.route {
                // SAFETY: NGX-owned handle and parameter block; the resources
                // outlive the recorded list (owned by the caller's frame).
                Route::Core => check("NVSDK_NGX_D3D12_EvaluateFeature(NR)", unsafe {
                    NVSDK_NGX_D3D12_EvaluateFeature_C(
                        list.as_raw(),
                        self.handle,
                        self.params,
                        std::ptr::null(),
                    )
                }),
                Route::Community(bridge) => {
                    // SAFETY: forwarded to the snippet under LOCK.
                    let r = unsafe {
                        (bridge.evaluate)(
                            list.as_raw(),
                            self.handle.cast(),
                            self.params.cast(),
                            std::ptr::null(),
                        )
                    };
                    if r == 1 {
                        Ok(())
                    } else {
                        Err(NgxError::Call {
                            call: "community snippet EvaluateFeature(NR)",
                            code: r as u32,
                        })
                    }
                }
            }
        }
    }

    impl Drop for NrFeature {
        fn drop(&mut self) {
            let _g = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // SAFETY: handle and parameters were created by NGX; the GPU must
            // be idle with respect to the feature (owner's responsibility).
            unsafe {
                if !self.handle.is_null() {
                    match &self.route {
                        Route::Core => {
                            let _ = NVSDK_NGX_D3D12_ReleaseFeature(self.handle);
                        }
                        Route::Community(bridge) => {
                            let _ = (bridge.release)(self.handle.cast());
                        }
                    }
                }
                if !self.params.is_null() {
                    let _ = NVSDK_NGX_D3D12_DestroyParameters(self.params);
                }
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
