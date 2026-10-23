//! NVIDIA Optical Flow (NVOFA hardware) via its D3D12 interface.
//!
//! `nvofapi64.dll` ships with the NVIDIA display driver, so nothing is
//! redistributed. Bindings follow `nvOpticalFlowCommon.h` /
//! `nvOpticalFlowD3D12.h` (Optical Flow SDK 5.0, MIT licensed); structs use
//! natural C alignment with the explicit padding fields of the headers.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows::core::{Interface, s, w};

type NvOFHandle = *mut c_void;
type NvOFGPUBufferHandle = *mut c_void;
type NV_OF_STATUS = i32;
const NV_OF_SUCCESS: NV_OF_STATUS = 0;

#[repr(C)]
struct NV_OF_INIT_PARAMS {
    width: u32,
    height: u32,
    outGridSize: u32,
    hintGridSize: u32,
    mode: u32,
    perfLevel: u32,
    enableExternalHints: u32,
    enableOutputCost: u32,
    hPrivData: *mut c_void,
    disparityRange: u32,
    enableRoi: u32,
    predDirection: u32,
    enableGlobalFlow: u32,
    inputBufferFormat: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NV_OF_FENCE_POINT {
    fence: *mut c_void, // ID3D12Fence*
    value: u64,
}

#[repr(C)]
struct NV_OF_REGISTER_RESOURCE_PARAMS_D3D12 {
    resource: *mut c_void, // ID3D12Resource*
    inputFencePoint: NV_OF_FENCE_POINT,
    hOFGpuBuffer: *mut NvOFGPUBufferHandle,
    outputFencePoint: NV_OF_FENCE_POINT,
}

#[repr(C)]
struct NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12 {
    hOFGpuBuffer: NvOFGPUBufferHandle,
}

#[repr(C)]
struct NV_OF_EXECUTE_INPUT_PARAMS_D3D12 {
    inputFrame: NvOFGPUBufferHandle,
    referenceFrame: NvOFGPUBufferHandle,
    externalHints: NvOFGPUBufferHandle,
    disableTemporalHints: u32,
    padding: u32,
    hPrivData: *mut c_void,
    padding2: u32,
    numRois: u32,
    roiData: *mut c_void,
    padding3: u32,
    numFencePoints: u32,
    fencePoint: *mut NV_OF_FENCE_POINT,
}

#[repr(C)]
struct NV_OF_EXECUTE_OUTPUT_PARAMS_D3D12 {
    outputBuffer: NvOFGPUBufferHandle,
    outputCostBuffer: NvOFGPUBufferHandle,
    hPrivData: *mut c_void,
    bwdOutputBuffer: NvOFGPUBufferHandle,
    bwdOutputCostBuffer: NvOFGPUBufferHandle,
    globalFlowBuffer: NvOFGPUBufferHandle,
    padding: [u32; 2],
    fencePoint: *mut NV_OF_FENCE_POINT,
}

type PfnCreate = unsafe extern "system" fn(*mut c_void, *mut NvOFHandle) -> NV_OF_STATUS;
type PfnInit = unsafe extern "system" fn(NvOFHandle, *const NV_OF_INIT_PARAMS) -> NV_OF_STATUS;
type PfnGetSurfaceFormatCount =
    unsafe extern "system" fn(NvOFHandle, u32, u32, *mut u32) -> NV_OF_STATUS;
type PfnGetSurfaceFormat =
    unsafe extern "system" fn(NvOFHandle, u32, u32, *mut u32) -> NV_OF_STATUS;
type PfnRegister = unsafe extern "system" fn(
    NvOFHandle,
    *mut NV_OF_REGISTER_RESOURCE_PARAMS_D3D12,
) -> NV_OF_STATUS;
type PfnUnregister =
    unsafe extern "system" fn(*mut NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12) -> NV_OF_STATUS;
type PfnExecute = unsafe extern "system" fn(
    NvOFHandle,
    *const NV_OF_EXECUTE_INPUT_PARAMS_D3D12,
    *mut NV_OF_EXECUTE_OUTPUT_PARAMS_D3D12,
) -> NV_OF_STATUS;
type PfnDestroy = unsafe extern "system" fn(NvOFHandle) -> NV_OF_STATUS;
type PfnGetLastError = unsafe extern "system" fn(NvOFHandle, *mut u8, *mut u32) -> NV_OF_STATUS;
type PfnGetCaps = unsafe extern "system" fn(NvOFHandle, u32, *mut u32, *mut u32) -> NV_OF_STATUS;

#[repr(C)]
#[derive(Clone, Copy)]
struct FunctionList {
    nvCreateOpticalFlowD3D12: Option<PfnCreate>,
    nvOFInit: Option<PfnInit>,
    nvOFGetSurfaceFormatCountD3D12: Option<PfnGetSurfaceFormatCount>,
    nvOFGetSurfaceFormatD3D12: Option<PfnGetSurfaceFormat>,
    nvOFRegisterResourceD3D12: Option<PfnRegister>,
    nvOFUnregisterResourceD3D12: Option<PfnUnregister>,
    nvOFExecuteD3D12: Option<PfnExecute>,
    nvOFDestroy: Option<PfnDestroy>,
    nvOFGetLastError: Option<PfnGetLastError>,
    nvOFGetCaps: Option<PfnGetCaps>,
}

type PfnCreateInstance = unsafe extern "system" fn(u32, *mut FunctionList) -> NV_OF_STATUS;
type PfnMaxVersion = unsafe extern "system" fn(*mut u32) -> NV_OF_STATUS;

const CAPS_SUPPORTED_OUTPUT_GRID_SIZES: u32 = 0;
const MODE_OPTICALFLOW: u32 = 1;
const BUFFER_FORMAT_GRAYSCALE8: u32 = 1;
const PRED_DIRECTION_BOTH: u32 = 2;

/// NVOFA speed/quality trade-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfLevel {
    Slow = 5,
    Medium = 10,
    Fast = 20,
}

#[derive(Debug, thiserror::Error)]
pub enum NvofError {
    #[error("nvofapi64.dll not available (NVIDIA driver missing?)")]
    NotAvailable,
    #[error("NvOF {call} failed with status {status}: {detail}")]
    Call {
        call: &'static str,
        status: i32,
        detail: String,
    },
}

struct Library {
    _module: HMODULE,
    fns: FunctionList,
}

// SAFETY: the module stays loaded for the process lifetime and the function
// table is immutable after initialization.
unsafe impl Send for Library {}
// SAFETY: as above.
unsafe impl Sync for Library {}

/// The DLL is loaded once and never unloaded: releasing registered D3D12
/// resources after FreeLibrary crashes inside the driver.
fn library() -> Result<&'static Library, NvofError> {
    static LIB: OnceLock<Option<Library>> = OnceLock::new();
    LIB.get_or_init(|| {
        // SAFETY: loading a system DLL by name from System32 only, and
        // resolving exports with the documented signatures.
        unsafe {
            let module =
                LoadLibraryExW(w!("nvofapi64.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
            let max: PfnMaxVersion =
                std::mem::transmute(GetProcAddress(module, s!("NvOFGetMaxSupportedApiVersion"))?);
            let create: PfnCreateInstance =
                std::mem::transmute(GetProcAddress(module, s!("NvOFAPICreateInstanceD3D12"))?);
            let mut version = 0u32;
            if max(&mut version) != NV_OF_SUCCESS {
                return None;
            }
            let mut fns: FunctionList = std::mem::zeroed();
            if create(version, &mut fns) != NV_OF_SUCCESS {
                return None;
            }
            tracing::info!(
                api_version = format!("{:#x}", version),
                "NVIDIA Optical Flow loaded"
            );
            Some(Library {
                _module: module,
                fns,
            })
        }
    })
    .as_ref()
    .ok_or(NvofError::NotAvailable)
}

/// Whether the NVOFA runtime can be loaded on this machine.
pub fn is_available() -> bool {
    library().is_ok()
}

/// Opaque handle of a registered D3D12 resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buffer(usize);

/// A synchronization point `(fence, value)`.
#[derive(Debug, Clone, Copy)]
pub struct FencePoint<'a> {
    pub fence: &'a ID3D12Fence,
    pub value: u64,
}

impl FencePoint<'_> {
    fn raw(&self) -> NV_OF_FENCE_POINT {
        NV_OF_FENCE_POINT {
            fence: self.fence.as_raw(),
            value: self.value,
        }
    }
}

/// Configuration of a session.
#[derive(Debug, Clone, Copy)]
pub struct SessionDesc {
    pub width: u32,
    pub height: u32,
    /// Preferred output grid (1, 2 or 4); the nearest supported is used.
    pub grid: u32,
    pub perf: PerfLevel,
}

/// One bidirectional optical-flow session on GRAYSCALE8 inputs.
pub struct Session {
    lib: &'static Library,
    handle: NvOFHandle,
    grid: u32,
    /// Registered buffers plus the resources they pin (kept alive until
    /// unregistered).
    registered: Vec<(NvOFGPUBufferHandle, ID3D12Resource)>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("grid", &self.grid)
            .field("registered", &self.registered.len())
            .finish()
    }
}

impl Session {
    fn check(&self, call: &'static str, status: NV_OF_STATUS) -> Result<(), NvofError> {
        if status == NV_OF_SUCCESS {
            return Ok(());
        }
        let mut buf = [0u8; 512];
        let mut len = buf.len() as u32;
        if let Some(f) = self.lib.fns.nvOFGetLastError {
            // SAFETY: buffer and length are valid.
            unsafe { f(self.handle, buf.as_mut_ptr(), &mut len) };
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Err(NvofError::Call {
            call,
            status,
            detail: String::from_utf8_lossy(&buf[..end]).into_owned(),
        })
    }

    pub fn new(device: &ID3D12Device, desc: SessionDesc) -> Result<Self, NvofError> {
        let lib = library()?;
        let missing = || NvofError::Call {
            call: "function table",
            status: -1,
            detail: "missing entry".into(),
        };
        let create = lib.fns.nvCreateOpticalFlowD3D12.ok_or_else(missing)?;
        let mut handle: NvOFHandle = std::ptr::null_mut();
        // SAFETY: valid device pointer and out handle.
        let st = unsafe { create(device.as_raw(), &mut handle) };
        if st != NV_OF_SUCCESS || handle.is_null() {
            return Err(NvofError::Call {
                call: "nvCreateOpticalFlowD3D12",
                status: st,
                detail: String::new(),
            });
        }
        let mut session = Self {
            lib,
            handle,
            grid: desc.grid,
            registered: Vec::new(),
        };

        // Pick the closest supported output grid.
        if let Some(caps) = lib.fns.nvOFGetCaps {
            let mut count = 0u32;
            // SAFETY: two-call protocol: count first, then values.
            unsafe {
                if caps(
                    handle,
                    CAPS_SUPPORTED_OUTPUT_GRID_SIZES,
                    std::ptr::null_mut(),
                    &mut count,
                ) == NV_OF_SUCCESS
                    && count > 0
                    && count <= 16
                {
                    let mut grids = vec![0u32; count as usize];
                    if caps(
                        handle,
                        CAPS_SUPPORTED_OUTPUT_GRID_SIZES,
                        grids.as_mut_ptr(),
                        &mut count,
                    ) == NV_OF_SUCCESS
                    {
                        grids.truncate(count as usize);
                        if let Some(g) = grids
                            .iter()
                            .copied()
                            .filter(|&g| g >= desc.grid)
                            .min()
                            .or_else(|| grids.iter().copied().max())
                        {
                            session.grid = g;
                        }
                        tracing::debug!(?grids, chosen = session.grid, "NvOF grid sizes");
                    }
                }
            }
        }

        let init = NV_OF_INIT_PARAMS {
            width: desc.width,
            height: desc.height,
            outGridSize: session.grid,
            hintGridSize: 0,
            mode: MODE_OPTICALFLOW,
            perfLevel: desc.perf as u32,
            enableExternalHints: 0,
            enableOutputCost: 1,
            hPrivData: std::ptr::null_mut(),
            disparityRange: 0,
            enableRoi: 0,
            predDirection: PRED_DIRECTION_BOTH,
            enableGlobalFlow: 0,
            inputBufferFormat: BUFFER_FORMAT_GRAYSCALE8,
        };
        let f = lib.fns.nvOFInit.ok_or_else(missing)?;
        // SAFETY: valid handle and init params.
        let st = unsafe { f(handle, &init) };
        session.check("nvOFInit", st)?;
        Ok(session)
    }

    /// Output grid in input pixels.
    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// Registers a resource. NVOFA waits for `wait` before touching it and
    /// signals `signal` when registration is done.
    pub fn register(
        &mut self,
        resource: &ID3D12Resource,
        wait: FencePoint<'_>,
        signal: FencePoint<'_>,
    ) -> Result<Buffer, NvofError> {
        let f = self
            .lib
            .fns
            .nvOFRegisterResourceD3D12
            .ok_or(NvofError::NotAvailable)?;
        let mut h: NvOFGPUBufferHandle = std::ptr::null_mut();
        let mut p = NV_OF_REGISTER_RESOURCE_PARAMS_D3D12 {
            resource: resource.as_raw(),
            inputFencePoint: wait.raw(),
            hOFGpuBuffer: &mut h,
            outputFencePoint: signal.raw(),
        };
        // SAFETY: all pointers valid for the call.
        let st = unsafe { f(self.handle, &mut p) };
        self.check("nvOFRegisterResourceD3D12", st)?;
        self.registered.push((h, resource.clone()));
        Ok(Buffer(h as usize))
    }

    /// Computes flow `input → reference` (into `forward`) and
    /// `reference → input` (into `backward`), waiting for `wait` and
    /// signaling `signal` on completion.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        input: Buffer,
        reference: Buffer,
        forward: Buffer,
        forward_cost: Buffer,
        backward: Buffer,
        backward_cost: Buffer,
        wait: FencePoint<'_>,
        signal: FencePoint<'_>,
    ) -> Result<(), NvofError> {
        let f = self
            .lib
            .fns
            .nvOFExecuteD3D12
            .ok_or(NvofError::NotAvailable)?;
        let mut wait_point = wait.raw();
        let mut signal_point = signal.raw();
        let inp = NV_OF_EXECUTE_INPUT_PARAMS_D3D12 {
            inputFrame: input.0 as NvOFGPUBufferHandle,
            referenceFrame: reference.0 as NvOFGPUBufferHandle,
            externalHints: std::ptr::null_mut(),
            // Frames are consecutive but pairs may skip (stalls, cuts);
            // temporal hints would carry stale vectors across those.
            disableTemporalHints: 1,
            padding: 0,
            hPrivData: std::ptr::null_mut(),
            padding2: 0,
            numRois: 0,
            roiData: std::ptr::null_mut(),
            padding3: 0,
            numFencePoints: 1,
            fencePoint: &mut wait_point,
        };
        let mut out = NV_OF_EXECUTE_OUTPUT_PARAMS_D3D12 {
            outputBuffer: forward.0 as NvOFGPUBufferHandle,
            outputCostBuffer: forward_cost.0 as NvOFGPUBufferHandle,
            hPrivData: std::ptr::null_mut(),
            bwdOutputBuffer: backward.0 as NvOFGPUBufferHandle,
            bwdOutputCostBuffer: backward_cost.0 as NvOFGPUBufferHandle,
            globalFlowBuffer: std::ptr::null_mut(),
            padding: [0; 2],
            fencePoint: &mut signal_point,
        };
        // SAFETY: all handles were registered on this session; pointers are
        // valid for the duration of the call.
        let st = unsafe { f(self.handle, &inp, &mut out) };
        self.check("nvOFExecuteD3D12", st)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Order matters: unregister → destroy → release resources.
        if let Some(unregister) = self.lib.fns.nvOFUnregisterResourceD3D12 {
            for (h, _) in &self.registered {
                let mut p = NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12 { hOFGpuBuffer: *h };
                // SAFETY: handle registered on this session.
                unsafe { unregister(&mut p) };
            }
        }
        if let Some(destroy) = self.lib.fns.nvOFDestroy {
            // SAFETY: handle created by this session.
            unsafe { destroy(self.handle) };
        }
        self.registered.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout checks against the C headers (x64, natural alignment).
    #[test]
    fn struct_layouts_match_headers() {
        assert_eq!(size_of::<NV_OF_INIT_PARAMS>(), 64);
        assert_eq!(size_of::<NV_OF_FENCE_POINT>(), 16);
        assert_eq!(size_of::<NV_OF_REGISTER_RESOURCE_PARAMS_D3D12>(), 48);
        assert_eq!(
            std::mem::offset_of!(NV_OF_EXECUTE_INPUT_PARAMS_D3D12, hPrivData),
            32
        );
        assert_eq!(
            std::mem::offset_of!(NV_OF_EXECUTE_INPUT_PARAMS_D3D12, roiData),
            48
        );
        assert_eq!(
            std::mem::offset_of!(NV_OF_EXECUTE_INPUT_PARAMS_D3D12, fencePoint),
            64
        );
        assert_eq!(
            std::mem::offset_of!(NV_OF_EXECUTE_OUTPUT_PARAMS_D3D12, fencePoint),
            56
        );
        assert_eq!(size_of::<FunctionList>(), 80);
    }
}
