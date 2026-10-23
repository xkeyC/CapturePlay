//! Swapchain and the HDR output contract.
//!
//! The swapchain is **always** FP16 scRGB (`R16G16B16A16_FLOAT` +
//! `G10_NONE_P709`). DWM composes it correctly on SDR and HDR displays, so
//! moving between monitors or toggling Windows HDR never requires a new
//! swapchain format — only the output shader's constants change.
//!
//! The reference implementation lost HDR after going fullscreen because the
//! color space was set once at creation and never again after
//! `ResizeBuffers`. Here [`Presenter::apply_output_contract`] runs after
//! *every* creation and resize, and is idempotent.

use std::rc::Rc;

use cp_core::time::HostTime;

use windows::Win32::Foundation::{DXGI_STATUS_OCCLUDED, HANDLE, HWND, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::WaitForSingleObjectEx;
use windows::core::{Interface, Result};

use crate::descriptors::{CpuDescriptor, StagingHeap};
use crate::display::DisplayTracker;
use crate::gpu::Gpu;
use crate::present_timing::FrameStats;

pub const BACKBUFFER_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;
pub const OUTPUT_COLOR_SPACE: DXGI_COLOR_SPACE_TYPE = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
const BUFFER_COUNT: u32 = 3;

pub struct Presenter {
    swapchain: IDXGISwapChain4,
    hwnd: HWND,
    buffers: Vec<ID3D12Resource>,
    rtvs: Vec<CpuDescriptor>,
    rtv_heap: Rc<StagingHeap>,
    width: u32,
    height: u32,
    waitable: HANDLE,
    tearing_supported: bool,
    flags: DXGI_SWAP_CHAIN_FLAG,
    pub display: DisplayTracker,
}

impl std::fmt::Debug for Presenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Presenter")
            .field("size", &(self.width, self.height))
            .finish()
    }
}

/// What happened on present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentResult {
    Ok,
    /// Window occluded/minimized; skip rendering until visible.
    Occluded,
    DeviceLost,
}

impl Presenter {
    pub fn new(gpu: &Gpu, hwnd: HWND, width: u32, height: u32) -> Result<Self> {
        // SAFETY: standard DXGI swapchain creation for a window we own.
        unsafe {
            let mut tearing = windows::core::BOOL(0);
            let tearing_supported = gpu
                .factory
                .cast::<IDXGIFactory5>()
                .and_then(|f5| {
                    f5.CheckFeatureSupport(
                        DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                        std::ptr::from_mut(&mut tearing).cast(),
                        4,
                    )
                })
                .is_ok()
                && tearing.as_bool();
            let mut flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
            if tearing_supported {
                flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
            }
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width.max(1),
                Height: height.max(1),
                Format: BACKBUFFER_FORMAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFER_COUNT,
                Scaling: DXGI_SCALING_NONE,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                AlphaMode: DXGI_ALPHA_MODE_IGNORE,
                Flags: flags.0 as u32,
                Stereo: false.into(),
            };
            let sc1 = gpu
                .factory
                .CreateSwapChainForHwnd(&gpu.queue, hwnd, &desc, None, None)?;
            // Fullscreen is borderless only; never let DXGI go exclusive.
            gpu.factory
                .MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES)?;
            let swapchain: IDXGISwapChain4 = sc1.cast()?;
            // One frame queued at most: the main latency knob on the output side.
            swapchain.SetMaximumFrameLatency(1)?;
            let waitable = swapchain.GetFrameLatencyWaitableObject();
            let rtv_heap = Rc::new(StagingHeap::new(
                &gpu.device,
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                BUFFER_COUNT,
            )?);
            let mut p = Self {
                swapchain,
                hwnd,
                buffers: Vec::new(),
                rtvs: Vec::new(),
                rtv_heap,
                width: desc.Width,
                height: desc.Height,
                waitable,
                tearing_supported,
                flags,
                display: DisplayTracker::default(),
            };
            p.apply_output_contract();
            p.create_views(gpu)?;
            tracing::info!(
                width,
                height,
                tearing_supported,
                "swapchain created (FP16 scRGB)"
            );
            Ok(p)
        }
    }

    pub fn hwnd(&self) -> HWND {
        self.hwnd
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn tearing_supported(&self) -> bool {
        self.tearing_supported
    }

    /// Color space (+ metadata) for this swapchain. Must run after creation
    /// and after every `ResizeBuffers`.
    pub fn apply_output_contract(&self) {
        // SAFETY: valid swapchain.
        unsafe {
            let support = self
                .swapchain
                .CheckColorSpaceSupport(OUTPUT_COLOR_SPACE)
                .unwrap_or(0);
            let ok = support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32 != 0;
            let set = if ok {
                self.swapchain.SetColorSpace1(OUTPUT_COLOR_SPACE)
            } else {
                Ok(())
            };
            // scRGB carries absolute luminance; HDR10 static metadata does
            // not apply and stale metadata from other paths must be cleared.
            let _ = self
                .swapchain
                .SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, None);
            tracing::debug!(supported = ok, result = ?set, "output contract applied (scRGB G10/P709)");
        }
    }

    fn create_views(&mut self, gpu: &Gpu) -> Result<()> {
        for d in self.rtvs.drain(..) {
            self.rtv_heap.free(d);
        }
        self.buffers.clear();
        for i in 0..BUFFER_COUNT {
            // SAFETY: index < BufferCount.
            let buf: ID3D12Resource = unsafe { self.swapchain.GetBuffer(i)? };
            let rtv = self
                .rtv_heap
                .alloc()
                .ok_or_else(windows::core::Error::empty)?;
            // SAFETY: valid resource and RTV handle.
            unsafe { gpu.device.CreateRenderTargetView(&buf, None, rtv.handle) };
            self.buffers.push(buf);
            self.rtvs.push(rtv);
        }
        Ok(())
    }

    /// Resizes the back buffers. The caller must have waited for the GPU to
    /// finish with them.
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) -> Result<()> {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return Ok(());
        }
        gpu.wait_idle()?;
        for d in self.rtvs.drain(..) {
            self.rtv_heap.free(d);
        }
        self.buffers.clear();
        // SAFETY: all buffer references were dropped above.
        unsafe {
            self.swapchain.ResizeBuffers(
                BUFFER_COUNT,
                width,
                height,
                BACKBUFFER_FORMAT,
                self.flags,
            )?
        };
        self.width = width;
        self.height = height;
        self.apply_output_contract();
        self.create_views(gpu)?;
        self.display.invalidate();
        tracing::debug!(width, height, "swapchain resized");
        Ok(())
    }

    /// Blocks until the swapchain can accept a frame (≤ `timeout_ms`).
    pub fn wait_for_frame(&self, timeout_ms: u32) -> bool {
        // SAFETY: waitable handle owned by the swapchain.
        unsafe { WaitForSingleObjectEx(self.waitable, timeout_ms, true) == WAIT_OBJECT_0 }
    }

    /// Current back buffer and its RTV.
    pub fn back_buffer(&self) -> (&ID3D12Resource, D3D12_CPU_DESCRIPTOR_HANDLE) {
        // SAFETY: valid swapchain.
        let i = unsafe { self.swapchain.GetCurrentBackBufferIndex() } as usize;
        (&self.buffers[i], self.rtvs[i].handle)
    }

    pub fn present(&self, vsync: bool) -> PresentResult {
        let (interval, flags) = if !vsync && self.tearing_supported {
            (0, DXGI_PRESENT_ALLOW_TEARING)
        } else {
            (u32::from(vsync), DXGI_PRESENT(0))
        };
        // SAFETY: valid swapchain.
        let hr = unsafe { self.swapchain.Present(interval, flags) };
        if hr == DXGI_STATUS_OCCLUDED {
            PresentResult::Occluded
        } else if hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET {
            PresentResult::DeviceLost
        } else {
            PresentResult::Ok
        }
    }

    /// Statistics of the latest displayed present (`None` before the first
    /// one or while DXGI reports a disjoint/unavailable state).
    pub fn frame_stats(&self) -> Option<FrameStats> {
        let mut s = DXGI_FRAME_STATISTICS::default();
        // SAFETY: valid swapchain and out pointer.
        unsafe { self.swapchain.GetFrameStatistics(&mut s).ok()? };
        Some(FrameStats {
            present_count: s.PresentCount,
            present_refresh_count: s.PresentRefreshCount,
            sync_refresh_count: s.SyncRefreshCount,
            sync_ns: HostTime::from_qpc(s.SyncQPCTime).0,
        })
    }

    /// Id of the most recent `Present` (matches `FrameStats::present_count`).
    pub fn last_present_count(&self) -> u32 {
        // SAFETY: valid swapchain.
        unsafe { self.swapchain.GetLastPresentCount().unwrap_or(0) }
    }
}
