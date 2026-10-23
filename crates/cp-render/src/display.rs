//! Which display the window is on and what it can do.
//!
//! Re-queried whenever the window moves/resizes, the DXGI factory goes stale
//! (`IsCurrent() == false`: HDR toggled, monitor plugged…) or a display
//! change is broadcast. A failed query never downgrades: the last good state
//! for the same monitor is kept.

use windows::Win32::Devices::Display::*;
use windows::Win32::Foundation::{ERROR_SUCCESS, HWND, RECT};
use windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;
use windows::core::Interface;

#[derive(Debug, Clone, PartialEq)]
pub struct DisplayInfo {
    /// GDI device name (`\\.\DISPLAY1`).
    pub device_name: String,
    /// Windows HDR ("Use HDR") is on for this display.
    pub hdr: bool,
    pub max_nits: f32,
    pub min_nits: f32,
    pub max_full_frame_nits: f32,
    /// Windows "SDR content brightness" in nits.
    pub sdr_white_nits: f32,
    pub refresh_hz: f64,
}

impl DisplayInfo {
    /// Conservative SDR fallback.
    pub fn fallback() -> Self {
        Self {
            device_name: String::new(),
            hdr: false,
            max_nits: 80.0,
            min_nits: 0.0,
            max_full_frame_nits: 80.0,
            sdr_white_nits: 80.0,
            refresh_hz: 60.0,
        }
    }

    pub fn refresh_period_ns(&self) -> i64 {
        (1e9 / self.refresh_hz.max(1.0)) as i64
    }
}

fn intersect_area(a: &RECT, b: &RECT) -> i64 {
    let w = (a.right.min(b.right) - a.left.max(b.left)).max(0);
    let h = (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0);
    i64::from(w) * i64::from(h)
}

fn wide_to_string(w: &[u16]) -> String {
    let n = w.iter().position(|&c| c == 0).unwrap_or(w.len());
    String::from_utf16_lossy(&w[..n])
}

/// Queries the display with the largest overlap with `hwnd`.
pub fn query(factory: &IDXGIFactory6, hwnd: HWND) -> Option<DisplayInfo> {
    let mut wr = RECT::default();
    // SAFETY: valid window handle owned by the app.
    unsafe { GetWindowRect(hwnd, &mut wr).ok()? };
    let mut best: Option<(i64, DXGI_OUTPUT_DESC1)> = None;
    // SAFETY: plain DXGI enumeration.
    unsafe {
        for a in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(a) else {
                break;
            };
            for o in 0.. {
                let Ok(output) = adapter.EnumOutputs(o) else {
                    break;
                };
                let Ok(o6) = output.cast::<IDXGIOutput6>() else {
                    continue;
                };
                let Ok(desc) = o6.GetDesc1() else { continue };
                let area = intersect_area(&wr, &desc.DesktopCoordinates);
                if best.as_ref().is_none_or(|(b, _)| area > *b) {
                    best = Some((area, desc));
                }
            }
        }
    }
    let (_, desc) = best?;
    let device_name = wide_to_string(&desc.DeviceName);
    let (sdr_white_nits, refresh_hz) = display_config(&device_name).unwrap_or((80.0, 60.0));
    Some(DisplayInfo {
        hdr: desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        max_nits: desc.MaxLuminance.max(80.0),
        min_nits: desc.MinLuminance,
        max_full_frame_nits: desc.MaxFullFrameLuminance.max(80.0),
        sdr_white_nits,
        refresh_hz,
        device_name,
    })
}

/// `(sdr_white_nits, refresh_hz)` for a GDI device name via DisplayConfig.
fn display_config(device_name: &str) -> Option<(f32, f64)> {
    // SAFETY: buffer sizes come from GetDisplayConfigBufferSizes; request
    // packets are fully initialized with matching type/size.
    unsafe {
        let (mut np, mut nm) = (0u32, 0u32);
        if GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut np, &mut nm) != ERROR_SUCCESS {
            return None;
        }
        let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); np as usize];
        let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); nm as usize];
        if QueryDisplayConfig(
            QDC_ONLY_ACTIVE_PATHS,
            &mut np,
            paths.as_mut_ptr(),
            &mut nm,
            modes.as_mut_ptr(),
            None,
        ) != ERROR_SUCCESS
        {
            return None;
        }
        paths.truncate(np as usize);
        for p in &paths {
            let mut src = DISPLAYCONFIG_SOURCE_DEVICE_NAME::default();
            src.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            src.header.size = size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32;
            src.header.adapterId = p.sourceInfo.adapterId;
            src.header.id = p.sourceInfo.id;
            if DisplayConfigGetDeviceInfo(&mut src.header) != 0
                || wide_to_string(&src.viewGdiDeviceName) != device_name
            {
                continue;
            }
            let mut white = DISPLAYCONFIG_SDR_WHITE_LEVEL::default();
            white.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
            white.header.size = size_of::<DISPLAYCONFIG_SDR_WHITE_LEVEL>() as u32;
            white.header.adapterId = p.targetInfo.adapterId;
            white.header.id = p.targetInfo.id;
            let nits = if DisplayConfigGetDeviceInfo(&mut white.header) == 0 {
                white.SDRWhiteLevel as f32 / 1000.0 * 80.0
            } else {
                80.0
            };
            let r = p.targetInfo.refreshRate;
            let hz = if r.Denominator != 0 {
                f64::from(r.Numerator) / f64::from(r.Denominator)
            } else {
                60.0
            };
            return Some((nits, hz));
        }
        None
    }
}

/// Tracks the current display and decides when to re-query. Owns its own
/// DXGI factory, replaced whenever it reports a stale adapter/output set.
#[derive(Debug)]
pub struct DisplayTracker {
    factory: Option<IDXGIFactory6>,
    info: DisplayInfo,
    dirty: bool,
}

impl Default for DisplayTracker {
    fn default() -> Self {
        Self {
            factory: None,
            info: DisplayInfo::fallback(),
            dirty: true,
        }
    }
}

impl DisplayTracker {
    pub fn info(&self) -> &DisplayInfo {
        &self.info
    }

    /// Requests a re-query (window moved/resized, display change).
    pub fn invalidate(&mut self) {
        self.dirty = true;
    }

    /// Refreshes if needed. Returns `true` when the info changed.
    pub fn update(&mut self, hwnd: HWND) -> bool {
        // SAFETY: plain query on a valid factory.
        let stale = self
            .factory
            .as_ref()
            .is_none_or(|f| !unsafe { f.IsCurrent() }.as_bool());
        if stale {
            // SAFETY: creating a factory has no preconditions.
            self.factory =
                unsafe { CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) }.ok();
            self.dirty = true;
        }
        if !self.dirty {
            return false;
        }
        self.dirty = false;
        match self.factory.as_ref().and_then(|f| query(f, hwnd)) {
            Some(info) if info != self.info => {
                tracing::info!(?info, "display changed");
                self.info = info;
                true
            }
            Some(_) => false,
            None => {
                tracing::warn!("display query failed; keeping last known state");
                false
            }
        }
    }
}
