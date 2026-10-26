//! What the display under the window can do, as reported by the platform
//! (DXGI + DisplayConfig on Windows, the Wayland color-management protocol
//! on Linux).

#[derive(Debug, Clone, PartialEq)]
pub struct DisplayInfo {
    /// Platform name of the output (GDI `\\.\DISPLAY1`, Wayland connector).
    pub device_name: String,
    /// HDR is enabled for this display (Windows "Use HDR", KDE "HDR").
    pub hdr: bool,
    pub max_nits: f32,
    pub min_nits: f32,
    pub max_full_frame_nits: f32,
    /// SDR content brightness in nits (Windows "SDR content brightness",
    /// the compositor's reference luminance on Wayland).
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
