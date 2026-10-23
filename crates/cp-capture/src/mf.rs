//! Thin Media Foundation helpers: COM/MF lifetime, devices, media types.

use cp_core::{PixelFormat, Rational, VideoMode};
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{
    COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree, CoUninitialize,
};
use windows::core::{GUID, Interface, PWSTR};

use crate::color_detect::DetectedColor;

/// Per-thread COM (MTA) + Media Foundation initialization.
#[derive(Debug)]
pub struct MfThread(());

impl MfThread {
    pub fn init() -> windows::core::Result<Self> {
        // SAFETY: balanced by Drop on the same thread.
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
            if let Err(e) = MFStartup(MF_VERSION, MFSTARTUP_FULL) {
                CoUninitialize();
                return Err(e);
            }
        }
        Ok(Self(()))
    }
}

impl Drop for MfThread {
    fn drop(&mut self) {
        // SAFETY: paired with the successful calls in `init`.
        unsafe {
            let _ = MFShutdown();
            CoUninitialize();
        }
    }
}

/// A video capture device as listed by Media Foundation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    /// Stable identifier (`\\?\usb#vid_...`), used to reopen the device.
    pub symbolic_link: String,
}

impl DeviceInfo {
    /// USB vendor id parsed from the symbolic link, if any.
    pub fn usb_vendor_id(&self) -> Option<u16> {
        let lower = self.symbolic_link.to_ascii_lowercase();
        let i = lower.find("vid_")?;
        u16::from_str_radix(lower.get(i + 4..i + 8)?, 16).ok()
    }

    /// PCI vendor id parsed from the symbolic link, if any.
    pub fn pci_vendor_id(&self) -> Option<u16> {
        let lower = self.symbolic_link.to_ascii_lowercase();
        let i = lower.find("ven_")?;
        u16::from_str_radix(lower.get(i + 4..i + 8)?, 16).ok()
    }
}

fn video_capture_attributes(capacity: u32) -> windows::core::Result<IMFAttributes> {
    let mut attrs = None;
    // SAFETY: out pointer is valid; attribute keys are static GUIDs.
    unsafe {
        MFCreateAttributes(&mut attrs, capacity)?;
        let attrs = attrs.ok_or_else(windows::core::Error::empty)?;
        attrs.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )?;
        Ok(attrs)
    }
}

fn get_string(attrs: &IMFAttributes, key: &GUID) -> windows::core::Result<String> {
    let mut p = PWSTR::null();
    let mut len = 0u32;
    // SAFETY: MF allocates the string with CoTaskMemAlloc; freed below.
    unsafe {
        attrs.GetAllocatedString(key, &mut p, &mut len)?;
        let s = p.to_string().unwrap_or_default();
        CoTaskMemFree(Some(p.0 as *const _));
        Ok(s)
    }
}

/// Lists video capture devices. Requires [`MfThread`] on the calling thread.
pub fn list_devices() -> windows::core::Result<Vec<DeviceInfo>> {
    let attrs = video_capture_attributes(1)?;
    let mut raw: *mut Option<IMFActivate> = std::ptr::null_mut();
    let mut count = 0u32;
    // SAFETY: MF returns a CoTaskMem array of `count` activates which we
    // take ownership of (dropping each element releases it).
    unsafe {
        MFEnumDeviceSources(&attrs, &mut raw, &mut count)?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            if let Some(act) = std::ptr::read(raw.add(i)) {
                let name = get_string(&act.cast()?, &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)
                    .unwrap_or_default();
                if let Ok(link) = get_string(
                    &act.cast()?,
                    &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                ) {
                    out.push(DeviceInfo {
                        name,
                        symbolic_link: link,
                    });
                }
            }
        }
        CoTaskMemFree(Some(raw as *const _));
        Ok(out)
    }
}

/// Opens a device by symbolic link with a bounded driver queue.
pub fn open_source(dev: &DeviceInfo, max_buffers: u32) -> windows::core::Result<IMFMediaSource> {
    let attrs = video_capture_attributes(3)?;
    let link: Vec<u16> = dev.symbolic_link.encode_utf16().chain(Some(0)).collect();
    // SAFETY: `link` outlives the SetString call (MF copies it).
    unsafe {
        attrs.SetString(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
            windows::core::PCWSTR(link.as_ptr()),
        )?;
        attrs.SetUINT32(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_MAX_BUFFERS,
            max_buffers,
        )?;
        MFCreateDeviceSource(&attrs)
    }
}

/// Source reader configured for minimum latency and no implicit conversion.
///
/// With `callback`, the reader runs in asynchronous mode.
pub fn create_reader(
    source: &IMFMediaSource,
    callback: Option<&IMFSourceReaderCallback>,
) -> windows::core::Result<IMFSourceReader> {
    let mut attrs = None;
    // SAFETY: plain attribute setup on a fresh store.
    unsafe {
        MFCreateAttributes(&mut attrs, 4)?;
        let attrs = attrs.ok_or_else(windows::core::Error::empty)?;
        if let Some(cb) = callback {
            attrs.SetUnknown(&MF_SOURCE_READER_ASYNC_CALLBACK, cb)?;
        }
        attrs.SetUINT32(&MF_LOW_LATENCY, 1)?;
        attrs.SetUINT32(&MF_READWRITE_DISABLE_CONVERTERS, 1)?;
        attrs.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 0)?;
        MFCreateSourceReaderFromMediaSource(source, &attrs)
    }
}

pub const VIDEO_STREAM: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;

pub fn pixel_format(subtype: &GUID) -> Option<PixelFormat> {
    match *subtype {
        s if s == MFVideoFormat_NV12 => Some(PixelFormat::Nv12),
        s if s == MFVideoFormat_P010 => Some(PixelFormat::P010),
        s if s == MFVideoFormat_YUY2 => Some(PixelFormat::Yuy2),
        s if s == MFVideoFormat_UYVY => Some(PixelFormat::Uyvy),
        s if s == MFVideoFormat_RGB32 => Some(PixelFormat::Bgra8),
        _ => None,
    }
}

/// Decodes a media type into a [`VideoMode`]; `None` for unsupported types.
pub fn video_mode(mt: &IMFMediaType) -> Option<VideoMode> {
    // SAFETY: attribute reads on a valid media type.
    unsafe {
        let format = pixel_format(&mt.GetGUID(&MF_MT_SUBTYPE).ok()?)?;
        let size = mt.GetUINT64(&MF_MT_FRAME_SIZE).ok()?;
        let rate = mt.GetUINT64(&MF_MT_FRAME_RATE).ok()?;
        Some(VideoMode {
            width: (size >> 32) as u32,
            height: size as u32,
            fps: Rational::new((rate >> 32) as u32, (rate as u32).max(1)),
            format,
        })
    }
}

pub fn detected_color(mt: &IMFMediaType) -> DetectedColor {
    // SAFETY: attribute reads on a valid media type.
    let get = |k: &GUID| unsafe { mt.GetUINT32(k).ok() };
    DetectedColor::from_mf(
        get(&MF_MT_TRANSFER_FUNCTION),
        get(&MF_MT_VIDEO_PRIMARIES),
        get(&MF_MT_YUV_MATRIX),
        get(&MF_MT_VIDEO_NOMINAL_RANGE),
    )
}

/// All native media types with their decoded modes, in driver order.
pub fn native_types(reader: &IMFSourceReader) -> Vec<(IMFMediaType, VideoMode)> {
    let mut out = Vec::new();
    for i in 0.. {
        // SAFETY: enumerating until MF_E_NO_MORE_TYPES.
        match unsafe { reader.GetNativeMediaType(VIDEO_STREAM, i) } {
            Ok(mt) => {
                if let Some(mode) = video_mode(&mt) {
                    out.push((mt, mode));
                }
            }
            Err(_) => break,
        }
    }
    out
}

/// Lists the distinct supported modes of a device (opens it briefly).
pub fn list_modes(dev: &DeviceInfo) -> windows::core::Result<Vec<VideoMode>> {
    let source = open_source(dev, 2)?;
    let reader = create_reader(&source, None);
    let modes = reader.map(|r| {
        let mut modes: Vec<VideoMode> = native_types(&r).into_iter().map(|(_, m)| m).collect();
        modes.dedup();
        modes
    });
    // SAFETY: the source is ours; shutting it down releases the device.
    unsafe {
        let _ = source.Shutdown();
    }
    modes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_ids_from_links() {
        let d = DeviceInfo {
            name: "x".into(),
            symbolic_link: r"\\?\usb#vid_07ca&pid_2553&mi_00#7&1ca8ca21&0&0000#{65e8773d}".into(),
        };
        assert_eq!(d.usb_vendor_id(), Some(0x07ca));
        assert_eq!(d.pci_vendor_id(), None);
    }
}
