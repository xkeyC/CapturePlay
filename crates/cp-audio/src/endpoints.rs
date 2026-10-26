//! WASAPI endpoint enumeration (capture-card matching: [`crate::matching`]).

use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Media::Audio::*;
use windows::Win32::System::Com::StructuredStorage::PropVariantToStringAlloc;
use windows::Win32::System::Com::{CLSCTX_ALL, CoCreateInstance, CoTaskMemFree, STGM_READ};
use windows::core::{HSTRING, PWSTR};

pub use crate::matching::Endpoint;

fn take_pwstr(p: PWSTR) -> String {
    // SAFETY: COM-allocated string, freed right after copying.
    unsafe {
        let s = p.to_string().unwrap_or_default();
        CoTaskMemFree(Some(p.0 as *const _));
        s
    }
}

/// Lists active endpoints. Requires COM on the calling thread.
pub fn list(capture: bool) -> windows::core::Result<Vec<Endpoint>> {
    // SAFETY: standard MMDevice enumeration.
    unsafe {
        let e: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let flow = if capture { eCapture } else { eRender };
        let coll = e.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE)?;
        let mut out = Vec::new();
        for i in 0..coll.GetCount()? {
            let dev = coll.Item(i)?;
            let id = take_pwstr(dev.GetId()?);
            let props = dev.OpenPropertyStore(STGM_READ)?;
            let name = props
                .GetValue(&PKEY_Device_FriendlyName)
                .and_then(|v| PropVariantToStringAlloc(&v))
                .map(take_pwstr)
                .unwrap_or_default();
            out.push(Endpoint { id, name });
        }
        Ok(out)
    }
}

pub(crate) fn open_device(id: &str) -> windows::core::Result<IMMDevice> {
    // SAFETY: standard MMDevice lookup.
    unsafe {
        let e: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        e.GetDevice(&HSTRING::from(id))
    }
}
