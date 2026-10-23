//! Audio endpoint enumeration and capture-card matching.

use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Media::Audio::*;
use windows::Win32::System::Com::StructuredStorage::PropVariantToStringAlloc;
use windows::Win32::System::Com::{CLSCTX_ALL, CoCreateInstance, CoTaskMemFree, STGM_READ};
use windows::core::{HSTRING, PWSTR};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub id: String,
    pub name: String,
}

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

/// Normalized product name of a capture device, e.g.
/// `"Live Gamer Ultra 2.1-Video"` → `"live gamer ultra 2.1"`.
fn product_key(name: &str) -> String {
    let lower = name.to_lowercase();
    let trimmed = lower.trim_end_matches(|c: char| c == ')' || c.is_whitespace());
    let base = ["-video", " video", "-audio", " audio"]
        .iter()
        .fold(trimmed.to_string(), |acc, suffix| {
            acc.trim_end_matches(suffix).to_string()
        });
    base.trim().to_string()
}

/// Finds the capture endpoint belonging to a video capture device by name.
/// Never falls back to an unrelated input (that would play a microphone).
pub fn match_capture_endpoint<'a>(
    video_name: &str,
    endpoints: &'a [Endpoint],
) -> Option<&'a Endpoint> {
    let key = product_key(video_name);
    if key.len() < 3 {
        return None;
    }
    endpoints.iter().find(|e| {
        let n = e.name.to_lowercase();
        // Endpoint names look like "HDMI/Line In (Live Gamer Ultra 2.1-Audio)".
        let inner = n.rsplit_once('(').map_or(n.as_str(), |(_, r)| r);
        product_key(inner) == key || n.contains(&key)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(name: &str) -> Endpoint {
        Endpoint {
            id: name.into(),
            name: name.into(),
        }
    }

    #[test]
    fn matches_avermedia_and_elgato_naming() {
        let eps = [
            ep("麦克风 (Steam Streaming Microphone)"),
            ep("HDMI/Line In (Live Gamer Ultra 2.1-Audio)"),
            ep("Game Capture 4K X Audio (Elgato)"),
        ];
        assert_eq!(
            match_capture_endpoint("Live Gamer Ultra 2.1-Video", &eps).map(|e| e.name.as_str()),
            Some("HDMI/Line In (Live Gamer Ultra 2.1-Audio)")
        );
        assert_eq!(
            match_capture_endpoint("Game Capture 4K X", &eps).map(|e| e.name.as_str()),
            Some("Game Capture 4K X Audio (Elgato)")
        );
        assert_eq!(match_capture_endpoint("OBS Virtual Camera", &eps), None);
    }
}
