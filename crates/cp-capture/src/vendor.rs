//! Vendor-specific capture-card controls (kernel-streaming properties).

use windows::Win32::Media::KernelStreaming::{
    IKsControl, KSIDENTIFIER, KSPROPERTY_TYPE_GET, KSPROPERTY_TYPE_SET,
};
use windows::Win32::Media::MediaFoundation::IMFMediaSource;
use windows::core::{GUID, Interface};

use crate::mf::DeviceInfo;

const AVERMEDIA_USB_VID: u16 = 0x07ca;
const AVERMEDIA_PCI_VEN: u16 = 0x1461;

/// AVerMedia hardware HDR→SDR tonemapper switch (property id 2). Same
/// property OBS (libdshowcapture `SetTonemapperAvermedia`) uses.
const AVERMEDIA_TONEMAP_SET: GUID = GUID::from_u128(0x8a80d56f_fac5_4692_a416_cf20d4a18f47);
const AVERMEDIA_TONEMAP_ID: u32 = 2;

fn ks_identifier(set: GUID, id: u32, flags: u32) -> KSIDENTIFIER {
    let mut k = KSIDENTIFIER::default();
    k.Anonymous.Anonymous.Set = set;
    k.Anonymous.Anonymous.Id = id;
    k.Anonymous.Anonymous.Flags = flags;
    k
}

/// Outcome of a hardware-tonemap request, for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapState {
    NotApplicable,
    /// Device confirmed the requested state (`true` = tonemapper on).
    Confirmed(bool),
    /// Set or readback failed; the driver may keep another app's setting.
    Unconfirmed,
}

/// For 10-bit capture, HDR must reach us untouched, so the card's own
/// tonemapper is switched off. 8-bit modes are left alone: forcing it on
/// can drop some cards to a slow path.
pub fn configure_hdr_passthrough(
    dev: &DeviceInfo,
    source: &IMFMediaSource,
    ten_bit: bool,
) -> TonemapState {
    let is_avermedia = dev.usb_vendor_id() == Some(AVERMEDIA_USB_VID)
        || dev.pci_vendor_id() == Some(AVERMEDIA_PCI_VEN);
    if !is_avermedia || !ten_bit {
        return TonemapState::NotApplicable;
    }
    let Ok(ks) = source.cast::<IKsControl>() else {
        return TonemapState::Unconfirmed;
    };
    // Same wire layout as OBS's IKsPropertySet::Set call: the property
    // header carries the value as instance data, and the data buffer is a
    // `{KSPROPERTY, DWORD}` struct.
    #[repr(C)]
    struct Payload {
        prop: KSIDENTIFIER,
        value: u32,
    }
    let size = size_of::<Payload>() as u32;
    let mut returned = 0u32;
    let set = Payload {
        prop: ks_identifier(
            AVERMEDIA_TONEMAP_SET,
            AVERMEDIA_TONEMAP_ID,
            KSPROPERTY_TYPE_SET,
        ),
        value: 0,
    };
    let mut data = Payload {
        prop: KSIDENTIFIER::default(),
        value: 0,
    };
    // SAFETY: both buffers are `size` bytes and live across the calls.
    unsafe {
        let set_result =
            ks.KsProperty(&set.prop, size, (&raw mut data).cast(), size, &mut returned);
        let get = Payload {
            prop: ks_identifier(
                AVERMEDIA_TONEMAP_SET,
                AVERMEDIA_TONEMAP_ID,
                KSPROPERTY_TYPE_GET,
            ),
            value: 0,
        };
        let mut actual = Payload {
            prop: KSIDENTIFIER::default(),
            value: u32::MAX,
        };
        let get_result = ks.KsProperty(
            &get.prop,
            size,
            (&raw mut actual).cast(),
            size,
            &mut returned,
        );
        tracing::debug!(
            ?set_result,
            ?get_result,
            returned,
            value = actual.value,
            "AVerMedia tonemap property"
        );
        // ERROR_SET_NOT_FOUND: this model has no hardware tonemapper (e.g. USB
        // Live Gamer Ultra 2.1); it passes HDR through as-is.
        const SET_NOT_FOUND: windows::core::HRESULT =
            windows::core::HRESULT(0x8007_0492_u32 as i32);
        if set_result
            .as_ref()
            .err()
            .is_some_and(|e| e.code() == SET_NOT_FOUND)
        {
            return TonemapState::NotApplicable;
        }
        match get_result {
            Ok(()) if actual.value <= 1 => TonemapState::Confirmed(actual.value == 1),
            _ if set_result.is_ok() => TonemapState::Confirmed(false),
            _ => TonemapState::Unconfirmed,
        }
    }
}
