//! Which display the window is on and what it can do (Linux).
//!
//! The luminances come from the Wayland color-management protocol: the
//! compositor's *preferred image description* for our surface
//! (`wp_color_management_surface_feedback_v1`), re-queried whenever it
//! announces a change (window moved to another output, HDR toggled, KDE's
//! brightness changed). The connector name and refresh rate come from the
//! window system via [`DisplayTracker::set_monitor`].
//!
//! Variable refresh is the compositor's decision. KWin's "Adaptive sync"
//! policy (never / automatic: for the active fullscreen window / always) is
//! published per output by `kde_output_device_v2` (listed by
//! `kde_output_device_registry_v2` on current KWin, one global per output
//! before), matched to our output by connector name; it seeds the refresh detection before measurements
//! decide. The surface is also tagged as game content (`wp_content_type_v1`),
//! which KWin forwards to the connector's content type: HDMI TVs switch to
//! their low-latency game mode for the fullscreen window. Tearing hints
//! (`wp_tearing_control_v1`) and FIFO/commit timing are the Vulkan driver's
//! business (it creates those objects for the IMMEDIATE / FIFO present
//! modes; a second one would be a protocol error).
//!
//! While the capture shows a picture the renderer keeps the session awake
//! with an idle inhibitor on the surface (`zwp_idle_inhibit_manager_v1`):
//! the compositor then neither blanks nor locks the screen while the window
//! is visible.
//!
//! The protocol objects live on their own event queue of the window's
//! `wl_display`. The window thread's event loop reads the socket and
//! libwayland routes our events to that queue; the render thread only
//! dispatches what is pending. It must never read the socket itself: a
//! second reader blocks in `wl_display_read_events` until the window thread
//! reads too, i.e. until the next window event. Without the protocols (X11,
//! other compositors) the SDR fallback is used and VRR is left to
//! measurement.

use std::ffi::c_void;
use std::ptr::NonNull;

use wayland_client::backend::{Backend, ObjectId};
use wayland_client::protocol::wl_registry::{self, WlRegistry};
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_client::{WEnum, event_created_child};
use wayland_protocols::wp::color_management::v1::client::{
    wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1},
    wp_color_manager_v1::{self, WpColorManagerV1},
    wp_image_description_info_v1::{self, WpImageDescriptionInfoV1},
    wp_image_description_v1::{self, WpImageDescriptionV1},
};
use wayland_protocols::wp::content_type::v1::client::{
    wp_content_type_manager_v1::{self, WpContentTypeManagerV1},
    wp_content_type_v1::{self, WpContentTypeV1},
};
use wayland_protocols::wp::idle_inhibit::zv1::client::{
    zwp_idle_inhibit_manager_v1::{self, ZwpIdleInhibitManagerV1},
    zwp_idle_inhibitor_v1::{self, ZwpIdleInhibitorV1},
};

use self::kde::{
    kde_output_device_mode_v2::{self, KdeOutputDeviceModeV2},
    kde_output_device_registry_v2::{self, KdeOutputDeviceRegistryV2},
    kde_output_device_v2::{self, KdeOutputDeviceV2},
};

use crate::common::CompositorVrr;
use crate::display_info::DisplayInfo;

/// KDE's output-device protocol (plasma-wayland-protocols, vendored: the
/// crates.io bindings predate the registry).
#[allow(clippy::all, missing_docs, unused_qualifications)]
mod kde {
    use wayland_client;

    pub mod __interfaces {
        wayland_scanner::generate_interfaces!("protocols/kde-output-device-v2.xml");
    }
    use self::__interfaces::*;

    wayland_scanner::generate_client_code!("protocols/kde-output-device-v2.xml");
}

/// Luminances of an image description, in cd/m².
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Luminances {
    pub min: f32,
    pub max: f32,
    pub reference: f32,
    /// Target (display) volume, when the compositor reports it.
    pub target_min: Option<f32>,
    pub target_max: Option<f32>,
    pub max_fall: Option<f32>,
}

impl Luminances {
    /// Display peak: the target volume if known, else the container's.
    pub fn peak(&self) -> f32 {
        self.target_max.unwrap_or(self.max)
    }

    /// HDR when the display can go clearly above the reference white.
    pub fn is_hdr(&self) -> bool {
        self.peak() > self.reference * 1.5
    }
}

/// One `kde_output_device_v2` (KWin).
struct OutputDevice {
    device: KdeOutputDeviceV2,
    /// Registry name, when the device is its own global (older KWin).
    global: Option<u32>,
    connector: Option<String>,
    vrr_capable: bool,
    policy: Option<kde_output_device_v2::VrrPolicy>,
}

impl std::fmt::Debug for OutputDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputDevice")
            .field("connector", &self.connector)
            .field("vrr", &self.vrr())
            .finish()
    }
}

impl OutputDevice {
    fn vrr(&self) -> Option<CompositorVrr> {
        use kde_output_device_v2::VrrPolicy as P;
        if !self.vrr_capable {
            return Some(CompositorVrr::Unsupported);
        }
        Some(match self.policy? {
            P::Never => CompositorVrr::Never,
            P::Always => CompositorVrr::Always,
            P::Automatic => CompositorVrr::Automatic,
        })
    }
}

/// Version of the vendored output-device protocol.
const OUTPUT_DEVICE_VERSION: u32 = 26;
/// The registry exists from this version on.
const OUTPUT_REGISTRY_MIN_VERSION: u32 = 21;

#[derive(Default)]
struct State {
    manager: Option<(u32, u32)>,
    content_type: Option<(u32, u32)>,
    idle_inhibit: Option<(u32, u32)>,
    outputs: Vec<OutputDevice>,
    partial: Luminances,
    /// Completed information of the latest preferred description.
    done: Option<Luminances>,
    /// The preferred description needs to be (re)queried.
    changed: bool,
    failed: bool,
    /// Description being queried; destroyed once its information is done.
    desc: Option<WpImageDescriptionV1>,
}

impl State {
    fn drop_desc(&mut self) {
        if let Some(d) = self.desc.take() {
            d.destroy();
        }
    }
}

impl Dispatch<WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } => {
                if interface == WpColorManagerV1::interface().name {
                    state.manager = Some((name, version));
                } else if interface == WpContentTypeManagerV1::interface().name {
                    state.content_type = Some((name, version));
                } else if interface == ZwpIdleInhibitManagerV1::interface().name {
                    state.idle_inhibit = Some((name, version));
                } else if interface == KdeOutputDeviceRegistryV2::interface().name
                    && version >= OUTPUT_REGISTRY_MIN_VERSION
                {
                    // Announces every output now and on hotplug.
                    let _: KdeOutputDeviceRegistryV2 =
                        registry.bind(name, version.min(OUTPUT_DEVICE_VERSION), qh, ());
                } else if interface == KdeOutputDeviceV2::interface().name {
                    // Older KWin: one global per output, also when
                    // hotplugged later.
                    let device: KdeOutputDeviceV2 =
                        registry.bind(name, version.min(OUTPUT_DEVICE_VERSION), qh, ());
                    state.add_output(device, Some(name));
                }
            }
            wl_registry::Event::GlobalRemove { name } => {
                state.outputs.retain(|o| o.global != Some(name));
            }
            _ => {}
        }
    }
}

impl State {
    fn add_output(&mut self, device: KdeOutputDeviceV2, global: Option<u32>) {
        self.outputs.push(OutputDevice {
            device,
            global,
            connector: None,
            vrr_capable: false,
            policy: None,
        });
    }
}

impl Dispatch<KdeOutputDeviceRegistryV2, ()> for State {
    fn event(
        state: &mut Self,
        _: &KdeOutputDeviceRegistryV2,
        event: kde_output_device_registry_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let kde_output_device_registry_v2::Event::Output { output } = event {
            state.add_output(output, None);
        }
    }

    event_created_child!(State, KdeOutputDeviceRegistryV2, [
        kde_output_device_registry_v2::EVT_OUTPUT_OPCODE => (KdeOutputDeviceV2, ()),
    ]);
}

impl Dispatch<KdeOutputDeviceV2, ()> for State {
    fn event(
        state: &mut Self,
        device: &KdeOutputDeviceV2,
        event: kde_output_device_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use kde_output_device_v2::Event as E;
        if let E::Removed = event {
            state.outputs.retain(|o| &o.device != device);
            device.release();
            return;
        }
        let Some(o) = state.outputs.iter_mut().find(|o| &o.device == device) else {
            return;
        };
        match event {
            E::Name { name } => o.connector = Some(name),
            E::Capabilities {
                flags: WEnum::Value(f),
            } => o.vrr_capable = f.contains(kde_output_device_v2::Capability::Vrr),
            E::VrrPolicy {
                vrr_policy: WEnum::Value(p),
            } => o.policy = Some(p),
            _ => {}
        }
    }

    event_created_child!(State, KdeOutputDeviceV2, [
        kde_output_device_v2::EVT_MODE_OPCODE => (KdeOutputDeviceModeV2, ()),
    ]);
}

impl Dispatch<KdeOutputDeviceModeV2, ()> for State {
    fn event(
        _: &mut Self,
        mode: &KdeOutputDeviceModeV2,
        event: kde_output_device_mode_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Mode list: not needed (the refresh comes from the window system).
        if let kde_output_device_mode_v2::Event::Removed = event
            && mode.version() >= kde_output_device_mode_v2::REQ_RELEASE_SINCE
        {
            mode.release();
        }
    }
}

impl Dispatch<WpContentTypeManagerV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &WpContentTypeManagerV1,
        _: wp_content_type_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwpIdleInhibitManagerV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &ZwpIdleInhibitManagerV1,
        _: zwp_idle_inhibit_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwpIdleInhibitorV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &ZwpIdleInhibitorV1,
        _: zwp_idle_inhibitor_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpContentTypeV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &WpContentTypeV1,
        _: wp_content_type_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpColorManagerV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &WpColorManagerV1,
        _: wp_color_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // Supported features/intents; nothing is created from them.
    }
}

impl Dispatch<WpColorManagementSurfaceFeedbackV1, ()> for State {
    fn event(
        state: &mut Self,
        _: &WpColorManagementSurfaceFeedbackV1,
        event: wp_color_management_surface_feedback_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // `preferred_changed` (v1) and `preferred_changed2` (v2) alike.
        let _ = event;
        state.changed = true;
    }
}

impl Dispatch<WpImageDescriptionV1, ()> for State {
    fn event(
        state: &mut Self,
        desc: &WpImageDescriptionV1,
        event: wp_image_description_v1::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wp_image_description_v1::Event::Failed { cause, msg } => {
                tracing::warn!(?cause, msg, "preferred image description failed");
                state.failed = true;
                state.drop_desc();
            }
            // `ready` (v1) / `ready2` (v2): the description can be queried.
            _ => {
                state.partial = Luminances::default();
                desc.get_information(qh, ());
            }
        }
    }
}

impl Dispatch<WpImageDescriptionInfoV1, ()> for State {
    fn event(
        state: &mut Self,
        _: &WpImageDescriptionInfoV1,
        event: wp_image_description_info_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use wp_image_description_info_v1::Event as E;
        let p = &mut state.partial;
        match event {
            E::Luminances {
                min_lum,
                max_lum,
                reference_lum,
            } => {
                p.min = min_lum as f32 / 10_000.0;
                p.max = max_lum as f32;
                p.reference = reference_lum as f32;
            }
            E::TargetLuminance { min_lum, max_lum } => {
                p.target_min = Some(min_lum as f32 / 10_000.0);
                p.target_max = Some(max_lum as f32);
            }
            E::TargetMaxFall { max_fall } if max_fall > 0 => {
                p.max_fall = Some(max_fall as f32);
            }
            E::Done => {
                state.done = Some(state.partial);
                state.drop_desc();
            }
            _ => {}
        }
    }
}

/// Our protocol objects on the window's Wayland connection: the surface's
/// preferred image description, its content type and the outputs' VRR
/// policies, kept up to date.
pub struct Compositor {
    conn: Connection,
    queue: EventQueue<State>,
    state: State,
    color: Option<(WpColorManagerV1, WpColorManagementSurfaceFeedbackV1)>,
    /// Kept alive: destroying it resets the content type.
    content_type: Option<(WpContentTypeManagerV1, WpContentTypeV1)>,
    surface: WlSurface,
    idle_manager: Option<ZwpIdleInhibitManagerV1>,
    /// Present while the session is kept awake.
    idle_inhibitor: Option<ZwpIdleInhibitorV1>,
}

impl std::fmt::Debug for Compositor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compositor")
            .field("color_managed", &self.color.is_some())
            .field("content_type", &self.content_type.is_some())
            .field("outputs", &self.state.outputs)
            .finish_non_exhaustive()
    }
}

impl Compositor {
    /// Binds the protocols on the window's display.
    ///
    /// # Safety
    /// `display` and `surface` must be the live `wl_display` / `wl_surface`
    /// of the window, and stay valid until this object is dropped.
    pub unsafe fn new(display: NonNull<c_void>, surface: NonNull<c_void>) -> Option<Self> {
        // SAFETY: caller guarantees a live wl_display.
        let backend = unsafe { Backend::from_foreign_display(display.as_ptr().cast()) };
        let conn = Connection::from_backend(backend);
        let mut queue = conn.new_event_queue();
        let qh = queue.handle();
        let mut state = State::default();
        let registry = conn.display().get_registry(&qh, ());
        queue.roundtrip(&mut state).ok()?;
        // SAFETY: caller guarantees a live wl_surface.
        let id =
            unsafe { ObjectId::from_ptr(WlSurface::interface(), surface.as_ptr().cast()) }.ok()?;
        let surface = WlSurface::from_id(&conn, id).ok()?;

        let color = state.manager.map(|(name, version)| {
            let manager: WpColorManagerV1 = registry.bind(
                name,
                version.min(WpColorManagerV1::interface().version),
                &qh,
                (),
            );
            let feedback = manager.get_surface_feedback(&surface, &qh, ());
            (manager, feedback)
        });
        if color.is_none() {
            tracing::info!("compositor has no wp_color_manager_v1; assuming SDR");
        }
        let content_type = state.content_type.map(|(name, version)| {
            let manager: WpContentTypeManagerV1 = registry.bind(name, version.min(1), &qh, ());
            let ct = manager.get_surface_content_type(&surface, &qh, ());
            ct.set_content_type(wp_content_type_v1::Type::Game);
            (manager, ct)
        });
        let idle_manager = state.idle_inhibit.map(|(name, version)| {
            registry.bind::<ZwpIdleInhibitManagerV1, _, _>(name, version.min(1), &qh, ())
        });
        let mut c = Self {
            conn,
            queue,
            state,
            color,
            content_type,
            surface,
            idle_manager,
            idle_inhibitor: None,
        };
        c.request();
        // Output-device properties; ready → get_information → done: a few
        // roundtrips at most.
        for _ in 0..4 {
            let colors_done = c.color.is_none() || c.state.done.is_some() || c.state.failed;
            let outputs_done = c.state.outputs.iter().all(|o| o.policy.is_some());
            if colors_done && outputs_done {
                break;
            }
            c.queue.roundtrip(&mut c.state).ok()?;
        }
        tracing::info!(compositor = ?c, "wayland display protocols");
        Some(c)
    }

    fn request(&mut self) {
        self.state.changed = false;
        self.state.failed = false;
        self.state.drop_desc();
        if let Some((_, feedback)) = &self.color {
            self.state.desc = Some(feedback.get_preferred(&self.queue.handle(), ()));
        }
    }

    /// Processes events the window thread already read (never blocks);
    /// returns new luminances once a re-queried description completed.
    pub fn poll(&mut self) -> Option<Luminances> {
        let _ = self.conn.flush();
        let _ = self.queue.dispatch_pending(&mut self.state);
        if self.state.changed {
            self.request();
            let _ = self.conn.flush();
        }
        self.state.done.take()
    }

    /// Keeps the session from idling (screen blanking, locking) while `on`
    /// and the window is visible. False when the compositor cannot.
    fn set_idle_inhibit(&mut self, on: bool) -> bool {
        let Some(manager) = &self.idle_manager else {
            return false;
        };
        match (on, self.idle_inhibitor.take()) {
            (true, None) => {
                let qh = self.queue.handle();
                self.idle_inhibitor = Some(manager.create_inhibitor(&self.surface, &qh, ()));
            }
            (true, Some(i)) => self.idle_inhibitor = Some(i),
            (false, Some(i)) => i.destroy(),
            (false, None) => {}
        }
        let _ = self.conn.flush();
        true
    }

    /// The compositor's VRR policy for the output named `connector`.
    fn vrr(&self, connector: &str) -> Option<CompositorVrr> {
        self.state
            .outputs
            .iter()
            .find(|o| o.connector.as_deref() == Some(connector))
            .and_then(OutputDevice::vrr)
    }
}

impl Drop for Compositor {
    fn drop(&mut self) {
        self.state.drop_desc();
        if let Some((manager, feedback)) = self.color.take() {
            feedback.destroy();
            manager.destroy();
        }
        if let Some(i) = self.idle_inhibitor.take() {
            i.destroy();
        }
        if let Some(m) = self.idle_manager.take() {
            m.destroy();
        }
        if let Some((manager, ct)) = self.content_type.take() {
            ct.destroy();
            manager.destroy();
        }
        let _ = self.conn.flush();
    }
}

/// Tracks the current display; see the module docs.
#[derive(Debug)]
pub struct DisplayTracker {
    compositor: Option<Compositor>,
    luminances: Option<Luminances>,
    monitor: Option<(String, f64)>,
    info: DisplayInfo,
    vrr: Option<CompositorVrr>,
    inhibit_warned: bool,
}

impl DisplayTracker {
    pub fn new(compositor: Option<Compositor>) -> Self {
        let mut t = Self {
            compositor,
            luminances: None,
            monitor: None,
            info: DisplayInfo::fallback(),
            vrr: None,
            inhibit_warned: false,
        };
        if let Some(l) = t.compositor.as_mut().and_then(|c| c.state.done.take()) {
            t.luminances = Some(l);
        }
        t.info = t.compose();
        tracing::info!(info = ?t.info, luminances = ?t.luminances, "display");
        t
    }

    /// The compositor's variable-refresh policy for our output, when it
    /// publishes one (KWin).
    /// Keeps the session awake while `on` (see the module docs).
    pub fn set_idle_inhibit(&mut self, on: bool) {
        let ok = self
            .compositor
            .as_mut()
            .is_some_and(|c| c.set_idle_inhibit(on));
        if on && !ok && !self.inhibit_warned {
            self.inhibit_warned = true;
            tracing::warn!(
                "the compositor cannot inhibit idling; the screen may lock while playing"
            );
        }
    }

    pub fn vrr(&self) -> Option<CompositorVrr> {
        self.vrr
    }

    pub fn info(&self) -> &DisplayInfo {
        &self.info
    }

    /// Whether the compositor color-manages our output.
    pub fn color_managed(&self) -> bool {
        self.luminances.is_some()
    }

    /// Connector name and refresh rate of the output under the window.
    pub fn set_monitor(&mut self, name: String, refresh_hz: f64) {
        self.monitor = Some((name, refresh_hz));
    }

    fn compose(&self) -> DisplayInfo {
        let mut info = match self.luminances {
            Some(l) => DisplayInfo {
                device_name: String::new(),
                hdr: l.is_hdr(),
                max_nits: l.peak(),
                min_nits: l.target_min.unwrap_or(l.min),
                max_full_frame_nits: l.max_fall.unwrap_or(l.peak()),
                sdr_white_nits: l.reference,
                refresh_hz: 60.0,
            },
            None => DisplayInfo::fallback(),
        };
        if let Some((name, hz)) = &self.monitor {
            info.device_name.clone_from(name);
            if *hz > 1.0 {
                info.refresh_hz = *hz;
            }
        }
        info
    }

    /// Processes compositor updates. Returns `true` when the display itself
    /// changed (another output, mode, HDR or VRR policy), not merely its
    /// reference luminance.
    pub fn update(&mut self) -> bool {
        if let Some(l) = self.compositor.as_mut().and_then(Compositor::poll) {
            self.luminances = Some(l);
        }
        let vrr = match (&self.compositor, &self.monitor) {
            (Some(c), Some((name, _))) => c.vrr(name),
            _ => None,
        };
        let vrr_changed = vrr != self.vrr;
        if vrr_changed {
            tracing::info!(?vrr, "compositor VRR policy");
            self.vrr = vrr;
        }
        let info = self.compose();
        if info == self.info {
            return vrr_changed;
        }
        // KDE animates the reference luminance (brightness slider, idle
        // dimming); that is not a display change.
        let reference_only = DisplayInfo {
            sdr_white_nits: self.info.sdr_white_nits,
            ..info.clone()
        } == self.info;
        if reference_only {
            tracing::debug!(
                sdr_white_nits = info.sdr_white_nits,
                "display reference changed"
            );
        } else {
            tracing::info!(?info, "display changed");
        }
        self.info = info;
        vrr_changed || !reference_only
    }
}

#[cfg(test)]
mod tests {
    use super::Luminances;

    #[test]
    fn kwin_hdr_and_sdr_outputs() {
        // KWin, HDR on: reference = SDR brightness × slider.
        let hdr = Luminances {
            min: 0.0,
            max: 1020.0,
            reference: 194.0,
            target_min: Some(0.005),
            target_max: Some(1020.0),
            max_fall: Some(315.0),
        };
        assert!(hdr.is_hdr());
        assert_eq!(hdr.peak(), 1020.0);
        let sdr = Luminances {
            max: 203.0,
            reference: 203.0,
            ..Default::default()
        };
        assert!(!sdr.is_hdr());
    }
}
