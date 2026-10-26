//! Which display the window is on and what it can do (Linux).
//!
//! The luminances come from the Wayland color-management protocol: the
//! compositor's *preferred image description* for our surface
//! (`wp_color_management_surface_feedback_v1`), re-queried whenever it
//! announces a change (window moved to another output, HDR toggled, KDE's
//! brightness changed). The connector name and refresh rate come from the
//! window system via [`DisplayTracker::set_monitor`].
//!
//! The protocol objects live on their own event queue of the window's
//! `wl_display`. The window thread's event loop reads the socket and
//! libwayland routes our events to that queue; the render thread only
//! dispatches what is pending. It must never read the socket itself: a
//! second reader blocks in `wl_display_read_events` until the window thread
//! reads too, i.e. until the next window event. Without the protocol (X11,
//! other compositors) the SDR fallback is used.

use std::ffi::c_void;
use std::ptr::NonNull;

use wayland_client::backend::{Backend, ObjectId};
use wayland_client::protocol::wl_registry::{self, WlRegistry};
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols::wp::color_management::v1::client::{
    wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1},
    wp_color_manager_v1::{self, WpColorManagerV1},
    wp_image_description_info_v1::{self, WpImageDescriptionInfoV1},
    wp_image_description_v1::{self, WpImageDescriptionV1},
};

use crate::display_info::DisplayInfo;

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

#[derive(Default)]
struct State {
    manager: Option<(u32, u32)>,
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
        _: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
            && interface == WpColorManagerV1::interface().name
        {
            state.manager = Some((name, version));
        }
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

/// The surface's preferred image description, kept up to date.
pub struct ColorManagement {
    conn: Connection,
    queue: EventQueue<State>,
    state: State,
    manager: WpColorManagerV1,
    feedback: WpColorManagementSurfaceFeedbackV1,
}

impl std::fmt::Debug for ColorManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ColorManagement")
    }
}

impl ColorManagement {
    /// Binds the color manager on the window's display.
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
        let Some((name, version)) = state.manager else {
            tracing::info!("compositor has no wp_color_manager_v1; assuming SDR");
            return None;
        };
        let manager: WpColorManagerV1 = registry.bind(
            name,
            version.min(WpColorManagerV1::interface().version),
            &qh,
            (),
        );
        // SAFETY: caller guarantees a live wl_surface.
        let id =
            unsafe { ObjectId::from_ptr(WlSurface::interface(), surface.as_ptr().cast()) }.ok()?;
        let surface = WlSurface::from_id(&conn, id).ok()?;
        let feedback = manager.get_surface_feedback(&surface, &qh, ());
        let mut cm = Self {
            conn,
            queue,
            state,
            manager,
            feedback,
        };
        cm.request();
        // ready → get_information → done: a few roundtrips at most.
        for _ in 0..4 {
            if cm.state.done.is_some() || cm.state.failed {
                break;
            }
            cm.queue.roundtrip(&mut cm.state).ok()?;
        }
        Some(cm)
    }

    fn request(&mut self) {
        self.state.changed = false;
        self.state.failed = false;
        self.state.drop_desc();
        self.state.desc = Some(self.feedback.get_preferred(&self.queue.handle(), ()));
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
}

impl Drop for ColorManagement {
    fn drop(&mut self) {
        self.state.drop_desc();
        self.feedback.destroy();
        self.manager.destroy();
        let _ = self.conn.flush();
    }
}

/// Tracks the current display; see the module docs.
#[derive(Debug)]
pub struct DisplayTracker {
    cm: Option<ColorManagement>,
    luminances: Option<Luminances>,
    monitor: Option<(String, f64)>,
    info: DisplayInfo,
}

impl DisplayTracker {
    pub fn new(cm: Option<ColorManagement>) -> Self {
        let mut t = Self {
            cm,
            luminances: None,
            monitor: None,
            info: DisplayInfo::fallback(),
        };
        if let Some(l) = t.cm.as_mut().and_then(|c| c.state.done.take()) {
            t.luminances = Some(l);
        }
        t.info = t.compose();
        tracing::info!(info = ?t.info, luminances = ?t.luminances, "display");
        t
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

    /// Processes compositor updates. Returns `true` when the info changed.
    pub fn update(&mut self) -> bool {
        if let Some(l) = self.cm.as_mut().and_then(ColorManagement::poll) {
            self.luminances = Some(l);
        }
        let info = self.compose();
        if info == self.info {
            return false;
        }
        // KDE animates the reference luminance (brightness slider, idle
        // dimming); only report other changes loudly.
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
        true
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
