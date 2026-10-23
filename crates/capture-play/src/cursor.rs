//! Mouse cursor auto-hide policy: hide after a short idle period over the
//! video, show again on real movement. Pure state machine; the window thread
//! applies its decisions.

use std::time::{Duration, Instant};

/// Idle time before the cursor disappears.
pub const IDLE: Duration = Duration::from_secs(2);

/// Movement below this (physical px) is ignored: Windows re-sends mouse-move
/// messages when the cursor is hidden or the window under it changes, which
/// would otherwise bring the cursor straight back.
const JITTER_PX: f64 = 2.0;

#[derive(Debug)]
pub struct CursorIdle {
    last_activity: Instant,
    last_pos: Option<(f64, f64)>,
    hidden: bool,
}

impl CursorIdle {
    pub fn new(now: Instant) -> Self {
        Self {
            last_activity: now,
            last_pos: None,
            hidden: false,
        }
    }

    pub fn hidden(&self) -> bool {
        self.hidden
    }

    /// Pointer moved to `pos`. Returns `true` if the cursor must be shown.
    pub fn moved(&mut self, now: Instant, pos: (f64, f64)) -> bool {
        let real = self
            .last_pos
            .is_none_or(|(x, y)| (pos.0 - x).hypot(pos.1 - y) >= JITTER_PX);
        self.last_pos = Some(pos);
        if real { self.activity(now) } else { false }
    }

    /// Click, wheel or key. Returns `true` if the cursor must be shown.
    pub fn activity(&mut self, now: Instant) -> bool {
        self.last_activity = now;
        std::mem::take(&mut self.hidden)
    }

    /// Periodic check. Returns `Some(deadline)` to wake up at, and sets
    /// `hidden` when the cursor should disappear now. Never hides while the
    /// pointer is over the UI.
    pub fn poll(&mut self, now: Instant, over_ui: bool) -> (bool, Option<Instant>) {
        if self.hidden {
            return (false, None);
        }
        if over_ui {
            self.last_activity = now;
        }
        let deadline = self.last_activity + IDLE;
        if now >= deadline {
            self.hidden = true;
            (true, None)
        } else {
            (false, Some(deadline))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hides_after_idle_and_shows_on_movement() {
        let t0 = Instant::now();
        let mut c = CursorIdle::new(t0);
        assert!(!c.moved(t0, (100.0, 100.0)));
        assert_eq!(
            c.poll(t0 + Duration::from_millis(500), false),
            (false, Some(t0 + IDLE))
        );
        assert_eq!(c.poll(t0 + IDLE, false), (true, None));
        assert!(c.hidden());
        // Re-sent mouse-move messages at the same spot do not bring it back.
        assert!(!c.moved(t0 + IDLE, (100.0, 100.0)));
        assert!(!c.moved(t0 + IDLE, (100.5, 100.0)));
        assert!(c.hidden());
        // Real movement does.
        assert!(c.moved(t0 + IDLE * 2, (140.0, 100.0)));
        assert!(!c.hidden());
    }

    #[test]
    fn never_hides_over_ui() {
        let t0 = Instant::now();
        let mut c = CursorIdle::new(t0);
        let (hide, deadline) = c.poll(t0 + IDLE * 3, true);
        assert!(!hide);
        assert_eq!(deadline, Some(t0 + IDLE * 4));
    }

    #[test]
    fn click_shows_hidden_cursor() {
        let t0 = Instant::now();
        let mut c = CursorIdle::new(t0);
        c.poll(t0 + IDLE, false);
        assert!(c.activity(t0 + IDLE * 2));
        assert!(!c.activity(t0 + IDLE * 2));
    }
}
