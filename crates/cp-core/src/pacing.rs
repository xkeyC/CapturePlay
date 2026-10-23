//! Event-driven presentation planning for frame interpolation.
//!
//! # Latency model
//!
//! Interpolating between unique frames `A` and `B` requires `B` to exist, so
//! some delay is unavoidable. The planner minimises it: the moment `B`
//! arrives, the `M-1` intermediate frames are presented back to back at
//! `interval / M` spacing and `B` itself follows last. For 30 → 60 fps the
//! added latency is therefore **half a source frame (16.7 ms) plus GPU time**
//! — the theoretical minimum for interpolation. Engine-external DLSS-FG
//! typically queues one or more extra frames on top of that.
//!
//! # Smoothness
//!
//! Slots are scheduled on the *capture* clock (device timestamps mapped to the
//! host clock), not on each frame's arrival, so USB/driver delivery jitter
//! does not reach the screen. A [`DelayEstimator`] supplies the processing
//! delay budget (a high percentile of recent frames).
//!
//! Target times are *display* times. On a variable-refresh display a frame
//! appears when it is presented, so any multiplier works and the render loop
//! presents exactly on target. On a fixed-refresh display targets are snapped
//! to the vblank grid and the multiplier is restricted to values whose step
//! is a whole number of refreshes (3× on 240 Hz would alternate 2 and 3
//! refreshes; 2× or 4× are even).
//!
//! The planner is pure: it never sleeps and never touches the GPU.

use std::collections::VecDeque;

/// A unique source frame known to the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRef {
    pub id: u64,
    /// Capture timestamp in ns (host clock).
    pub time_ns: i64,
}

/// What to put on screen for one presentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Slot {
    Real { id: u64 },
    Interp { prev: u64, next: u64, t: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Planned {
    pub slot: Slot,
    /// Host time (ns) this slot should appear on screen.
    pub target_ns: i64,
}

/// How the display turns presents into refreshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refresh {
    /// Variable refresh (G-SYNC / FreeSync / HDMI VRR): a frame appears when
    /// presented, no faster than `min_period_ns`.
    Variable { min_period_ns: i64 },
    /// Fixed refresh: frames appear on vblanks `phase_ns + k · period_ns`.
    Fixed { period_ns: i64, phase_ns: i64 },
}

impl Refresh {
    /// Shortest time between two refreshes.
    pub fn period_ns(self) -> i64 {
        match self {
            Self::Variable { min_period_ns } => min_period_ns,
            Self::Fixed { period_ns, .. } => period_ns,
        }
        .max(1)
    }

    /// Earliest time `>= t` a frame can appear (a small tolerance keeps a
    /// target that sits on a vblank from moving to the next one).
    pub fn snap_up(self, t: i64) -> i64 {
        match self {
            Self::Variable { .. } => t,
            Self::Fixed {
                period_ns,
                phase_ns,
            } => {
                let p = period_ns.max(1);
                let k = (t - phase_ns - p / 20).div_euclid(p) + 1;
                phase_ns + k * p
            }
        }
    }
}

/// Delay from a frame's capture time until it is ready to be shown, as the
/// 95th percentile of recent frames.
#[derive(Debug, Default)]
pub struct DelayEstimator {
    samples: VecDeque<i64>,
}

impl DelayEstimator {
    const WINDOW: usize = 90;
    const MIN_SAMPLES: usize = 8;

    pub fn push(&mut self, delay_ns: i64) {
        if self.samples.len() == Self::WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(delay_ns);
    }

    pub fn reset(&mut self) {
        self.samples.clear();
    }

    pub fn estimate(&self) -> Option<i64> {
        if self.samples.len() < Self::MIN_SAMPLES {
            return None;
        }
        let mut v: Vec<i64> = self.samples.iter().copied().collect();
        v.sort_unstable();
        Some(v[(v.len() * 95 / 100).min(v.len() - 1)])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PacingConfig {
    /// Interpolation enabled at all.
    pub interpolate: bool,
    /// Highest number of output frames per source frame.
    pub max_multiplier: u32,
    /// Output frame-rate cap (e.g. 60 for strict 30 → 60). `None` = display rate.
    pub output_fps_cap: Option<f64>,
    /// Gaps longer than `content_interval * gap_factor` are discontinuities
    /// (menus, stalls, scene loads) and are not interpolated across.
    pub gap_factor: f64,
}

impl Default for PacingConfig {
    fn default() -> Self {
        Self {
            interpolate: true,
            max_multiplier: 4,
            output_fps_cap: None,
            gap_factor: 1.75,
        }
    }
}

#[derive(Debug, Default)]
pub struct Pacer {
    cfg: PacingConfig,
    prev: Option<FrameRef>,
    queue: VecDeque<Planned>,
}

impl Pacer {
    pub fn new(cfg: PacingConfig) -> Self {
        Self {
            cfg,
            ..Default::default()
        }
    }

    pub fn config(&self) -> &PacingConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: PacingConfig) {
        self.cfg = cfg;
    }

    /// Forgets history (source change, seek, scene cut).
    pub fn reset(&mut self) {
        self.prev = None;
        self.queue.clear();
    }

    /// Number of output frames to produce per source interval.
    pub fn multiplier(&self, interval_ns: i64, refresh: Refresh) -> u32 {
        if !self.cfg.interpolate || interval_ns <= 0 {
            return 1;
        }
        let cap_period = self
            .cfg
            .output_fps_cap
            .map_or(0.0, |fps| 1e9 / fps.max(1.0));
        let display = refresh.period_ns() as f64;
        let period = display.max(cap_period);
        let wanted =
            ((interval_ns as f64 / period).round() as u32).clamp(1, self.cfg.max_multiplier.max(1));
        if let Refresh::Fixed { .. } = refresh {
            // Largest multiplier whose step is a whole number of refreshes.
            let even = (1..=wanted).rev().find(|&m| {
                let refreshes = interval_ns as f64 / f64::from(m) / display;
                refreshes >= 0.95 && (refreshes - refreshes.round()).abs() < 0.08
            });
            if let Some(m) = even {
                return m;
            }
        }
        wanted
    }

    /// Schedules presentation for a newly arrived unique frame.
    ///
    /// `base_ns` is the earliest display time of the first slot (capture time
    /// plus the processing budget). `content_interval_ns` is the smoothed
    /// source frame interval (from [`crate::cadence::CadenceTracker`]); it
    /// guards against interpolating across stalls.
    pub fn on_unique_frame(
        &mut self,
        frame: FrameRef,
        base_ns: i64,
        refresh: Refresh,
        content_interval_ns: Option<i64>,
    ) {
        let base_ns = refresh.snap_up(base_ns);
        // Slots of the previous pair scheduled before this pair starts still
        // play out (the schedule runs a processing budget behind arrival);
        // anything later — or too close for the display to refresh in
        // between — would only add latency or delay this pair.
        let keep_before = base_ns - refresh.period_ns() + 1;
        self.queue.retain(|p| p.target_ns < keep_before);
        let prev = self.prev.replace(frame);

        let Some(prev) = prev else {
            self.queue.push_back(Planned {
                slot: Slot::Real { id: frame.id },
                target_ns: base_ns,
            });
            return;
        };
        let interval = frame.time_ns - prev.time_ns;
        let max_gap = content_interval_ns.map_or(100_000_000.0, |c| c as f64 * self.cfg.gap_factor);
        let m = if interval as f64 > max_gap {
            1
        } else {
            self.multiplier(interval, refresh)
        };

        let step = match refresh {
            Refresh::Variable { .. } => interval / i64::from(m),
            Refresh::Fixed { period_ns, .. } => {
                let p = period_ns.max(1);
                ((interval / i64::from(m) + p / 2) / p).max(1) * p
            }
        };
        for k in 1..m {
            let t = k as f32 / m as f32;
            self.queue.push_back(Planned {
                slot: Slot::Interp {
                    prev: prev.id,
                    next: frame.id,
                    t,
                },
                target_ns: base_ns + i64::from(k - 1) * step,
            });
        }
        self.queue.push_back(Planned {
            slot: Slot::Real { id: frame.id },
            target_ns: base_ns + i64::from(m - 1) * step,
        });
    }

    /// Returns the newest slot whose target is at or before `deadline_ns`,
    /// dropping older ones that were missed. `None` means "keep showing what
    /// is on screen".
    pub fn take_due(&mut self, deadline_ns: i64) -> Option<Planned> {
        let mut due = None;
        while let Some(p) = self.queue.front() {
            if p.target_ns > deadline_ns {
                break;
            }
            due = self.queue.pop_front();
        }
        due
    }

    /// Target time of the next pending slot, for waking the render loop.
    pub fn next_target(&self) -> Option<i64> {
        self.queue.front().map(|p| p.target_ns)
    }

    /// Frame ids that pending or future slots may still reference.
    pub fn retained_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.prev.iter().map(|f| f.id).chain(
            self.queue
                .iter()
                .flat_map(|p| match p.slot {
                    Slot::Real { id } => [Some(id), None],
                    Slot::Interp { prev, next, .. } => [Some(prev), Some(next)],
                })
                .flatten(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;
    const HZ60: i64 = 16_666_667;
    const HZ240: i64 = 4_166_667;
    const SRC30: i64 = 33_333_333;

    fn vrr(period: i64) -> Refresh {
        Refresh::Variable {
            min_period_ns: period,
        }
    }

    fn fixed(period: i64) -> Refresh {
        Refresh::Fixed {
            period_ns: period,
            phase_ns: 0,
        }
    }

    fn frame(id: u64) -> FrameRef {
        FrameRef {
            id,
            time_ns: id as i64 * SRC30,
        }
    }

    #[test]
    fn thirty_to_sixty_schedules_half_step_then_real() {
        let mut p = Pacer::new(PacingConfig::default());
        p.on_unique_frame(frame(0), 0, vrr(HZ60), Some(SRC30));
        assert_eq!(p.take_due(0).map(|x| x.slot), Some(Slot::Real { id: 0 }));

        let now = SRC30 + 2 * MS;
        p.on_unique_frame(frame(1), now, vrr(HZ60), Some(SRC30));
        let first = p.take_due(now).expect("interp due immediately");
        assert_eq!(
            first.slot,
            Slot::Interp {
                prev: 0,
                next: 1,
                t: 0.5
            }
        );
        assert_eq!(p.take_due(now + HZ60 - MS), None);
        let real = p.take_due(now + HZ60).expect("real after one step");
        assert_eq!(real.slot, Slot::Real { id: 1 });
    }

    #[test]
    fn multiplier_follows_display_and_cap() {
        let mut p = Pacer::new(PacingConfig::default());
        assert_eq!(p.multiplier(SRC30, vrr(HZ60)), 2);
        assert_eq!(p.multiplier(SRC30, vrr(8_333_333)), 4); // 120 Hz
        assert_eq!(p.multiplier(SRC30, vrr(6_944_444)), 4); // 144 Hz, clamped
        p.set_config(PacingConfig {
            output_fps_cap: Some(60.0),
            ..Default::default()
        });
        assert_eq!(p.multiplier(SRC30, vrr(6_944_444)), 2);
        p.set_config(PacingConfig {
            interpolate: false,
            ..Default::default()
        });
        assert_eq!(p.multiplier(SRC30, vrr(HZ60)), 1);
    }

    #[test]
    fn fixed_refresh_only_uses_even_multipliers() {
        let p = Pacer::new(PacingConfig {
            max_multiplier: 3,
            ..Default::default()
        });
        // 90 fps on VRR is fine; on fixed 240 Hz it would alternate 2/3
        // refreshes, so 2× (4 refreshes per step) is used instead.
        assert_eq!(p.multiplier(SRC30, vrr(HZ240)), 3);
        assert_eq!(p.multiplier(SRC30, fixed(HZ240)), 2);
        // 30 fps on 144 Hz has no even multiplier: keep the wanted one.
        assert_eq!(p.multiplier(SRC30, fixed(6_944_444)), 3);
        let p4 = Pacer::new(PacingConfig::default());
        assert_eq!(p4.multiplier(SRC30, fixed(HZ240)), 4);
    }

    #[test]
    fn fixed_refresh_targets_sit_on_vblanks() {
        let mut p = Pacer::new(PacingConfig::default());
        let r = Refresh::Fixed {
            period_ns: HZ240,
            phase_ns: MS,
        };
        p.on_unique_frame(frame(0), 0, r, Some(SRC30));
        p.take_due(i64::MAX);
        p.on_unique_frame(frame(1), SRC30 + 3 * MS, r, Some(SRC30));
        let mut last = None;
        while let Some(t) = p.next_target() {
            assert_eq!((t - MS) % HZ240, 0, "target {t} off the vblank grid");
            if let Some(l) = last {
                assert_eq!(t - l, 2 * HZ240);
            }
            last = Some(t);
            p.take_due(t);
        }
    }

    #[test]
    fn snap_up_keeps_targets_on_a_vblank() {
        let r = fixed(HZ240);
        assert_eq!(r.snap_up(HZ240), HZ240);
        assert_eq!(r.snap_up(HZ240 + 1), HZ240); // within tolerance
        assert_eq!(r.snap_up(HZ240 + HZ240 / 10), 2 * HZ240);
        assert_eq!(r.snap_up(-1), 0);
        assert_eq!(vrr(HZ240).snap_up(123), 123);
    }

    #[test]
    fn delay_estimator_ignores_rare_outliers() {
        let mut d = DelayEstimator::default();
        assert_eq!(d.estimate(), None);
        for i in 0..90 {
            d.push(if i % 45 == 0 { 20 * MS } else { 3 * MS + i });
        }
        let e = d.estimate().expect("enough samples");
        assert!(e < 4 * MS, "{e}");
    }

    #[test]
    fn stalls_are_not_interpolated() {
        let mut p = Pacer::new(PacingConfig::default());
        p.on_unique_frame(frame(0), 0, vrr(HZ60), Some(SRC30));
        p.take_due(0);
        let late = FrameRef {
            id: 1,
            time_ns: 500 * MS,
        };
        p.on_unique_frame(late, 500 * MS, vrr(HZ60), Some(SRC30));
        assert_eq!(
            p.take_due(500 * MS).map(|x| x.slot),
            Some(Slot::Real { id: 1 })
        );
    }

    #[test]
    fn late_render_loop_skips_to_newest_due() {
        let mut p = Pacer::new(PacingConfig {
            max_multiplier: 4,
            ..Default::default()
        });
        p.on_unique_frame(frame(0), 0, vrr(8_333_333), Some(SRC30));
        p.take_due(0);
        p.on_unique_frame(frame(1), SRC30, vrr(8_333_333), Some(SRC30));
        // Loop woke up two steps late.
        let got = p.take_due(SRC30 + 2 * 8_333_333).expect("due");
        assert_eq!(
            got.slot,
            Slot::Interp {
                prev: 0,
                next: 1,
                t: 0.75
            }
        );
        assert!(p.next_target().is_some());
    }

    #[test]
    fn new_frame_keeps_earlier_slots_and_drops_overlapping_ones() {
        let mut p = Pacer::new(PacingConfig {
            max_multiplier: 2,
            ..Default::default()
        });
        p.on_unique_frame(frame(0), 0, vrr(HZ240), Some(SRC30));
        // Pair 0→1 starts at 10 ms: interp at 10, real 1 at 26.7 ms.
        p.on_unique_frame(frame(1), 10 * MS, vrr(HZ240), Some(SRC30));
        // Pair 1→2 starts at 20 ms: real 1 (26.7 ms) overlaps and is dropped,
        // the interp at 10 ms still plays.
        p.on_unique_frame(frame(2), 20 * MS, vrr(HZ240), Some(SRC30));
        let mut seen = Vec::new();
        while let Some(t) = p.next_target() {
            seen.push(p.take_due(t).expect("due").slot);
        }
        assert_eq!(
            seen,
            vec![
                Slot::Real { id: 0 },
                Slot::Interp {
                    prev: 0,
                    next: 1,
                    t: 0.5
                },
                Slot::Interp {
                    prev: 1,
                    next: 2,
                    t: 0.5
                },
                Slot::Real { id: 2 },
            ]
        );
        assert!(!p.retained_ids().any(|id| id == 0));
    }
}
