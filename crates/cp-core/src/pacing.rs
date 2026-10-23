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
//! The planner is pure: it never sleeps and never touches the GPU. The render
//! loop asks [`Pacer::take_due`] once per vblank and renders whatever slot is
//! returned.

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
    /// Earliest host time (ns) this slot should be presented at.
    pub target_ns: i64,
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
    pub fn multiplier(&self, interval_ns: i64, display_period_ns: i64) -> u32 {
        if !self.cfg.interpolate || interval_ns <= 0 {
            return 1;
        }
        let cap_period = self
            .cfg
            .output_fps_cap
            .map_or(0.0, |fps| 1e9 / fps.max(1.0));
        let period = (display_period_ns as f64).max(cap_period).max(1.0);
        let m = (interval_ns as f64 / period).round() as u32;
        m.clamp(1, self.cfg.max_multiplier.max(1))
    }

    /// Schedules presentation for a newly arrived unique frame.
    ///
    /// `content_interval_ns` is the smoothed source frame interval (from
    /// [`crate::cadence::CadenceTracker`]); it guards against interpolating
    /// across stalls.
    pub fn on_unique_frame(
        &mut self,
        frame: FrameRef,
        now_ns: i64,
        display_period_ns: i64,
        content_interval_ns: Option<i64>,
    ) {
        // Whatever was still pending belongs to an older pair; showing it now
        // would only add latency.
        self.queue.clear();
        let prev = self.prev.replace(frame);

        let Some(prev) = prev else {
            self.queue.push_back(Planned {
                slot: Slot::Real { id: frame.id },
                target_ns: now_ns,
            });
            return;
        };
        let interval = frame.time_ns - prev.time_ns;
        let max_gap = content_interval_ns.map_or(100_000_000.0, |c| c as f64 * self.cfg.gap_factor);
        let m = if interval as f64 > max_gap {
            1
        } else {
            self.multiplier(interval, display_period_ns)
        };

        let step = interval / i64::from(m);
        for k in 1..m {
            let t = k as f32 / m as f32;
            self.queue.push_back(Planned {
                slot: Slot::Interp {
                    prev: prev.id,
                    next: frame.id,
                    t,
                },
                target_ns: now_ns + i64::from(k - 1) * step,
            });
        }
        self.queue.push_back(Planned {
            slot: Slot::Real { id: frame.id },
            target_ns: now_ns + i64::from(m - 1) * step,
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
    const SRC30: i64 = 33_333_333;

    fn frame(id: u64) -> FrameRef {
        FrameRef {
            id,
            time_ns: id as i64 * SRC30,
        }
    }

    #[test]
    fn thirty_to_sixty_schedules_half_step_then_real() {
        let mut p = Pacer::new(PacingConfig::default());
        p.on_unique_frame(frame(0), 0, HZ60, Some(SRC30));
        assert_eq!(p.take_due(0).map(|x| x.slot), Some(Slot::Real { id: 0 }));

        let now = SRC30 + 2 * MS;
        p.on_unique_frame(frame(1), now, HZ60, Some(SRC30));
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
        assert_eq!(p.multiplier(SRC30, HZ60), 2);
        assert_eq!(p.multiplier(SRC30, 8_333_333), 4); // 120 Hz
        assert_eq!(p.multiplier(SRC30, 6_944_444), 4); // 144 Hz, clamped
        p.set_config(PacingConfig {
            output_fps_cap: Some(60.0),
            ..Default::default()
        });
        assert_eq!(p.multiplier(SRC30, 6_944_444), 2);
        p.set_config(PacingConfig {
            interpolate: false,
            ..Default::default()
        });
        assert_eq!(p.multiplier(SRC30, HZ60), 1);
    }

    #[test]
    fn stalls_are_not_interpolated() {
        let mut p = Pacer::new(PacingConfig::default());
        p.on_unique_frame(frame(0), 0, HZ60, Some(SRC30));
        p.take_due(0);
        let late = FrameRef {
            id: 1,
            time_ns: 500 * MS,
        };
        p.on_unique_frame(late, 500 * MS, HZ60, Some(SRC30));
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
        p.on_unique_frame(frame(0), 0, 8_333_333, Some(SRC30));
        p.take_due(0);
        p.on_unique_frame(frame(1), SRC30, 8_333_333, Some(SRC30));
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
    fn new_frame_drops_stale_queue() {
        let mut p = Pacer::new(PacingConfig::default());
        p.on_unique_frame(frame(0), 0, HZ60, Some(SRC30));
        p.on_unique_frame(frame(1), SRC30, HZ60, Some(SRC30));
        p.on_unique_frame(frame(2), 2 * SRC30, HZ60, Some(SRC30));
        let ids: Vec<u64> = p.retained_ids().collect();
        assert!(!ids.contains(&0));
        assert_eq!(
            p.take_due(2 * SRC30).map(|x| x.slot),
            Some(Slot::Interp {
                prev: 1,
                next: 2,
                t: 0.5
            })
        );
    }
}
