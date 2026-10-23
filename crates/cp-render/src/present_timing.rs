//! When presented frames actually reached the screen.
//!
//! DXGI frame statistics report, for the latest displayed present, the QPC
//! time of the refresh that showed it. Matching those against the intended
//! display times gives the presentation jitter, and tells variable-refresh
//! displays from fixed-refresh ones: when off-grid targets still land on a
//! regular vblank grid the display is fixed-refresh; when they land where
//! they were aimed, VRR is active.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefreshKind {
    #[default]
    Unknown,
    Variable,
    Fixed,
}

/// One frame-statistics reading: the latest vblank (`sync_*`) and the
/// refresh on which present `present_count` was first shown.
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    pub present_count: u32,
    pub present_refresh_count: u32,
    pub sync_refresh_count: u32,
    pub sync_ns: i64,
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    id: u32,
    target: i64,
    shown: i64,
}

#[derive(Debug, Default)]
pub struct PresentTiming {
    /// Presents not yet seen in the statistics: (present id, target).
    pending: VecDeque<(u32, i64)>,
    samples: VecDeque<Sample>,
    /// Off-grid intervals: did the display snap them to its grid?
    votes: VecDeque<bool>,
    kind: RefreshKind,
    /// First and latest (refresh count, time), for the true refresh period.
    anchor: Option<(u32, i64)>,
    vblank: Option<(u32, i64)>,
}

impl PresentTiming {
    const SAMPLES: usize = 120;
    const VOTES: usize = 60;
    const MIN_VOTES: usize = 20;

    /// Forgets everything (display or mode changed).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn kind(&self) -> RefreshKind {
        self.kind
    }

    /// A scheduled frame was presented as `id`, meant to appear at `target`.
    pub fn on_present(&mut self, id: u32, target_ns: i64) {
        if self.pending.len() == 64 {
            self.pending.pop_front();
        }
        self.pending.push_back((id, target_ns));
    }

    pub fn on_stats(&mut self, s: FrameStats, nominal_period_ns: i64) {
        if s.sync_ns > 0 && self.vblank.is_none_or(|(c, _)| c != s.sync_refresh_count) {
            // Restart the period anchor after long gaps (occlusion, mode
            // change) where counts and times may not line up.
            let stale = self
                .vblank
                .is_none_or(|(c, t)| s.sync_refresh_count < c || s.sync_ns - t > 1_000_000_000);
            if stale {
                self.anchor = Some((s.sync_refresh_count, s.sync_ns));
            }
            self.vblank = Some((s.sync_refresh_count, s.sync_ns));
        }
        while let Some(&(id, target)) = self.pending.front() {
            if id > s.present_count {
                break;
            }
            self.pending.pop_front();
            // The frame was shown `behind` refreshes before the latest
            // vblank. On VRR that is normally 0 (the display waits for the
            // next present) and refreshes are not evenly spaced, so only
            // exact samples count there; on fixed refresh step back by the
            // period.
            let behind = s.sync_refresh_count.wrapping_sub(s.present_refresh_count);
            let usable = behind == 0 || (behind <= 4 && self.kind != RefreshKind::Variable);
            if id == s.present_count && s.sync_ns > 0 && usable {
                let shown = s.sync_ns - i64::from(behind) * self.period_ns(nominal_period_ns);
                self.add_sample(Sample { id, target, shown }, nominal_period_ns);
            }
        }
    }

    fn add_sample(&mut self, cur: Sample, nominal_period_ns: i64) {
        let period = self.period_ns(nominal_period_ns) as f64;
        if let Some(prev) = self.samples.back().filter(|p| p.id + 1 == cur.id) {
            let frac = |ns: i64| {
                let q = ns as f64 / period;
                (q - q.round()).abs()
            };
            if frac(cur.target - prev.target) > 0.2 {
                if self.votes.len() == Self::VOTES {
                    self.votes.pop_front();
                }
                self.votes.push_back(frac(cur.shown - prev.shown) < 0.06);
                self.classify();
            }
        }
        if self.samples.len() == Self::SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(cur);
    }

    fn classify(&mut self) {
        if self.votes.len() < Self::MIN_VOTES {
            return;
        }
        let snapped = self.votes.iter().filter(|&&v| v).count() as f64 / self.votes.len() as f64;
        if snapped > 0.85 {
            self.kind = RefreshKind::Fixed;
        } else if snapped < 0.35 {
            self.kind = RefreshKind::Variable;
        }
    }

    /// Measured refresh period (falls back to the nominal one).
    pub fn period_ns(&self, nominal_ns: i64) -> i64 {
        if let (Some((c0, t0)), Some((c1, t1))) = (self.anchor, self.vblank)
            && c1.wrapping_sub(c0) >= 120
        {
            let p = (t1 - t0) / i64::from(c1.wrapping_sub(c0));
            if (p - nominal_ns).abs() * 50 < nominal_ns {
                return p;
            }
        }
        nominal_ns
    }

    /// Time of the latest known vblank.
    pub fn vblank_ns(&self) -> Option<i64> {
        self.vblank.map(|(_, t)| t)
    }

    /// 95th percentile of |error − median error| (ms): how unevenly frames
    /// reached the screen relative to their schedule.
    pub fn jitter_ms(&self) -> Option<f64> {
        if self.samples.len() < 10 {
            return None;
        }
        let mut e: Vec<i64> = self.samples.iter().map(|s| s.shown - s.target).collect();
        e.sort_unstable();
        let med = e[e.len() / 2];
        let mut d: Vec<i64> = e.iter().map(|x| (x - med).abs()).collect();
        d.sort_unstable();
        Some(d[(d.len() * 95 / 100).min(d.len() - 1)] as f64 / 1e6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: i64 = 4_166_667; // 240 Hz
    const STEP: i64 = 11_111_111; // 90 fps targets, off the 240 Hz grid

    fn feed(t: &mut PresentTiming, shown: impl Fn(i64) -> i64) {
        for i in 0..80u32 {
            let target = 1_000_000_000 + i64::from(i) * STEP;
            t.on_present(i + 1, target);
            let s = shown(target);
            t.on_stats(
                FrameStats {
                    present_count: i + 1,
                    present_refresh_count: (s / P) as u32,
                    sync_refresh_count: (s / P) as u32,
                    sync_ns: s,
                },
                P,
            );
        }
    }

    #[test]
    fn detects_variable_refresh() {
        let mut t = PresentTiming::default();
        feed(&mut t, |target| target + 300_000);
        assert_eq!(t.kind(), RefreshKind::Variable);
        assert!(t.jitter_ms().expect("samples") < 0.01);
    }

    #[test]
    fn detects_fixed_refresh_and_its_judder() {
        let mut t = PresentTiming::default();
        feed(&mut t, |target| (target / P + 1) * P);
        assert_eq!(t.kind(), RefreshKind::Fixed);
        assert!(t.jitter_ms().expect("samples") > 1.0);
    }
}
