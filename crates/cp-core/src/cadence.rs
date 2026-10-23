//! Duplicate-frame detection and content frame-rate estimation.
//!
//! A console rendering at 30 fps still outputs a 60 Hz HDMI signal, so the
//! capture card delivers every game frame twice (or in uneven 3:1 / 2:2
//! cadences when the game misses its target). Interpolating between two
//! identical frames does nothing, so the pipeline must work on *unique*
//! frames and their real timestamps. This module decides which captured
//! frames are unique.

use std::collections::VecDeque;

/// Sparse luma fingerprint of one captured frame, computed on the capture
/// thread while the frame is copied (a few thousand samples, microseconds).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrameSignature {
    /// Luma samples normalized to 16 bits.
    pub samples: Vec<u16>,
}

impl FrameSignature {
    pub const GRID_W: usize = 64;
    pub const GRID_H: usize = 36;

    /// Samples a `GRID_W × GRID_H` grid out of a luma plane.
    ///
    /// `read(x, y)` returns the luma value at pixel `(x, y)` scaled to 16 bits.
    pub fn from_plane(width: u32, height: u32, mut read: impl FnMut(u32, u32) -> u16) -> Self {
        let mut samples = Vec::with_capacity(Self::GRID_W * Self::GRID_H);
        for gy in 0..Self::GRID_H {
            let y = ((gy as u64 * 2 + 1) * u64::from(height) / (Self::GRID_H as u64 * 2)) as u32;
            for gx in 0..Self::GRID_W {
                let x = ((gx as u64 * 2 + 1) * u64::from(width) / (Self::GRID_W as u64 * 2)) as u32;
                samples.push(read(x, y));
            }
        }
        Self { samples }
    }
}

/// Tuning for [`CadenceTracker`].
#[derive(Debug, Clone, Copy)]
pub struct CadenceConfig {
    /// Max mean absolute difference (16-bit units) for a duplicate.
    pub max_mean_diff: f32,
    /// Max fraction of samples that may differ by more than `noise_floor`.
    pub max_changed_fraction: f32,
    /// Per-sample difference below which a change is treated as noise.
    pub noise_floor: u16,
    /// Unique-frame intervals kept for the rate estimate.
    pub window: usize,
}

impl Default for CadenceConfig {
    fn default() -> Self {
        Self {
            // ≈0.4 of an 8-bit code value on average.
            max_mean_diff: 100.0,
            max_changed_fraction: 0.002,
            // 2 codes at 8 bit.
            noise_floor: 512,
            window: 32,
        }
    }
}

/// Classification of one captured frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceInfo {
    pub unique: bool,
    /// Time since the previous unique frame (ns); `None` for the first one.
    pub interval_ns: Option<i64>,
    /// Smoothed content frame interval (ns) over recent unique frames.
    pub content_interval_ns: Option<i64>,
}

#[derive(Debug, Default)]
pub struct CadenceTracker {
    cfg: CadenceConfig,
    last_sig: Option<FrameSignature>,
    last_unique_ns: Option<i64>,
    intervals: VecDeque<i64>,
}

impl CadenceTracker {
    pub fn new(cfg: CadenceConfig) -> Self {
        Self {
            cfg,
            ..Default::default()
        }
    }

    pub fn reset(&mut self) {
        self.last_sig = None;
        self.last_unique_ns = None;
        self.intervals.clear();
    }

    /// Feeds one captured frame (timestamped in ns on any monotonic clock)
    /// together with the nominal signal frame interval.
    pub fn push(
        &mut self,
        sig: FrameSignature,
        time_ns: i64,
        signal_interval_ns: i64,
    ) -> CadenceInfo {
        let unique = match &self.last_sig {
            Some(prev) => !self.is_duplicate(prev, &sig),
            None => true,
        };
        if !unique {
            return CadenceInfo {
                unique: false,
                interval_ns: None,
                content_interval_ns: self.content_interval(),
            };
        }
        let interval = self.last_unique_ns.map(|t| time_ns - t);
        if let Some(iv) = interval {
            // Long holds are static content or menus, not the game's frame rate.
            if iv > 0 && iv <= signal_interval_ns.saturating_mul(4) {
                if self.intervals.len() == self.cfg.window.max(1) {
                    self.intervals.pop_front();
                }
                self.intervals.push_back(iv);
            }
        }
        self.last_sig = Some(sig);
        self.last_unique_ns = Some(time_ns);
        CadenceInfo {
            unique: true,
            interval_ns: interval,
            content_interval_ns: self.content_interval(),
        }
    }

    /// Mean of recent unique intervals. Not the median: a 42 fps game on a
    /// 60 Hz signal alternates 16.7 / 33.3 ms intervals, whose median would
    /// read as 60 fps.
    pub fn content_interval(&self) -> Option<i64> {
        if self.intervals.len() < 3 {
            return None;
        }
        Some(self.intervals.iter().sum::<i64>() / self.intervals.len() as i64)
    }

    fn is_duplicate(&self, a: &FrameSignature, b: &FrameSignature) -> bool {
        if a.samples.len() != b.samples.len() || a.samples.is_empty() {
            return false;
        }
        let mut sum = 0u64;
        let mut changed = 0usize;
        for (&x, &y) in a.samples.iter().zip(&b.samples) {
            let d = x.abs_diff(y);
            sum += u64::from(d);
            if d > self.cfg.noise_floor {
                changed += 1;
            }
        }
        let n = a.samples.len() as f32;
        (sum as f32 / n) <= self.cfg.max_mean_diff
            && (changed as f32 / n) <= self.cfg.max_changed_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGNAL: i64 = 16_666_667;

    fn sig(seed: u16) -> FrameSignature {
        FrameSignature::from_plane(1920, 1080, |x, y| {
            ((x as u16).wrapping_mul(31) ^ (y as u16).wrapping_mul(17))
                .wrapping_add(seed.wrapping_mul(4099))
        })
    }

    #[test]
    fn detects_30_in_60_cadence() {
        let mut t = CadenceTracker::new(CadenceConfig::default());
        let mut uniques = 0;
        for i in 0..60i64 {
            let info = t.push(sig((i / 2) as u16), i * SIGNAL, SIGNAL);
            assert_eq!(info.unique, i % 2 == 0, "frame {i}");
            uniques += usize::from(info.unique);
        }
        assert_eq!(uniques, 30);
        let est = t.content_interval().expect("estimate");
        assert!((est - 2 * SIGNAL).abs() < 1000);
    }

    #[test]
    fn tolerates_capture_noise() {
        let mut t = CadenceTracker::new(CadenceConfig::default());
        let base = sig(1);
        t.push(base.clone(), 0, SIGNAL);
        let mut noisy = base;
        for (i, s) in noisy.samples.iter_mut().enumerate() {
            if i % 7 == 0 {
                *s = s.saturating_add(200);
            }
        }
        assert!(!t.push(noisy, SIGNAL, SIGNAL).unique);
    }

    #[test]
    fn small_moving_object_is_unique() {
        let mut t = CadenceTracker::new(CadenceConfig::default());
        let a = sig(1);
        t.push(a.clone(), 0, SIGNAL);
        let mut b = a;
        for s in &mut b.samples[100..110] {
            *s = s.wrapping_add(20_000);
        }
        assert!(t.push(b, SIGNAL, SIGNAL).unique);
    }

    #[test]
    fn static_holds_do_not_pollute_rate() {
        let mut t = CadenceTracker::new(CadenceConfig::default());
        for i in 0..10i64 {
            t.push(sig(i as u16), i * SIGNAL, SIGNAL);
        }
        // 2 s static screen, then motion resumes.
        t.push(sig(1000), 10 * SIGNAL + 2_000_000_000, SIGNAL);
        let est = t.content_interval().expect("estimate");
        assert!((est - SIGNAL).abs() < 1000);
    }
}
