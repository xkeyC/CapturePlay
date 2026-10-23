//! Clock-drift compensation between the capture card's audio clock and the
//! output device's clock.
//!
//! Instead of dropping/duplicating samples (audible clicks), the output side
//! resamples by a ratio within ±0.5 % chosen by a slow controller that keeps
//! the ring-buffer fill at the latency target.

/// Linear-interpolating fractional-rate reader over interleaved frames.
#[derive(Debug, Clone)]
pub struct RatePuller {
    channels: usize,
    frac: f64,
    cur: Vec<f32>,
    next: Vec<f32>,
    primed: bool,
}

impl RatePuller {
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            frac: 0.0,
            cur: vec![0.0; channels],
            next: vec![0.0; channels],
            primed: false,
        }
    }

    /// Fills `out` (interleaved). `ratio` = input frames consumed per output
    /// frame. `pop` writes one input frame and returns false when empty.
    /// Returns the number of output frames backed by real input (the rest is
    /// silence after an underrun).
    pub fn fill(
        &mut self,
        out: &mut [f32],
        ratio: f64,
        mut pop: impl FnMut(&mut [f32]) -> bool,
    ) -> usize {
        let ch = self.channels;
        if !self.primed {
            if !pop(&mut self.cur) || !pop(&mut self.next) {
                out.fill(0.0);
                return 0;
            }
            self.primed = true;
        }
        let frames = out.len() / ch;
        for f in 0..frames {
            let w = self.frac as f32;
            for c in 0..ch {
                out[f * ch + c] = self.cur[c] + (self.next[c] - self.cur[c]) * w;
            }
            self.frac += ratio;
            while self.frac >= 1.0 {
                self.frac -= 1.0;
                std::mem::swap(&mut self.cur, &mut self.next);
                if !pop(&mut self.next) {
                    // Underrun: silence the remainder and re-prime later.
                    self.primed = false;
                    self.frac = 0.0;
                    out[(f + 1) * ch..].fill(0.0);
                    return f + 1;
                }
            }
        }
        frames
    }
}

/// Chooses the resampling ratio from the buffer fill.
#[derive(Debug, Clone)]
pub struct DriftController {
    target: f64,
    filtered: Option<f64>,
}

/// Maximum correction (±0.5 %, inaudible pitch change).
pub const MAX_CORRECTION: f64 = 0.005;

impl DriftController {
    pub fn new(target_frames: f64) -> Self {
        Self {
            target: target_frames.max(1.0),
            filtered: None,
        }
    }

    pub fn set_target(&mut self, target_frames: f64) {
        self.target = target_frames.max(1.0);
    }

    pub fn target(&self) -> f64 {
        self.target
    }

    /// Ratio for the next output period given the current fill (frames).
    pub fn ratio(&mut self, fill_frames: f64) -> f64 {
        let f = match self.filtered {
            Some(prev) => prev + (fill_frames - prev) * 0.05,
            None => fill_frames,
        };
        self.filtered = Some(f);
        let err = (f - self.target) / self.target;
        1.0 + (err * 0.02).clamp(-MAX_CORRECTION, MAX_CORRECTION)
    }

    /// Frames to discard immediately when far above target (e.g. after a
    /// stall), so latency recovers at once instead of over minutes.
    pub fn excess(&self, fill_frames: f64) -> usize {
        if fill_frames > self.target * 3.0 {
            (fill_frames - self.target) as usize
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.filtered = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn unity_ratio_is_transparent() {
        let mut p = RatePuller::new(1);
        let mut src: VecDeque<f32> = (0..100).map(|i| i as f32).collect();
        let mut out = vec![0.0; 50];
        let n = p.fill(&mut out, 1.0, |f| {
            src.pop_front().map(|v| f[0] = v).is_some()
        });
        assert_eq!(n, 50);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[49], 49.0);
    }

    #[test]
    fn underrun_outputs_silence() {
        let mut p = RatePuller::new(2);
        let mut src: VecDeque<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let mut out = vec![9.0; 20];
        let n = p.fill(&mut out, 1.0, |f| {
            if src.len() < 2 {
                return false;
            }
            f[0] = src.pop_front().unwrap_or(0.0);
            f[1] = src.pop_front().unwrap_or(0.0);
            true
        });
        assert!(n < 10);
        assert!(out[n * 2..].iter().all(|&v| v == 0.0));
    }

    /// Capture runs 0.2 % fast; the controller must hold the fill near the
    /// target instead of letting latency grow.
    #[test]
    fn controller_absorbs_clock_drift() {
        let rate = 48_000.0;
        let target = rate * 0.03;
        let mut ctl = DriftController::new(target);
        let mut puller = RatePuller::new(1);
        let mut fill: VecDeque<f32> = VecDeque::new();
        let mut produced = 0.0f64;
        let period = 480; // 10 ms output periods
        let mut out = vec![0.0; period];
        for step in 0..6000 {
            // 60 s simulated; capture delivers 1.002 × nominal.
            produced += period as f64 * 1.002;
            while produced >= 1.0 {
                fill.push_back(0.0);
                produced -= 1.0;
            }
            let r = ctl.ratio(fill.len() as f64);
            puller.fill(&mut out, r, |f| {
                fill.pop_front().map(|v| f[0] = v).is_some()
            });
            if step > 3000 {
                let err = (fill.len() as f64 - target).abs() / target;
                assert!(err < 0.35, "fill {} target {target}", fill.len());
            }
        }
    }
}
