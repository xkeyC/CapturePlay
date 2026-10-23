//! Reconstructing when the game finished each frame.
//!
//! The capture card delivers a fixed-rate signal (e.g. 60 Hz) even when the
//! console renders at a variable rate: a game running at 42 fps shows each
//! frame on the next 60 Hz refresh, so unique frames arrive 16.7 or 33.3 ms
//! apart in an irregular pattern — judder that interpolating between the
//! observed times would faithfully reproduce.
//!
//! The quantization still bounds the truth: a frame first seen at `obs`
//! finished in `(obs - signal_period, obs]`. Assuming locally even pacing,
//! [`Retimer`] fits the line `t_i = a + b·i` through recent frames that
//! satisfies every bound and takes the *latest* such line (least latency).
//! Content locked to the signal (30 fps in 60 Hz) fits with zero offset, so
//! its timing is unchanged. When no even line fits (the game's frame time
//! really changes) the window shrinks, down to the observed time itself.

use std::collections::VecDeque;

#[derive(Debug, Default)]
pub struct Retimer {
    /// Observed times of recent unique frames, oldest first.
    obs: VecDeque<i64>,
}

impl Retimer {
    const WINDOW: usize = 24;
    /// Timestamp noise allowed on each bound.
    const SLACK_NS: i64 = 750_000;

    pub fn reset(&mut self) {
        self.obs.clear();
    }

    /// Estimated content time of a unique frame first observed at `obs_ns`
    /// on a signal with frame interval `signal_ns`.
    pub fn push(&mut self, obs_ns: i64, signal_ns: i64) -> i64 {
        // Holds (menus, loading) break the even-pacing assumption.
        if self
            .obs
            .back()
            .is_some_and(|&last| obs_ns - last > signal_ns * 4 || obs_ns <= last)
        {
            self.obs.clear();
        }
        if self.obs.len() == Self::WINDOW {
            self.obs.pop_front();
        }
        self.obs.push_back(obs_ns);
        let v: Vec<i64> = self.obs.iter().copied().collect();
        for start in 0..v.len().saturating_sub(2) {
            if let Some(t) = fit_latest(&v[start..], signal_ns) {
                return t;
            }
        }
        obs_ns
    }
}

/// Latest even line through `obs` within the quantization bounds; returns
/// its value at the newest frame.
fn fit_latest(obs: &[i64], signal_ns: i64) -> Option<i64> {
    let n = obs.len();
    let width = (signal_ns + 2 * Retimer::SLACK_NS) as f64;
    // Slope: middle of the range every pair of bounds allows. Much steadier
    // than least squares on quantized data, whose bias moves with the
    // cadence pattern.
    let (mut b_lo, mut b_hi) = (0.0f64, f64::MAX);
    for i in 0..n {
        for j in i + 1..n {
            let d = (obs[j] - obs[i]) as f64;
            let k = (j - i) as f64;
            b_lo = b_lo.max((d - width) / k);
            b_hi = b_hi.min((d + width) / k);
        }
    }
    if b_lo > b_hi {
        return None;
    }
    let b = (b_lo + b_hi) / 2.0;
    // Feasible intercepts: o_i - S - slack <= a + b·i <= o_i + slack.
    let base = obs[0];
    let mut lo = f64::MIN;
    let mut hi = f64::MAX;
    for (i, &o) in obs.iter().enumerate() {
        let y = (o - base) as f64 - b * i as f64;
        lo = lo.max(y - (signal_ns + Retimer::SLACK_NS) as f64);
        hi = hi.min(y + Retimer::SLACK_NS as f64);
    }
    if lo > hi {
        return None;
    }
    // Latest line, but never after the newest observation.
    let t = base + (hi + b * (n - 1) as f64).round() as i64;
    Some(t.min(obs[n - 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGNAL: i64 = 16_666_667;
    const MS: i64 = 1_000_000;

    /// Unique frames a 60 Hz signal shows for content finishing at `times`.
    fn quantize(times: &[i64]) -> Vec<i64> {
        times
            .iter()
            .map(|&t| (t + SIGNAL - 1).div_euclid(SIGNAL) * SIGNAL)
            .collect()
    }

    fn spread(v: &[i64]) -> i64 {
        v.iter().max().expect("non-empty") - v.iter().min().expect("non-empty")
    }

    #[test]
    fn locked_content_keeps_its_timing() {
        let mut r = Retimer::default();
        for i in 0..40 {
            let obs = 1_000 * MS + i * 2 * SIGNAL;
            assert_eq!(r.push(obs, SIGNAL), obs, "frame {i}");
        }
    }

    #[test]
    fn variable_rate_is_evened_out() {
        let content = 23_809_524; // 42 fps
        let truth: Vec<i64> = (0..80).map(|i| 5 * MS + i * content).collect();
        let obs = quantize(&truth);
        let mut r = Retimer::default();
        let est: Vec<i64> = obs.iter().map(|&o| r.push(o, SIGNAL)).collect();

        let obs_iv: Vec<i64> = obs.windows(2).map(|w| w[1] - w[0]).collect();
        let est_iv: Vec<i64> = est[30..].windows(2).map(|w| w[1] - w[0]).collect();
        assert!(spread(&obs_iv) > 16 * MS, "observed judder");
        assert!(
            spread(&est_iv) < 2 * MS,
            "estimated spread {}",
            spread(&est_iv)
        );
        for (i, (&e, &o)) in est.iter().zip(&obs).enumerate() {
            assert!(
                e <= o && e > o - SIGNAL - Retimer::SLACK_NS,
                "frame {i}: bound"
            );
        }
    }

    #[test]
    fn rate_changes_fall_back_to_bounds() {
        // 30 fps, then a stretch at 45 fps.
        let mut truth: Vec<i64> = (0..20).map(|i| i * 2 * SIGNAL).collect();
        let t0 = *truth.last().expect("non-empty");
        truth.extend((1..30).map(|i| t0 + i * 22_222_222));
        let obs = quantize(&truth);
        let mut r = Retimer::default();
        for (i, &o) in obs.iter().enumerate() {
            let e = r.push(o, SIGNAL);
            assert!(e <= o && e > o - SIGNAL - Retimer::SLACK_NS, "frame {i}");
        }
    }

    #[test]
    fn holds_reset_history() {
        let mut r = Retimer::default();
        for i in 0..10 {
            r.push(i * 2 * SIGNAL, SIGNAL);
        }
        let after = 10 * 2 * SIGNAL + 500 * MS;
        assert_eq!(r.push(after, SIGNAL), after);
    }
}
