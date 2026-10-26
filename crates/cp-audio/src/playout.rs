//! Output side of the passthrough, backend independent: prefill to the
//! latency target, drift-corrected resampling out of the ring, underrun and
//! fill accounting, volume. Each backend calls [`Playout::fill`] with the
//! output buffer its device asks for.

use std::sync::atomic::Ordering;

use rtrb::Consumer;

use crate::config::AudioShared;
use crate::drift::{DriftController, RatePuller};

#[derive(Debug)]
pub struct Playout {
    channels: usize,
    frames_per_ms: f64,
    target_latency_ms: u32,
    ctl: DriftController,
    puller: RatePuller,
    /// Prefilling to the target before playing.
    waiting: bool,
}

impl Playout {
    pub fn new(rate: u32, channels: usize, target_latency_ms: u32, offset_ms: i32) -> Self {
        let frames_per_ms = f64::from(rate) / 1000.0;
        Self {
            channels,
            frames_per_ms,
            target_latency_ms,
            ctl: DriftController::new(target_frames(target_latency_ms, offset_ms, frames_per_ms)),
            puller: RatePuller::new(channels),
            waiting: true,
        }
    }

    /// Fills `out` (interleaved, whole frames) from `ring`.
    pub fn fill(&mut self, out: &mut [f32], ring: &mut Consumer<f32>, shared: &AudioShared) {
        let ch = self.channels;
        let offset = shared.offset_ms.load(Ordering::Relaxed);
        self.ctl.set_target(target_frames(
            self.target_latency_ms,
            offset,
            self.frames_per_ms,
        ));
        let fill = (ring.slots() / ch) as f64;
        shared
            .fill_ms
            .store((fill / self.frames_per_ms) as u32, Ordering::Relaxed);
        let excess = self.ctl.excess(fill);
        if excess > 0 {
            if let Ok(c) = ring.read_chunk(excess * ch) {
                c.commit_all();
            }
            self.ctl.reset();
        }
        if self.waiting && fill >= self.ctl.target() {
            self.waiting = false;
        }
        if self.waiting {
            out.fill(0.0);
            return;
        }
        let ratio = self.ctl.ratio(fill);
        let real = self.puller.fill(out, ratio, |frame| {
            if ring.slots() < frame.len() {
                return false;
            }
            match ring.read_chunk(frame.len()) {
                Ok(c) => {
                    let (a, b) = c.as_slices();
                    frame[..a.len()].copy_from_slice(a);
                    frame[a.len()..].copy_from_slice(b);
                    c.commit_all();
                    true
                }
                Err(_) => false,
            }
        });
        if real < out.len() / ch {
            shared.underruns.fetch_add(1, Ordering::Relaxed);
            self.waiting = true;
        }
        let vol = shared.volume_milli.load(Ordering::Relaxed) as f32 / 1000.0;
        if (vol - 1.0).abs() > f32::EPSILON {
            out.iter_mut().for_each(|s| *s *= vol);
        }
    }
}

fn target_frames(latency_ms: u32, offset_ms: i32, frames_per_ms: f64) -> f64 {
    f64::from((latency_ms as i32 + offset_ms).max(10)) * frames_per_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefills_then_plays_and_counts_underruns() {
        let shared = AudioShared::default();
        shared.volume_milli.store(1000, Ordering::Relaxed);
        let (mut tx, mut rx) = rtrb::RingBuffer::<f32>::new(48_000);
        // 48 kHz stereo, 20 ms target.
        let mut p = Playout::new(48_000, 2, 20, 0);
        let mut out = vec![1.0f32; 480 * 2];
        p.fill(&mut out, &mut rx, &shared);
        assert!(out.iter().all(|&s| s == 0.0), "silence while prefilling");
        for i in 0..(48 * 25 * 2) {
            let _ = tx.push((i % 7) as f32 * 0.1);
        }
        p.fill(&mut out, &mut rx, &shared);
        assert!(
            out.iter().any(|&s| s != 0.0),
            "plays once the target is reached"
        );
        assert_eq!(shared.underruns.load(Ordering::Relaxed), 0);
        for _ in 0..4 {
            p.fill(&mut out, &mut rx, &shared);
        }
        assert!(
            shared.underruns.load(Ordering::Relaxed) >= 1,
            "drained ring underruns"
        );
    }
}
