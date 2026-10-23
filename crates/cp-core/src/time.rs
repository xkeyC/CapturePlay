//! Host clock shared by capture, audio and presentation.
//!
//! All timestamps are nanoseconds on the QueryPerformanceCounter timeline, so
//! they compare directly with DXGI frame statistics and WASAPI positions.

use std::ops::{Add, Sub};
use std::sync::OnceLock;
use std::time::Duration;

/// Nanoseconds on the host QPC timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HostTime(pub i64);

impl HostTime {
    pub fn now() -> Self {
        Self::from_qpc(qpc::counter())
    }

    /// Converts a raw QPC tick count (e.g. `DXGI_FRAME_STATISTICS::SyncQPCTime`).
    pub fn from_qpc(ticks: i64) -> Self {
        let freq = qpc::frequency();
        Self(((i128::from(ticks) * 1_000_000_000) / i128::from(freq)) as i64)
    }

    /// Converts back to QPC ticks.
    pub fn to_qpc(self) -> i64 {
        let freq = qpc::frequency();
        ((i128::from(self.0) * i128::from(freq)) / 1_000_000_000) as i64
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 * 1e-9
    }

    pub fn saturating_duration_since(self, earlier: Self) -> Duration {
        Duration::from_nanos(u64::try_from(self.0 - earlier.0).unwrap_or(0))
    }
}

impl Add<Duration> for HostTime {
    type Output = Self;
    fn add(self, d: Duration) -> Self {
        Self(self.0 + d.as_nanos() as i64)
    }
}

impl Sub<Duration> for HostTime {
    type Output = Self;
    fn sub(self, d: Duration) -> Self {
        Self(self.0 - d.as_nanos() as i64)
    }
}

/// Signed difference in nanoseconds.
impl Sub for HostTime {
    type Output = i64;
    fn sub(self, rhs: Self) -> i64 {
        self.0 - rhs.0
    }
}

#[cfg(windows)]
mod qpc {
    use super::OnceLock;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn QueryPerformanceCounter(count: *mut i64) -> i32;
        fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    }

    pub fn counter() -> i64 {
        let mut v = 0i64;
        // SAFETY: valid out pointer; the call cannot fail on Windows XP+.
        unsafe { QueryPerformanceCounter(&mut v) };
        v
    }

    pub fn frequency() -> i64 {
        static FREQ: OnceLock<i64> = OnceLock::new();
        *FREQ.get_or_init(|| {
            let mut v = 0i64;
            // SAFETY: valid out pointer; the call cannot fail on Windows XP+.
            unsafe { QueryPerformanceFrequency(&mut v) };
            v.max(1)
        })
    }
}

#[cfg(not(windows))]
mod qpc {
    use super::OnceLock;
    use std::time::Instant;

    static EPOCH: OnceLock<Instant> = OnceLock::new();

    pub fn counter() -> i64 {
        EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as i64
    }

    pub fn frequency() -> i64 {
        1_000_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_and_round_trips() {
        let a = HostTime::now();
        let b = HostTime::now();
        assert!(b >= a);
        let t = HostTime(123_456_789_000);
        assert!((HostTime::from_qpc(t.to_qpc()) - t).abs() < 1_000);
    }
}
