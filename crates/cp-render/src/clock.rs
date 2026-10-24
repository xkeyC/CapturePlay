//! Sleeping until an exact host time.
//!
//! On a variable-refresh display a frame appears when it is presented, so
//! presentation timing is only as even as the wake-up. A high-resolution
//! waitable timer gets within a fraction of a millisecond; the rest is a
//! short yield loop.

use cp_core::time::HostTime;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::System::Threading::{
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, INFINITE, SetWaitableTimer,
    TIMER_ALL_ACCESS, WaitForMultipleObjects, WaitForSingleObject,
};

#[derive(Debug)]
pub struct PreciseSleep {
    timer: HANDLE,
}

impl PreciseSleep {
    /// Timer wake-ups are this early; the remainder is spun.
    const SPIN_NS: i64 = 600_000;

    pub fn new() -> windows::core::Result<Self> {
        // SAFETY: anonymous high-resolution timer.
        let timer = unsafe {
            CreateWaitableTimerExW(
                None,
                None,
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS.0,
            )?
        };
        Ok(Self { timer })
    }

    /// Returns at `deadline_ns` (host time), or immediately if it passed.
    pub fn until(&self, deadline_ns: i64) {
        let _ = self.until_or(deadline_ns, None);
    }

    /// Like [`Self::until`], but returns `true` early when `event` is
    /// signaled before the final spin (the caller handles it and waits
    /// again).
    pub fn until_or(&self, deadline_ns: i64, event: Option<HANDLE>) -> bool {
        let left = deadline_ns - HostTime::now().0;
        if left > Self::SPIN_NS {
            // Relative due time in 100 ns units.
            let due = -((left - Self::SPIN_NS) / 100).max(1);
            // SAFETY: valid timer/event handles; no completion routine.
            unsafe {
                if SetWaitableTimer(self.timer, &due, 0, None, None, false).is_ok() {
                    match event {
                        Some(e) => {
                            let r = WaitForMultipleObjects(&[self.timer, e], false, INFINITE);
                            if r.0 == WAIT_OBJECT_0.0 + 1 {
                                return true;
                            }
                        }
                        None => {
                            WaitForSingleObject(self.timer, INFINITE);
                        }
                    }
                }
            }
        }
        while HostTime::now().0 < deadline_ns {
            std::thread::yield_now();
        }
        false
    }
}

impl Drop for PreciseSleep {
    fn drop(&mut self) {
        // SAFETY: handle owned by self.
        unsafe {
            let _ = CloseHandle(self.timer);
        }
    }
}
