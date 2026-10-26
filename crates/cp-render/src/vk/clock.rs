//! Sleeping until an exact host time, interruptible by an event.
//!
//! On a variable-refresh display a frame appears when it is presented, so
//! presentation timing is only as even as the wake-up. An absolute
//! `CLOCK_MONOTONIC` timerfd gets within the timer slack (~50 µs); the rest
//! is a short yield loop. [`Event`] is an eventfd used like the D3D12
//! backend's auto-reset event.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use cp_core::time::HostTime;

/// Auto-reset event (eventfd): `set` from any thread, consumed by a wait.
#[derive(Debug)]
pub struct Event {
    fd: OwnedFd,
}

impl Event {
    pub fn new() -> io::Result<Self> {
        // SAFETY: plain eventfd creation; the fd is owned below.
        let fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            // SAFETY: `fd` is a fresh descriptor we own.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
        })
    }

    pub fn set(&self) {
        let one = 1u64;
        // SAFETY: 8-byte write to a valid eventfd (EAGAIN only on overflow,
        // which still leaves it signaled).
        unsafe { libc::write(self.fd.as_raw_fd(), std::ptr::from_ref(&one).cast(), 8) };
    }

    /// Resets the event; returns whether it was signaled.
    pub fn take(&self) -> bool {
        let mut v = 0u64;
        // SAFETY: 8-byte read from a valid non-blocking eventfd.
        unsafe { libc::read(self.fd.as_raw_fd(), std::ptr::from_mut(&mut v).cast(), 8) == 8 }
    }

    pub fn raw_fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }
}

/// Absolute-deadline timer (timerfd on `CLOCK_MONOTONIC`, the host clock).
#[derive(Debug)]
pub struct Timer {
    fd: OwnedFd,
}

impl Timer {
    pub fn new() -> io::Result<Self> {
        // SAFETY: plain timerfd creation; the fd is owned below.
        let fd = unsafe {
            libc::timerfd_create(
                libc::CLOCK_MONOTONIC,
                libc::TFD_NONBLOCK | libc::TFD_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            // SAFETY: `fd` is a fresh descriptor we own.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
        })
    }

    /// Arms the timer for host time `at_ns`.
    pub fn arm(&self, at_ns: i64) -> bool {
        let at = at_ns.max(1);
        let spec = libc::itimerspec {
            it_interval: libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
            it_value: libc::timespec {
                tv_sec: (at / 1_000_000_000) as _,
                tv_nsec: (at % 1_000_000_000) as _,
            },
        };
        // SAFETY: valid timerfd and itimerspec.
        unsafe {
            libc::timerfd_settime(
                self.fd.as_raw_fd(),
                libc::TFD_TIMER_ABSTIME,
                &spec,
                std::ptr::null_mut(),
            ) == 0
        }
    }

    fn drain(&self) {
        let mut v = 0u64;
        // SAFETY: 8-byte read from a valid non-blocking timerfd.
        unsafe { libc::read(self.fd.as_raw_fd(), std::ptr::from_mut(&mut v).cast(), 8) };
    }
}

/// Waits on `fds` (readable) until one is ready or `timer` fires. Returns
/// the indices of the ready fds.
fn wait_any(timer: &Timer, fds: &[i32]) -> Vec<usize> {
    let mut pfds: Vec<libc::pollfd> = std::iter::once(timer.fd.as_raw_fd())
        .chain(fds.iter().copied())
        .map(|fd| libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        })
        .collect();
    loop {
        // SAFETY: valid pollfd array of the given length.
        let n = unsafe { libc::poll(pfds.as_mut_ptr(), pfds.len() as _, -1) };
        if n >= 0 {
            break;
        }
        if io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
            break;
        }
    }
    if pfds[0].revents & libc::POLLIN != 0 {
        timer.drain();
    }
    pfds[1..]
        .iter()
        .enumerate()
        .filter(|(_, p)| p.revents & libc::POLLIN != 0)
        .map(|(i, _)| i)
        .collect()
}

/// Waits until `event` is set or host time `deadline_ns` (sub-ms accurate);
/// consumes the event. Returns whether the event was set.
pub fn wait_event_until(timer: &Timer, event: &Event, deadline_ns: i64) -> bool {
    if event.take() {
        return true;
    }
    if !timer.arm(deadline_ns) {
        return false;
    }
    let ready = wait_any(timer, &[event.raw_fd()]);
    !ready.is_empty() && event.take()
}

#[derive(Debug)]
pub struct PreciseSleep {
    timer: Timer,
}

impl PreciseSleep {
    /// Timer wake-ups are this early; the remainder is spun.
    const SPIN_NS: i64 = 300_000;

    pub fn new() -> io::Result<Self> {
        Ok(Self {
            timer: Timer::new()?,
        })
    }

    /// Returns at `deadline_ns` (host time), or immediately if it passed.
    pub fn until(&self, deadline_ns: i64) {
        let _ = self.until_or(deadline_ns, None);
    }

    /// Like [`Self::until`], but returns `true` early when `event` is set
    /// before the final spin (the caller handles it and waits again).
    pub fn until_or(&self, deadline_ns: i64, event: Option<&Event>) -> bool {
        let left = deadline_ns - HostTime::now().0;
        if left > Self::SPIN_NS {
            let wake = deadline_ns - Self::SPIN_NS;
            match event {
                Some(e) => {
                    if wait_event_until(&self.timer, e, wake) {
                        return true;
                    }
                }
                None => {
                    if self.timer.arm(wake) {
                        let _ = wait_any(&self.timer, &[]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleeps_to_deadline_and_wakes_on_event() {
        let s = PreciseSleep::new().expect("timer");
        let t0 = HostTime::now().0;
        s.until(t0 + 3_000_000);
        let late = HostTime::now().0 - (t0 + 3_000_000);
        assert!((0..2_000_000).contains(&late), "late by {late} ns");

        let e = Event::new().expect("eventfd");
        e.set();
        let t1 = HostTime::now().0;
        assert!(s.until_or(t1 + 500_000_000, Some(&e)));
        assert!(HostTime::now().0 - t1 < 100_000_000);
        assert!(!e.take(), "event is auto-reset");
    }
}
