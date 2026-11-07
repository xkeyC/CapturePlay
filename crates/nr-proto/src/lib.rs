//! The contract between CapturePlay (native Linux) and `captureplay-nr-host`
//! (a Windows program under Proton that runs DLSS Neural Rendering).
//!
//! # Objects
//! The host creates and exports everything that crosses the boundary — under
//! Wine only it can export, and only as Linux file descriptors the client
//! takes with `pidfd_getfd`:
//! - three timeline semaphores (fds in [`Shared::semaphore_fds`]):
//!   [`REQUEST`] is signalled by the client's *host* (CPU) when frame `n` is
//!   announced, [`INPUT`] by the client's GPU when the proxy of frame `n` is
//!   written, [`OUTPUT`] by the host's GPU when the edit of frame `n` is
//!   written;
//! - two images per surface size ([`Surfaces`]): the proxy the model reads
//!   and its edit, both [`IMAGE_FORMAT`] with [`IMAGE_USAGE`], optimal
//!   tiling, dedicated allocations, left in `GENERAL` layout. Each side
//!   creates an identical image on the shared memory and hands it over with
//!   queue-family-external release/acquire barriers.
//!
//! # Frame `n`
//! client: write [`FrameParams`] into slot `n % FRAME_SLOTS`, submit the
//! encode signalling `INPUT = n`, signal `REQUEST = n` on the CPU, make the
//! composite wait for `OUTPUT >= n`.
//! host: wake on `REQUEST >= n`, submit the evaluation waiting for
//! `INPUT >= n` and signalling `OUTPUT = n`. Every announced frame gets its
//! `OUTPUT` signal, also when the host could not evaluate it (then
//! [`FrameParams::status`] says so), so the client's queue never stalls.
//!
//! # Shared memory
//! A file in `/dev/shm` created by the client, mapped by both ([`Shared`],
//! all fields atomics; floats as bits). Both sides are x86-64, so the
//! `repr(C)` layout is the same.

use std::sync::atomic::{AtomicI32, AtomicU8, AtomicU32, AtomicU64, Ordering};

pub const MAGIC: u32 = u32::from_le_bytes(*b"CPNR");
pub const VERSION: u32 = 1;
/// Bytes to map (≥ `size_of::<Shared>()`, one page multiple).
pub const MAP_SIZE: usize = 16 * 1024;
pub const FRAME_SLOTS: usize = 8;

/// Semaphore indices in [`Shared::semaphore_fds`].
pub const REQUEST: usize = 0;
pub const INPUT: usize = 1;
pub const OUTPUT: usize = 2;

/// `VK_FORMAT_R16G16B16A16_SFLOAT`.
pub const IMAGE_FORMAT: i32 = 97;
/// `SAMPLED | STORAGE | TRANSFER_SRC | TRANSFER_DST`.
pub const IMAGE_USAGE: u32 = 0x4 | 0x8 | 0x1 | 0x2;

/// [`Shared::state`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostState {
    Starting = 0,
    /// Semaphores published; waiting for a size or for frames.
    Ready = 1,
    /// Creating the feature and surfaces for a new size.
    Building = 2,
    /// Gave up; see [`Shared::error`].
    Failed = 3,
    Exited = 4,
}

impl HostState {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Ready,
            2 => Self::Building,
            3 => Self::Failed,
            4 => Self::Exited,
            _ => Self::Starting,
        }
    }
}

/// [`FrameParams::status`].
pub const FRAME_PENDING: u32 = 0;
pub const FRAME_DONE: u32 = 1;
pub const FRAME_SKIPPED: u32 = 2;

/// Size and model choice the client asks for.
#[repr(C)]
#[derive(Debug, Default)]
pub struct Config {
    pub width: AtomicU32,
    pub height: AtomicU32,
    /// `DLSSNR.Hint.Render.Preset`.
    pub preset: AtomicU32,
}

/// Images the host published for a [`Config`].
#[repr(C)]
#[derive(Debug, Default)]
pub struct Surfaces {
    pub width: AtomicU32,
    pub height: AtomicU32,
    pub proxy_fd: AtomicI32,
    pub edited_fd: AtomicI32,
    pub proxy_size: AtomicU64,
    pub edited_size: AtomicU64,
}

/// One frame's evaluation parameters.
#[repr(C)]
#[derive(Debug, Default)]
pub struct FrameParams {
    /// The frame number these parameters belong to.
    pub seq: AtomicU64,
    pub status: AtomicU32,
    pub reset: AtomicU32,
    pub style: AtomicU32,
    pub auto_mask: AtomicU32,
    pub ui_correction: AtomicU32,
    pub intensity: AtomicU32,
    pub local_structure: AtomicU32,
    pub local_tone: AtomicU32,
    pub skin_structure: AtomicU32,
}

/// The mapped block.
#[repr(C)]
#[derive(Debug)]
pub struct Shared {
    pub magic: AtomicU32,
    pub version: AtomicU32,
    /// [`HostState`], written by the host.
    pub state: AtomicU32,
    /// Set by the client to make the host exit.
    pub quit: AtomicU32,
    /// Incremented by the host's loop.
    pub heartbeat: AtomicU64,
    /// Linux fds in the host process, valid once `state >= Ready`.
    pub semaphore_fds: [AtomicI32; 3],
    /// `VkPhysicalDeviceIDProperties::deviceUUID` of the host's device.
    pub device_uuid: [AtomicU8; 16],
    /// Bumped by the client after writing [`Self::config`].
    pub config_seq: AtomicU32,
    pub config: Config,
    /// Set to the `config_seq` the host built [`Self::surfaces`] for.
    pub surfaces_seq: AtomicU32,
    pub surfaces: Surfaces,
    pub frames: [FrameParams; FRAME_SLOTS],
    /// Milliseconds the last feature creation took.
    pub build_ms: AtomicU32,
    /// GPU-free time of the host's last evaluation submit+wait (µs, 0 when
    /// not measured).
    pub eval_us: AtomicU32,
    pub error_len: AtomicU32,
    pub error: [AtomicU8; 240],
}

const _: () = assert!(std::mem::size_of::<Shared>() <= MAP_SIZE);

impl Shared {
    /// # Safety
    /// `ptr` must point to at least [`MAP_SIZE`] mapped, writable, suitably
    /// aligned bytes that stay mapped for `'a`.
    pub unsafe fn from_ptr<'a>(ptr: *mut u8) -> &'a Self {
        // SAFETY: per the contract; every field is an atomic, valid for any
        // bit pattern (zeroed when the file was created).
        unsafe { &*ptr.cast::<Self>() }
    }

    pub fn state(&self) -> HostState {
        HostState::from_u32(self.state.load(Ordering::Acquire))
    }

    pub fn set_state(&self, s: HostState) {
        self.state.store(s as u32, Ordering::Release);
    }

    /// Records `message` and enters [`HostState::Failed`].
    pub fn fail(&self, message: &str) {
        let bytes = message.as_bytes();
        let n = bytes.len().min(self.error.len());
        for (dst, &b) in self.error.iter().zip(&bytes[..n]) {
            dst.store(b, Ordering::Relaxed);
        }
        self.error_len.store(n as u32, Ordering::Release);
        self.set_state(HostState::Failed);
    }

    pub fn error(&self) -> String {
        let n = (self.error_len.load(Ordering::Acquire) as usize).min(self.error.len());
        let bytes: Vec<u8> = self.error[..n]
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn slot(&self, n: u64) -> &FrameParams {
        &self.frames[(n % FRAME_SLOTS as u64) as usize]
    }
}

/// `f32` through an [`AtomicU32`].
pub fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

pub fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

/// Windows path of a Linux path, through Wine's `Z:` drive.
pub fn wine_path(path: &std::path::Path) -> String {
    format!("Z:{}", path.display().to_string().replace('/', "\\"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_fits_and_errors_round_trip() {
        let mut buf = vec![0u64; MAP_SIZE / 8];
        // SAFETY: the buffer is MAP_SIZE bytes, 8-aligned and outlives `s`.
        let s = unsafe { Shared::from_ptr(buf.as_mut_ptr().cast()) };
        assert_eq!(s.state(), HostState::Starting);
        s.fail("snippet missing");
        assert_eq!(s.state(), HostState::Failed);
        assert_eq!(s.error(), "snippet missing");
        assert!(std::ptr::eq(s.slot(9), &s.frames[1]));
        assert_eq!(
            wine_path(std::path::Path::new("/dev/shm/x")),
            r"Z:\dev\shm\x"
        );
    }
}
