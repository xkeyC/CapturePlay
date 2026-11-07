//! DLSS Neural Rendering (NGX feature 18) on Linux.
//!
//! NVIDIA ships the model only as a Windows DLL, so it runs in
//! `captureplay-nr-host`, a Windows program started under Proton (crate
//! `nr-host`, contract in `nr_proto`). Per unique frame, on the frame's own
//! recording:
//!
//!   working frame → sRGB proxy (shared image, `nr_encode`)
//!     → [host: feature 18 → edit (shared image)]
//!     → bounded residual composite over the working frame (`nr_composite`)
//!
//! exactly as the D3D12 enhancer does, with the model step handed over
//! through timeline semaphores the host exported: the encode signals
//! `INPUT = n`, the CPU announces `REQUEST = n`, the composite waits for
//! `OUTPUT >= n`. No CPU waits on the frame path.
//!
//! The host is started on first use and needs seconds (Proton, then the
//! feature); until its surfaces are imported frames pass through. A
//! watchdog thread keeps the queue from waiting forever: if the host dies or
//! leaves an announced frame unanswered for [`STALL`], it signals `OUTPUT`
//! itself and the enhancer gives up.

use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ash::vk;
use cp_nvidia::ngx_vk::NrTuning;
use nr_proto::{HostState, Shared};

use super::device::Gpu;
use super::pipeline::{self, Pipeline};
use super::renderer::{Enhancer, FrameInput, RecordCtx};
use super::texture::{Texture, WORKING_USAGE, barrier, external_barrier};
use super::{Result, VkError};
use crate::constants::{NrCompositeConstants, NrEncodeConstants};

/// Answer time after which a frame counts as lost.
const STALL: Duration = Duration::from_secs(2);
/// The host's log, in the Wine prefix (the host may be installed read-only).
const HOST_LOG: &str = "captureplay-nr-host.log";

/// Time the host may take to start (Proton, prefix, NGX init).
const START_TIMEOUT: Duration = Duration::from_secs(90);

/// Where and how to start the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtonNrConfig {
    /// Command running Proton's `proton` script, e.g.
    /// `["steam-run", "…/Proton 11.0/proton"]` on NixOS.
    pub proton: Vec<String>,
    /// Wine prefix (`STEAM_COMPAT_DATA_PATH`), created on first start.
    pub prefix: PathBuf,
    /// `captureplay-nr-host.exe` (with `captureplay_nvngx.dll` beside it).
    pub host_exe: PathBuf,
    /// `nvngx_dlssnr.dll`.
    pub snippet: PathBuf,
    /// `DLSSNR.Hint.Render.Preset`.
    pub preset: u32,
}

/// The shared-memory file, mapped.
struct Mapping {
    path: PathBuf,
    ptr: *mut u8,
    unlinked: bool,
}

impl Mapping {
    /// Removes files left by processes that are gone (crashed or killed
    /// before the host mapped their file).
    fn remove_stale() {
        for e in std::fs::read_dir("/dev/shm")
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = e.file_name();
            let Some(pid) = name
                .to_str()
                .and_then(|n| n.strip_prefix("captureplay-nr-"))
                .and_then(|r| r.split('-').next())
            else {
                continue;
            };
            if !Path::new("/proc").join(pid).exists() {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    /// The host mapped the file: the name is no longer needed (both
    /// mappings stay), so nothing is left behind if we die.
    fn unlink(&mut self) {
        if !self.unlinked {
            let _ = std::fs::remove_file(&self.path);
            self.unlinked = true;
        }
    }

    fn create() -> std::io::Result<Self> {
        Self::remove_stale();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = PathBuf::from(format!(
            "/dev/shm/captureplay-nr-{}-{nanos:x}",
            std::process::id()
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.set_len(nr_proto::MAP_SIZE as u64)?;
        // SAFETY: a fresh file of MAP_SIZE bytes, mapped shared; the mapping
        // outlives the fd.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                nr_proto::MAP_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let e = std::io::Error::last_os_error();
            let _ = std::fs::remove_file(&path);
            return Err(e);
        }
        let m = Self {
            path,
            ptr: ptr.cast(),
            unlinked: false,
        };
        m.shared()
            .version
            .store(nr_proto::VERSION, Ordering::Relaxed);
        m.shared().magic.store(nr_proto::MAGIC, Ordering::Release);
        Ok(m)
    }

    fn shared(&self) -> &Shared {
        // SAFETY: MAP_SIZE bytes stay mapped until drop.
        unsafe { Shared::from_ptr(self.ptr) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: our own mapping.
        unsafe { libc::munmap(self.ptr.cast(), nr_proto::MAP_SIZE) };
        self.unlink();
    }
}

// SAFETY: the mapping is plain shared memory accessed through atomics.
unsafe impl Send for Mapping {}

/// The host process.
struct Host {
    map: Mapping,
    launcher: Child,
    token: String,
    /// Linux pid and pidfd of the host, once found.
    process: Option<(i32, OwnedFd)>,
    started: Instant,
}

impl Host {
    fn spawn(cfg: &ProtonNrConfig) -> std::result::Result<Self, String> {
        let (program, args) = cfg
            .proton
            .split_first()
            .ok_or("no Proton command configured")?;
        let map = Mapping::create().map_err(|e| format!("shared memory: {e}"))?;
        let _ = std::fs::create_dir_all(&cfg.prefix);
        let token = format!(
            "cp-nr-{}-{:x}",
            std::process::id(),
            map.ptr as usize ^ Instant::now().elapsed().as_nanos() as usize
        );
        let steam = std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".local/share/Steam"))
            .filter(|p| p.is_dir())
            .unwrap_or_else(|| cfg.prefix.clone());
        let mut cmd = Command::new(program);
        cmd.args(args)
            .arg("run")
            .arg(&cfg.host_exe)
            .args(["serve", "--shm"])
            .arg(nr_proto::wine_path(&map.path))
            .arg("--snippet")
            .arg(nr_proto::wine_path(&cfg.snippet))
            .args(["--parent", &std::process::id().to_string()])
            .arg("--log")
            .arg(nr_proto::wine_path(&cfg.prefix.join(HOST_LOG)))
            .args(["--token", &token])
            .env("STEAM_COMPAT_DATA_PATH", &cfg.prefix)
            .env("STEAM_COMPAT_CLIENT_INSTALL_PATH", steam)
            .env("PROTON_ENABLE_NVAPI", "1")
            // Ours, not Proton's: libraries and layers of this process.
            .env_remove("LD_LIBRARY_PATH")
            .env_remove("VK_ADD_LAYER_PATH")
            .env_remove("VK_INSTANCE_LAYERS")
            .env_remove("VK_LOADER_LAYERS_ENABLE")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let launcher = cmd.spawn().map_err(|e| format!("{program}: {e}"))?;
        tracing::info!(
            command = ?cfg.proton,
            prefix = %cfg.prefix.display(),
            host = %cfg.host_exe.display(),
            "starting the DLSS NR host under Proton"
        );
        Ok(Self {
            map,
            launcher,
            token,
            process: None,
            started: Instant::now(),
        })
    }

    fn shared(&self) -> &Shared {
        self.map.shared()
    }

    /// Finds the host among the processes Proton started: argv[0] is the
    /// exe and our token is among its arguments (Proton's `steam.exe`
    /// wrapper carries the same arguments under another argv[0]).
    fn find_process(&mut self) {
        if self.process.is_some() {
            return;
        }
        let found = std::fs::read_dir("/proc").ok().and_then(|dir| {
            dir.flatten().find_map(|e| {
                let pid: i32 = e.file_name().to_str()?.parse().ok()?;
                let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
                let mut args = cmdline.split(|&b| b == 0);
                let exe = String::from_utf8_lossy(args.next()?);
                (exe.ends_with("captureplay-nr-host.exe")
                    && args.any(|a| a == self.token.as_bytes()))
                .then_some(pid)
            })
        });
        if let Some(pid) = found {
            // SAFETY: plain syscall; the fd is owned below.
            let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
            if fd >= 0 {
                // SAFETY: a fresh descriptor we own.
                self.process = Some((pid, unsafe { OwnedFd::from_raw_fd(fd) }));
            }
        }
    }

    /// Our own descriptor for the host's `fd`.
    fn adopt(&self, fd: i32) -> std::result::Result<OwnedFd, String> {
        let (pid, pidfd) = self.process.as_ref().ok_or("host process not found")?;
        // SAFETY: plain syscall on a pidfd we own.
        let ours = unsafe { libc::syscall(libc::SYS_pidfd_getfd, pidfd.as_raw_fd(), fd, 0) } as i32;
        if ours < 0 {
            return Err(format!(
                "pidfd_getfd({pid}, {fd}): {}",
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: a fresh descriptor we own.
        Ok(unsafe { OwnedFd::from_raw_fd(ours) })
    }
}

/// Whether the process behind `pidfd` has exited.
fn exited(pidfd: i32) -> bool {
    let mut p = libc::pollfd {
        fd: pidfd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid pollfd, no waiting.
    unsafe { libc::poll(&mut p, 1, 0) > 0 }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.shared().quit.store(1, Ordering::Release);
        if let Some((_, pidfd)) = &self.process {
            let deadline = Instant::now() + Duration::from_secs(2);
            while !exited(pidfd.as_raw_fd()) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            if !exited(pidfd.as_raw_fd()) {
                // SAFETY: plain syscall on a pidfd we own.
                unsafe {
                    libc::syscall(
                        libc::SYS_pidfd_send_signal,
                        pidfd.as_raw_fd(),
                        libc::SIGKILL,
                        std::ptr::null::<libc::siginfo_t>(),
                        0,
                    );
                }
            }
        }
        let _ = self.launcher.kill();
        let _ = self.launcher.wait();
    }
}

/// Imported semaphores and the watchdog guarding `OUTPUT`.
struct Link {
    device: ash::Device,
    semaphores: [vk::Semaphore; 3],
    watch: Arc<Watch>,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct Watch {
    device: ash::Device,
    output: vk::Semaphore,
    /// Last frame announced to the host.
    requested: AtomicU64,
    failed: AtomicBool,
    stop: AtomicBool,
    /// Host pidfd (raw; owned by `Host`, which outlives the watchdog).
    pidfd: i32,
}

impl Watch {
    /// Signals `OUTPUT` up to the last request on the host's behalf.
    fn release_pending(&self) -> bool {
        let requested = self.requested.load(Ordering::Acquire);
        // SAFETY: a live imported semaphore.
        let done = unsafe { self.device.get_semaphore_counter_value(self.output) }.unwrap_or(0);
        if done >= requested {
            return false;
        }
        // SAFETY: as above; the value is above the current one.
        let _ = unsafe {
            self.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.output)
                    .value(requested),
            )
        };
        true
    }

    fn run(&self) {
        let mut pending_since: Option<(u64, Instant)> = None;
        while !self.stop.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(20));
            if self.failed.load(Ordering::Acquire) {
                continue;
            }
            let requested = self.requested.load(Ordering::Acquire);
            // SAFETY: a live imported semaphore.
            let done = unsafe { self.device.get_semaphore_counter_value(self.output) }.unwrap_or(0);
            let stalled = match pending_since {
                _ if done >= requested => {
                    pending_since = None;
                    false
                }
                Some((n, t)) if n == done => t.elapsed() > STALL,
                _ => {
                    pending_since = Some((done, Instant::now()));
                    false
                }
            };
            if stalled || exited(self.pidfd) {
                tracing::error!(
                    stalled,
                    requested,
                    done,
                    "DLSS NR host stopped answering; releasing the queue"
                );
                self.failed.store(true, Ordering::Release);
                self.release_pending();
            }
        }
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        self.watch.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        // Nothing may wait on the host any more.
        self.watch.release_pending();
        // SAFETY: the owner waited for the GPU; the semaphores are ours.
        unsafe {
            let _ = self.device.device_wait_idle();
            for s in self.semaphores {
                self.device.destroy_semaphore(s, None);
            }
        }
    }
}

fn link(gpu: &Gpu, host: &Host) -> std::result::Result<Link, String> {
    let shared = host.shared();
    let uuid: Vec<u8> = shared
        .device_uuid
        .iter()
        .map(|b| b.load(Ordering::Relaxed))
        .collect();
    if uuid != gpu.device_uuid {
        return Err("the host runs on another GPU".into());
    }
    let ext = gpu
        .external_semaphore_fn
        .as_ref()
        .ok_or("VK_KHR_external_semaphore_fd unavailable")?;
    let mut semaphores = [vk::Semaphore::null(); 3];
    for (i, s) in semaphores.iter_mut().enumerate() {
        let fd = host.adopt(shared.semaphore_fds[i].load(Ordering::Relaxed))?;
        let mut timeline =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        // SAFETY: plain creation; a successful import takes the fd.
        unsafe {
            *s = gpu
                .device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )
                .map_err(|e| format!("vkCreateSemaphore: {e:?}"))?;
            ext.import_semaphore_fd(
                &vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(*s)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                    .fd(fd.as_raw_fd()),
            )
            .map_err(|e| format!("vkImportSemaphoreFdKHR: {e:?}"))?;
        }
        std::mem::forget(fd);
    }
    // SAFETY: a live imported semaphore.
    let done = unsafe {
        gpu.device
            .get_semaphore_counter_value(semaphores[nr_proto::OUTPUT])
    }
    .unwrap_or(0);
    let watch = Arc::new(Watch {
        device: gpu.device.clone(),
        output: semaphores[nr_proto::OUTPUT],
        requested: AtomicU64::new(done),
        failed: AtomicBool::new(false),
        stop: AtomicBool::new(false),
        pidfd: host.process.as_ref().map_or(-1, |(_, fd)| fd.as_raw_fd()),
    });
    let w = watch.clone();
    let thread = std::thread::Builder::new()
        .name("dlss-nr-watch".into())
        .spawn(move || w.run())
        .map_err(|e| e.to_string())?;
    Ok(Link {
        device: gpu.device.clone(),
        semaphores,
        watch,
        thread: Some(thread),
    })
}

/// Shared images of one size plus the composite target.
struct Surfaces {
    size: (u32, u32),
    proxy: Texture,
    edited: Texture,
    composed: Texture,
}

pub struct ProtonNrEnhancer {
    encode: Pipeline,
    composite: Pipeline,
    cfg: ProtonNrConfig,
    tuning: NrTuning,
    detail: f32,
    colour: f32,
    hdr: bool,
    sdr_white: f32,
    // Drop order: surfaces and link (GPU objects, watchdog) before the host.
    surfaces: Option<Surfaces>,
    link: Option<Link>,
    host: Option<Host>,
    /// Size requested from the host, and its config sequence.
    wanted: Option<((u32, u32), u32)>,
    /// Last frame announced.
    frame: u64,
    failed: bool,
    reported: Option<HostState>,
}

impl std::fmt::Debug for ProtonNrEnhancer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProtonNrEnhancer")
            .field("cfg", &self.cfg)
            .field("ready", &self.surfaces.is_some())
            .finish_non_exhaustive()
    }
}

impl ProtonNrEnhancer {
    pub fn new(
        gpu: &Gpu,
        layouts: &pipeline::Layouts,
        cfg: ProtonNrConfig,
        tuning: NrTuning,
        detail: f32,
    ) -> Result<Self> {
        if gpu.external_memory_fn.is_none() || gpu.external_semaphore_fn.is_none() {
            return Err(VkError::Other(
                "DLSS NR needs VK_KHR_external_memory_fd and VK_KHR_external_semaphore_fd".into(),
            ));
        }
        for (what, p) in [("host", &cfg.host_exe), ("model", &cfg.snippet)] {
            if !p.is_file() {
                return Err(VkError::Other(format!(
                    "DLSS NR {what} not found: {}",
                    p.display()
                )));
            }
        }
        Ok(Self {
            encode: Pipeline::compute(gpu, layouts, pipeline::NR_ENCODE)?,
            composite: Pipeline::compute(gpu, layouts, pipeline::NR_COMPOSITE)?,
            cfg,
            tuning,
            detail,
            colour: 1.0,
            hdr: false,
            sdr_white: 1.0,
            surfaces: None,
            link: None,
            host: None,
            wanted: None,
            frame: 0,
            failed: false,
            reported: None,
        })
    }

    /// `INPUT` and `OUTPUT` counter values (diagnostics).
    pub fn progress(&self) -> Option<(u64, u64, u64)> {
        let l = self.link.as_ref()?;
        // SAFETY: live imported semaphores.
        let value =
            |i: usize| unsafe { l.device.get_semaphore_counter_value(l.semaphores[i]) }.ok();
        Some((
            self.frame,
            value(nr_proto::INPUT)?,
            value(nr_proto::OUTPUT)?,
        ))
    }

    /// The backend gave up (host failed, stalled or could not start).
    pub fn failed(&self) -> bool {
        self.failed
    }

    fn give_up(&mut self, why: &str) {
        tracing::warn!("DLSS NR unavailable: {why}");
        self.failed = true;
    }

    /// Non-blocking: starts the host, links up, asks for and imports the
    /// surfaces of `size`. False means "pass the frame through for now".
    fn prepare(&mut self, ctx: &mut RecordCtx<'_>, size: (u32, u32)) -> bool {
        if self.failed {
            return false;
        }
        if self
            .link
            .as_ref()
            .is_some_and(|l| l.watch.failed.load(Ordering::Acquire))
        {
            self.give_up("the host stopped answering");
            return false;
        }
        if self.host.is_none() {
            match Host::spawn(&self.cfg) {
                Ok(h) => self.host = Some(h),
                Err(e) => {
                    self.give_up(&e);
                    return false;
                }
            }
        }
        let Some(host) = self.host.as_mut() else {
            return false;
        };
        host.find_process();
        let state = host.shared().state();
        if self.reported != Some(state) {
            tracing::info!(?state, "DLSS NR host");
            self.reported = Some(state);
        }
        match state {
            HostState::Failed | HostState::Exited => {
                let e = host.shared().error();
                self.give_up(&format!("host: {e}"));
                return false;
            }
            HostState::Starting => {
                if host.started.elapsed() > START_TIMEOUT {
                    self.give_up(&format!(
                        "the host did not start (see {})",
                        self.cfg.prefix.join(HOST_LOG).display()
                    ));
                } else if host
                    .process
                    .as_ref()
                    .is_some_and(|(_, fd)| exited(fd.as_raw_fd()))
                {
                    self.give_up("the host exited during start-up");
                }
                return false;
            }
            HostState::Ready | HostState::Building => host.map.unlink(),
        }
        if self.link.is_none() {
            if host.process.is_none() {
                return false;
            }
            match link(ctx.gpu, host) {
                Ok(l) => self.link = Some(l),
                Err(e) => {
                    self.give_up(&e);
                    return false;
                }
            }
        }
        let shared = host.shared();
        if self.wanted.is_none_or(|(s, _)| s != size) {
            if let Some(old) = self.surfaces.take() {
                ctx.rec.retire.extend([old.proxy, old.edited, old.composed]);
            }
            let seq = shared.config_seq.load(Ordering::Relaxed) + 1;
            shared.config.width.store(size.0, Ordering::Relaxed);
            shared.config.height.store(size.1, Ordering::Relaxed);
            shared
                .config
                .preset
                .store(self.cfg.preset, Ordering::Relaxed);
            shared.config_seq.store(seq, Ordering::Release);
            self.wanted = Some((size, seq));
            tracing::info!(w = size.0, h = size.1, "DLSS NR: requesting surfaces");
            return false;
        }
        if self.surfaces.is_none() {
            let seq = self.wanted.map_or(0, |(_, s)| s);
            if shared.surfaces_seq.load(Ordering::Acquire) != seq || state != HostState::Ready {
                return false;
            }
            match import_surfaces(ctx, host, size) {
                Ok(s) => {
                    tracing::info!(
                        w = size.0,
                        h = size.1,
                        build_ms = shared.build_ms.load(Ordering::Relaxed),
                        "DLSS Neural Rendering ready (Proton host)"
                    );
                    self.surfaces = Some(s);
                }
                Err(e) => {
                    self.give_up(&format!("importing the host's images: {e}"));
                    return false;
                }
            }
        }
        true
    }
}

fn import_surfaces(
    ctx: &mut RecordCtx<'_>,
    host: &Host,
    size: (u32, u32),
) -> std::result::Result<Surfaces, String> {
    let s = &host.shared().surfaces;
    if (
        s.width.load(Ordering::Relaxed),
        s.height.load(Ordering::Relaxed),
    ) != size
    {
        return Err("size mismatch".into());
    }
    let format = vk::Format::from_raw(nr_proto::IMAGE_FORMAT);
    let usage = vk::ImageUsageFlags::from_raw(nr_proto::IMAGE_USAGE);
    let import = |ctx: &mut RecordCtx<'_>, fd: i32, bytes: u64| {
        let fd = host.adopt(fd)?;
        Texture::import_opaque_fd(ctx.gpu, ctx.cmd(), size.0, size.1, format, usage, fd, bytes)
            .map_err(|e| e.to_string())
    };
    let proxy = import(
        ctx,
        s.proxy_fd.load(Ordering::Relaxed),
        s.proxy_size.load(Ordering::Relaxed),
    )?;
    let edited = import(
        ctx,
        s.edited_fd.load(Ordering::Relaxed),
        s.edited_size.load(Ordering::Relaxed),
    )?;
    let composed = Texture::new(ctx.gpu, ctx.cmd(), size.0, size.1, format, WORKING_USAGE)
        .map_err(|e| e.to_string())?;
    Ok(Surfaces {
        size,
        proxy,
        edited,
        composed,
    })
}

/// One compute pass with the shared layout: `srcs` sampled, `dst` storage.
fn pass<T: bytemuck::Pod>(
    ctx: &mut RecordCtx<'_>,
    p: &Pipeline,
    constants: &T,
    srcs: &[&Texture],
    dst: &Texture,
) -> Result<()> {
    let cbuf = ctx
        .rec
        .arena
        .constants(constants, ctx.gpu.min_uniform_alignment)
        .ok_or_else(|| VkError::Other("frame arena exhausted".into()))?;
    let cbufs = [cbuf];
    let sampled: Vec<[vk::DescriptorImageInfo; 1]> =
        srcs.iter().map(|t| pipeline::sampled(t.view)).collect();
    let storage = pipeline::sampled(dst.view);
    let sampler = ctx.layouts.sampler_info();
    let mut writes: Vec<vk::WriteDescriptorSet<'_>> = sampled
        .iter()
        .enumerate()
        .map(|(i, info)| {
            pipeline::image_write(
                pipeline::BINDING_SRV + i as u32,
                vk::DescriptorType::SAMPLED_IMAGE,
                info,
            )
        })
        .collect();
    writes.push(pipeline::image_write(
        pipeline::BINDING_UAV,
        vk::DescriptorType::STORAGE_IMAGE,
        &storage,
    ));
    writes.push(pipeline::sampler_write(&sampler));
    writes.push(pipeline::constants_write(&cbufs));
    // SAFETY: recording into the frame's open command buffer; descriptors
    // reference live views and the frame arena.
    unsafe {
        ctx.gpu
            .device
            .cmd_bind_pipeline(ctx.cmd(), vk::PipelineBindPoint::COMPUTE, p.pipeline);
        ctx.gpu.push_fn.cmd_push_descriptor_set(
            ctx.cmd(),
            vk::PipelineBindPoint::COMPUTE,
            ctx.layouts.compute,
            0,
            &writes,
        );
    }
    pipeline::dispatch_2d(&ctx.gpu.device, ctx.cmd(), dst.width, dst.height);
    Ok(())
}

impl Enhancer for ProtonNrEnhancer {
    fn name(&self) -> &'static str {
        "NVIDIA DLSS NR (Proton)"
    }

    fn ready(&self) -> bool {
        !self.failed && self.surfaces.is_some()
    }

    fn set_strength(&mut self, strength: f32) {
        self.detail = strength;
    }

    fn set_nr_tuning(&mut self, tuning: NrTuning) {
        self.tuning = tuning;
    }

    fn set_content(&mut self, hdr: bool, sdr_white: f32) {
        self.hdr = hdr;
        self.sdr_white = sdr_white.max(1e-3);
    }

    fn enhance(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        frame: FrameInput<'_>,
        reset: bool,
    ) -> Result<()> {
        let size = (frame.tex.width, frame.tex.height);
        if size.0 == 0 || size.1 == 0 || !self.prepare(ctx, size) {
            return Ok(());
        }
        let (Some(s), Some(link), Some(host)) = (
            self.surfaces.as_ref(),
            self.link.as_ref(),
            self.host.as_ref(),
        ) else {
            return Ok(());
        };
        debug_assert_eq!(s.size, size);
        let d = &ctx.gpu.device;
        let family = ctx.gpu.queue_family;
        let n = self.frame + 1;
        let sem = link.semaphores;

        // The frame's model parameters, before the announcement.
        let slot = host.shared().slot(n);
        let t = self.tuning;
        slot.status
            .store(nr_proto::FRAME_PENDING, Ordering::Relaxed);
        slot.reset.store(u32::from(reset), Ordering::Relaxed);
        slot.style.store(t.style, Ordering::Relaxed);
        slot.auto_mask
            .store(u32::from(t.auto_mask), Ordering::Relaxed);
        slot.ui_correction
            .store(u32::from(t.ui_correction), Ordering::Relaxed);
        nr_proto::store_f32(&slot.intensity, t.intensity);
        nr_proto::store_f32(&slot.local_structure, t.local_structure);
        nr_proto::store_f32(&slot.local_tone, t.local_tone);
        nr_proto::store_f32(&slot.skin_structure, t.skin_structure);
        slot.seq.store(n, Ordering::Release);

        // 1. Working frame → display-referred sRGB proxy, handed to the host.
        external_barrier(d, ctx.cmd(), family, s.proxy.image, true);
        barrier(d, ctx.cmd());
        let encode = NrEncodeConstants {
            width: size.0,
            height: size.1,
            hdr: u32::from(self.hdr),
            sdr_white: self.sdr_white,
        };
        pass(ctx, &self.encode, &encode, &[frame.tex], &s.proxy)?;
        barrier(d, ctx.cmd());
        external_barrier(d, ctx.cmd(), family, s.proxy.image, false);
        ctx.submit_and_continue(&[(sem[nr_proto::INPUT], n)])?;

        // 2. The model, in the host.
        link.watch.requested.store(n, Ordering::Release);
        // SAFETY: a live imported semaphore; `n` is above its value.
        unsafe {
            d.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(sem[nr_proto::REQUEST])
                    .value(n),
            )
        }?;
        ctx.wait_next(sem[nr_proto::OUTPUT], n);
        self.frame = n;

        // 3. Bounded residual composite over the untouched linear frame.
        external_barrier(d, ctx.cmd(), family, s.edited.image, true);
        barrier(d, ctx.cmd());
        let composite = NrCompositeConstants {
            width: size.0,
            height: size.1,
            hdr: u32::from(self.hdr),
            sdr_white: self.sdr_white,
            detail: self.detail,
            colour: self.colour,
            _pad: [0.0; 2],
        };
        pass(
            ctx,
            &self.composite,
            &composite,
            &[frame.tex, &s.edited],
            &s.composed,
        )?;
        barrier(d, ctx.cmd());
        external_barrier(d, ctx.cmd(), family, s.edited.image, false);
        // SAFETY: recording; both images are in GENERAL and the same size.
        unsafe {
            d.cmd_copy_image(
                ctx.cmd(),
                s.composed.image,
                vk::ImageLayout::GENERAL,
                frame.tex.image,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy::default()
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .extent(vk::Extent3D {
                        width: size.0,
                        height: size.1,
                        depth: 1,
                    })],
            );
        }
        barrier(d, ctx.cmd());
        Ok(())
    }
}

/// Where the host and model are looked up and how Proton is found when the
/// settings do not say: helpers for the render thread.
pub mod discover {
    use super::{Path, PathBuf};

    /// `proton` scripts of installed Proton builds, newest name first:
    /// Steam's compatibility tools, then Steam's own Proton installs.
    pub fn proton_scripts() -> Vec<PathBuf> {
        let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
            return Vec::new();
        };
        let steam = home.join(".local/share/Steam");
        let mut found = Vec::new();
        for (dir, prefix) in [
            (steam.join("compatibilitytools.d"), ""),
            (steam.join("steamapps/common"), "Proton"),
        ] {
            let mut here: Vec<PathBuf> = std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
                .map(|e| e.path().join("proton"))
                .filter(|p| p.is_file())
                .collect();
            here.sort();
            here.reverse();
            found.extend(here);
        }
        found
    }

    /// The command to run `proton` with: NixOS needs Steam's FHS
    /// environment (`steam-run`) around it.
    pub fn proton_command(script: &Path) -> Vec<String> {
        let nixos = Path::new("/etc/NIXOS").exists();
        let steam_run = std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|d| d.join("steam-run"))
                .find(|p| p.is_file())
        });
        let mut cmd = Vec::new();
        if nixos && let Some(sr) = steam_run {
            cmd.push(sr.display().to_string());
        }
        cmd.push(script.display().to_string());
        cmd
    }
}
