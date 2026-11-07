//! `serve`: the long-running host CapturePlay starts. See `nr_proto` for
//! the contract; this side owns the NGX feature, exports the semaphores and
//! the shared images, and evaluates every announced frame.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ash::vk;
use nr_proto::{FRAME_DONE, FRAME_SKIPPED, HostState, Shared};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, MapViewOfFile, PAGE_READWRITE,
};

use super::gpu::{self, Gpu, Image, Result};
use super::ngx::{Bridge, Feature, Frame, Tuning};
use super::share;

pub struct Args {
    pub bridge: PathBuf,
    pub snippet: PathBuf,
    pub shm: PathBuf,
    pub parent: Option<u32>,
}

/// Maps the client's shared-memory file (Wine maps file views shared with
/// the native process).
fn map(path: &Path) -> Result<&'static Shared> {
    use std::os::windows::io::AsRawHandle as _;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: a valid file handle; the view stays mapped for the process
    // (the mapping handle may be closed once the view exists).
    unsafe {
        let mapping = CreateFileMappingW(
            HANDLE(file.as_raw_handle()),
            None,
            PAGE_READWRITE,
            0,
            nr_proto::MAP_SIZE as u32,
            None,
        )
        .map_err(|e| format!("CreateFileMapping: {e}"))?;
        let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, nr_proto::MAP_SIZE);
        let _ = CloseHandle(mapping);
        if view.Value.is_null() {
            return Err("MapViewOfFile failed".into());
        }
        Ok(Shared::from_ptr(view.Value.cast()))
    }
}

/// The client process is gone (checked through Wine's view of /proc).
fn parent_gone(pid: Option<u32>) -> bool {
    pid.is_some_and(|p| !Path::new(&format!(r"Z:\proc\{p}")).exists())
}

/// Feature and images of one size.
struct Surfaces<'b> {
    size: (u32, u32),
    feature: Feature<'b>,
    proxy: Image,
    edited: Image,
    depth: Image,
    motion: Image,
}

fn build<'b>(gpu: &Gpu, bridge: &'b Bridge, size: (u32, u32), preset: u32) -> Result<Surfaces<'b>> {
    let export = Some(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let format = vk::Format::from_raw(nr_proto::IMAGE_FORMAT);
    let proxy = Image::new(gpu, format, size, export)?;
    let edited = Image::new(gpu, format, size, export)?;
    let depth = Image::new(gpu, vk::Format::R32_SFLOAT, size, None)?;
    let motion = Image::new(gpu, vk::Format::R16G16_SFLOAT, size, None)?;
    let mut feature = None;
    gpu.run(|cmd| {
        let d = &gpu.device;
        // SAFETY: recording command buffer; images are in GENERAL.
        unsafe {
            for (img, v) in [(&depth, [0.5, 0.0, 0.0, 0.0]), (&motion, [0.0; 4])] {
                d.cmd_clear_color_image(
                    cmd,
                    img.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { float32: v },
                    &[gpu::color_range()],
                );
            }
        }
        gpu.barrier(cmd);
        // Both shared images start with the client.
        for img in [&proxy, &edited] {
            gpu::external_barrier(d, cmd, gpu.family, img.image, false);
        }
        feature = Some(Feature::create(bridge, cmd, size, preset)?);
        Ok(())
    })?;
    Ok(Surfaces {
        size,
        feature: feature.ok_or("feature not created")?,
        proxy,
        edited,
        depth,
        motion,
    })
}

pub fn serve(a: &Args) -> Result<()> {
    let shared = map(&a.shm)?;
    if shared.magic.load(Ordering::Acquire) != nr_proto::MAGIC
        || shared.version.load(Ordering::Acquire) != nr_proto::VERSION
    {
        return Err("shared memory has another protocol version".into());
    }
    let r = run(a, shared);
    match &r {
        Ok(()) => shared.set_state(HostState::Exited),
        Err(e) => shared.fail(e),
    }
    r
}

fn run(a: &Args, shared: &Shared) -> Result<()> {
    let gpu = Gpu::new()?;
    log!("nr-host: serving on {} ({})", gpu.name, a.shm.display());
    let bridge = Bridge::load(&a.bridge, &a.snippet)?;
    bridge.init(&gpu, &std::env::temp_dir())?;

    let mut semaphores = [vk::Semaphore::null(); 3];
    for (i, s) in semaphores.iter_mut().enumerate() {
        let (sem, fd) = share::timeline_semaphore(&gpu)?;
        *s = sem;
        shared.semaphore_fds[i].store(fd, Ordering::Relaxed);
    }
    for (dst, b) in shared.device_uuid.iter().zip(gpu.uuid()) {
        dst.store(b, Ordering::Relaxed);
    }
    shared.set_state(HostState::Ready);
    log!("nr-host: ready");

    let result = frames(a, shared, &gpu, &bridge, &semaphores);
    // SAFETY: waits for every submission before destroying.
    unsafe {
        let _ = gpu.device.device_wait_idle();
        for s in semaphores {
            gpu.device.destroy_semaphore(s, None);
        }
    }
    result
}

fn frames(
    a: &Args,
    shared: &Shared,
    gpu: &Gpu,
    bridge: &Bridge,
    sem: &[vk::Semaphore; 3],
) -> Result<()> {
    let mut ring = gpu.ring(4)?;
    let mut surfaces: Option<Surfaces<'_>> = None;
    let mut built_seq = 0;
    let mut next = 1u64;
    let mut checked = Instant::now();
    loop {
        shared.heartbeat.fetch_add(1, Ordering::Relaxed);
        if shared.quit.load(Ordering::Acquire) != 0 {
            log!("nr-host: quit requested");
            break;
        }
        if checked.elapsed() > Duration::from_secs(1) {
            checked = Instant::now();
            if parent_gone(a.parent) {
                log!("nr-host: client gone");
                break;
            }
        }

        let config = shared.config_seq.load(Ordering::Acquire);
        if config != built_seq {
            built_seq = config;
            shared.set_state(HostState::Building);
            // SAFETY: drains the evaluations of the old surfaces.
            unsafe {
                let _ = gpu.device.device_wait_idle();
            }
            surfaces = None;
            let size = (
                shared.config.width.load(Ordering::Relaxed),
                shared.config.height.load(Ordering::Relaxed),
            );
            let preset = shared.config.preset.load(Ordering::Relaxed);
            let t = Instant::now();
            match build(gpu, bridge, size, preset) {
                Ok(s) => {
                    let out = &shared.surfaces;
                    out.width.store(size.0, Ordering::Relaxed);
                    out.height.store(size.1, Ordering::Relaxed);
                    out.proxy_fd
                        .store(s.proxy.fd.unwrap_or(-1), Ordering::Relaxed);
                    out.edited_fd
                        .store(s.edited.fd.unwrap_or(-1), Ordering::Relaxed);
                    out.proxy_size.store(s.proxy.size, Ordering::Relaxed);
                    out.edited_size.store(s.edited.size, Ordering::Relaxed);
                    let ms = t.elapsed().as_millis() as u32;
                    shared.build_ms.store(ms, Ordering::Relaxed);
                    log!("nr-host: feature 18 at {}x{} in {ms} ms", size.0, size.1);
                    surfaces = Some(s);
                    shared.surfaces_seq.store(config, Ordering::Release);
                    shared.set_state(HostState::Ready);
                }
                Err(e) => {
                    log!("nr-host: building {}x{} failed: {e}", size.0, size.1);
                    // Keep answering frames (as skipped) so the client never
                    // waits on a signal that does not come.
                    shared.fail(&e);
                }
            }
        }

        // Wait for the next announcement (short, to keep the checks above
        // responsive).
        let sems = [sem[nr_proto::REQUEST]];
        let values = [next];
        // SAFETY: a live semaphore.
        let r = unsafe {
            gpu.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&sems)
                    .values(&values),
                20_000_000,
            )
        };
        match r {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => continue,
            Err(e) => return Err(format!("vkWaitSemaphores(request): {e:?}")),
        }
        // SAFETY: a live semaphore.
        let upto = unsafe {
            gpu.device
                .get_semaphore_counter_value(sem[nr_proto::REQUEST])
        }
        .map_err(|e| format!("vkGetSemaphoreCounterValue: {e:?}"))?;
        for n in next..=upto {
            evaluate(shared, gpu, &mut ring, surfaces.as_ref(), sem, n)?;
        }
        next = upto + 1;
    }
    Ok(())
}

/// Submits frame `n`: the evaluation when everything matches, otherwise
/// only the semaphore hand-over.
fn evaluate(
    shared: &Shared,
    gpu: &Gpu,
    ring: &mut gpu::Ring,
    surfaces: Option<&Surfaces<'_>>,
    sem: &[vk::Semaphore; 3],
    n: u64,
) -> Result<()> {
    let slot = shared.slot(n);
    let (cmd, fence) = ring.begin()?;
    let d = &gpu.device;
    let mut status = FRAME_SKIPPED;
    if slot.seq.load(Ordering::Acquire) == n
        && let Some(s) = surfaces
    {
        gpu::external_barrier(d, cmd, gpu.family, s.proxy.image, true);
        gpu::external_barrier(d, cmd, gpu.family, s.edited.image, true);
        let tuning = Tuning {
            style: slot.style.load(Ordering::Relaxed),
            intensity: nr_proto::load_f32(&slot.intensity),
            local_structure: nr_proto::load_f32(&slot.local_structure),
            local_tone: nr_proto::load_f32(&slot.local_tone),
            skin_structure: nr_proto::load_f32(&slot.skin_structure),
            auto_mask: slot.auto_mask.load(Ordering::Relaxed) != 0,
            ui_correction: slot.ui_correction.load(Ordering::Relaxed) != 0,
        };
        let frame = Frame {
            color: &s.proxy,
            depth: &s.depth,
            motion: &s.motion,
            output: &s.edited,
            reset: slot.reset.load(Ordering::Relaxed) != 0,
            tuning,
        };
        match s.feature.evaluate(cmd, &frame) {
            Ok(()) => status = FRAME_DONE,
            Err(e) => log!("nr-host: frame {n} ({}x{}): {e}", s.size.0, s.size.1),
        }
        gpu.barrier(cmd);
        gpu::external_barrier(d, cmd, gpu.family, s.proxy.image, false);
        gpu::external_barrier(d, cmd, gpu.family, s.edited.image, false);
    }
    // SAFETY: the buffer was begun by the ring.
    unsafe { d.end_command_buffer(cmd) }.map_err(|e| format!("vkEndCommandBuffer: {e:?}"))?;
    slot.status.store(status, Ordering::Release);
    gpu.submit_timeline(
        cmd,
        (sem[nr_proto::INPUT], n),
        (sem[nr_proto::OUTPUT], n),
        fence,
    )
}
