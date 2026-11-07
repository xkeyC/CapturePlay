//! The host program:
//! - `serve` ([`serve`]) is what CapturePlay runs;
//! - `probe` checks every link of the chain on this machine: winevulkan's
//!   extensions, snippet init, feature creation, one evaluated frame and the
//!   export of its output as a Linux file descriptor.

/// Writes a line to stderr and to the log file: `--log`, else
/// `captureplay-nr-host.log` beside the executable (Proton does not pass a
/// program's stderr on; installed read-only, the client names a file).
macro_rules! log {
    ($($arg:tt)*) => {
        $crate::host::log_line(&format!($($arg)*))
    };
}

mod gpu;
mod ngx;
mod params;
mod serve;
mod share;

/// The `--log` file, set while parsing the arguments.
static LOG_PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

pub fn log_line(line: &str) {
    use std::io::Write as _;
    use std::sync::Mutex;
    static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);
    eprintln!("{line}");
    let mut f = FILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if f.is_none() {
        let path = LOG_PATH.get().cloned().or_else(|| {
            std::env::current_exe()
                .ok()
                .map(|p| p.with_extension("log"))
        });
        *f = path.and_then(|p| std::fs::File::create(p).ok());
    }
    if let Some(f) = f.as_mut() {
        let _ = writeln!(f, "{line}");
    }
}

use std::path::PathBuf;
use std::time::Instant;

use ash::vk;

use gpu::{Gpu, Image, Result, Staging};
use ngx::{Bridge, Feature, Frame, Tuning};

struct Args {
    bridge: PathBuf,
    snippet: PathBuf,
    size: (u32, u32),
    frames: u32,
    hold_s: u64,
}

enum Command {
    Probe(Args),
    Serve(serve::Args),
}

const USAGE: &str = "usage: captureplay-nr-host probe --snippet <nvngx_dlssnr.dll> \
    [--bridge <captureplay_nvngx.dll>] [--size WxH] [--frames N] [--hold SECONDS]\n\
    or: captureplay-nr-host serve --snippet <dll> --shm <file> [--bridge <dll>] [--parent PID]";

fn parse() -> Result<Command> {
    let mut a = std::env::args().skip(1);
    let command = a.next().ok_or(USAGE)?;
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_default();
    let mut args = Args {
        bridge: exe_dir.join("captureplay_nvngx.dll"),
        snippet: PathBuf::new(),
        size: (1280, 720),
        frames: 30,
        hold_s: 0,
    };
    let (mut shm, mut parent) = (PathBuf::new(), None);
    while let Some(k) = a.next() {
        let v = a.next().ok_or_else(|| format!("{k} needs a value"))?;
        match k.as_str() {
            "--bridge" => args.bridge = v.into(),
            "--snippet" => args.snippet = v.into(),
            "--size" => {
                let (w, h) = v.split_once('x').ok_or("--size WxH")?;
                args.size = (
                    w.parse().map_err(|_| "--size WxH")?,
                    h.parse().map_err(|_| "--size WxH")?,
                );
            }
            "--frames" => args.frames = v.parse().map_err(|_| "--frames N")?,
            "--hold" => args.hold_s = v.parse().map_err(|_| "--hold SECONDS")?,
            "--shm" => shm = v.into(),
            "--parent" => parent = Some(v.parse().map_err(|_| "--parent PID")?),
            // Lets the native side find this process; otherwise unused.
            "--token" => {}
            "--log" => {
                let _ = LOG_PATH.set(v.into());
            }
            _ => return Err(format!("unknown option {k}\n{USAGE}")),
        }
    }
    if args.snippet.as_os_str().is_empty() {
        return Err("--snippet is required".into());
    }
    match command.as_str() {
        "probe" => Ok(Command::Probe(args)),
        "serve" if !shm.as_os_str().is_empty() => Ok(Command::Serve(serve::Args {
            bridge: args.bridge,
            snippet: args.snippet,
            shm,
            parent,
        })),
        _ => Err(USAGE.into()),
    }
}

pub fn main() -> i32 {
    let r = parse().and_then(|c| match c {
        Command::Probe(a) => probe(&a),
        Command::Serve(a) => serve::serve(&a),
    });
    match r {
        Ok(()) => 0,
        Err(e) => {
            log!("nr-host: FAILED: {e}");
            1
        }
    }
}

/// IEEE half from f32 (normal range and zero; enough for 0..1 pixels).
fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

fn f32_of(h: u16) -> f32 {
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let sign = u32::from(h & 0x8000) << 16;
    let bits = match exp {
        0 => return if sign != 0 { -0.0 } else { 0.0 },
        31 => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// A synthetic frame with smooth gradients, edges and fine texture
/// (sRGB-encoded 0..1, like the Windows build's proxy).
fn pattern(w: u32, h: u32) -> Vec<u16> {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let checker = if ((x / 40) + (y / 40)) % 2 == 0 {
                0.15
            } else {
                0.0
            };
            let ripple = 0.1 * ((x as f32 * 0.35).sin() * (y as f32 * 0.21).cos());
            let disc = if (u - 0.5).hypot(v - 0.5) < 0.2 {
                0.3
            } else {
                0.0
            };
            let r = (0.2 + 0.6 * u + checker + ripple).clamp(0.0, 1.0);
            let g = (0.2 + 0.5 * v + disc + ripple).clamp(0.0, 1.0);
            let b = (0.6 - 0.4 * u + checker).clamp(0.0, 1.0);
            px.extend([f16(r), f16(g), f16(b), f16(1.0)]);
        }
    }
    px
}

fn probe(a: &Args) -> Result<()> {
    let gpu = Gpu::new()?;
    log!("nr-host: device {} queue family {}", gpu.name, gpu.family);
    for e in gpu::SHARING_EXTENSIONS {
        log!(
            "nr-host:   {:<34} {}",
            e.to_string_lossy(),
            if gpu.has(e) { "yes" } else { "no" }
        );
    }

    let bridge = Bridge::load(&a.bridge, &a.snippet)?;
    let data_dir = std::env::temp_dir();
    let (app, version) = bridge.init(&gpu, &data_dir)?;
    log!("nr-host: snippet initialised (app {app:#x}, version {version:#x})");

    let size = a.size;
    let export = gpu
        .has(c"VK_KHR_external_memory_win32")
        .then_some(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let color = Image::new(&gpu, vk::Format::R16G16B16A16_SFLOAT, size, None)?;
    let depth = Image::new(&gpu, vk::Format::R32_SFLOAT, size, None)?;
    let motion = Image::new(&gpu, vk::Format::R16G16_SFLOAT, size, None)?;
    let output = Image::new(&gpu, vk::Format::R16G16B16A16_SFLOAT, size, export)?;

    let input = pattern(size.0, size.1);
    let bytes = input.len() * 2;
    let mut upload = Staging::new(&gpu, bytes)?;
    upload.bytes_mut().copy_from_slice(
        &input
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let readback = Staging::new(&gpu, bytes)?;
    let d = &gpu.device;
    gpu.run(|cmd| {
        // SAFETY: recording command buffer; images are in GENERAL.
        unsafe {
            d.cmd_copy_buffer_to_image(
                cmd,
                upload.buffer,
                color.image,
                vk::ImageLayout::GENERAL,
                &[color.region()],
            );
            let clear = |img: &Image, v: [f32; 4]| {
                d.cmd_clear_color_image(
                    cmd,
                    img.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { float32: v },
                    &[gpu::color_range()],
                );
            };
            clear(&depth, [0.5, 0.0, 0.0, 0.0]);
            clear(&motion, [0.0; 4]);
            clear(&output, [0.0; 4]);
        }
        gpu.barrier(cmd);
        Ok(())
    })?;

    let t = Instant::now();
    let mut feature = None;
    gpu.run(|cmd| {
        feature = Some(Feature::create(&bridge, cmd, size, 0)?);
        Ok(())
    })?;
    let feature = feature.expect("set by the closure");
    log!(
        "nr-host: feature 18 created at {}x{} in {} ms",
        size.0,
        size.1,
        t.elapsed().as_millis()
    );

    let mut times = Vec::new();
    for i in 0..a.frames.max(1) {
        let t = Instant::now();
        gpu.run(|cmd| {
            feature.evaluate(
                cmd,
                &Frame {
                    color: &color,
                    depth: &depth,
                    motion: &motion,
                    output: &output,
                    reset: i == 0,
                    tuning: Tuning::default(),
                },
            )
        })?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    log!(
        "nr-host: {} evaluations, submit+wait median {:.2} ms (first {:.2} ms)",
        times.len(),
        times[times.len() / 2],
        times.first().copied().unwrap_or_default()
    );

    gpu.run(|cmd| {
        gpu.barrier(cmd);
        // SAFETY: recording command buffer.
        unsafe {
            d.cmd_copy_image_to_buffer(
                cmd,
                output.image,
                vk::ImageLayout::GENERAL,
                readback.buffer,
                &[output.region()],
            );
        }
        Ok(())
    })?;
    let out: Vec<u16> = readback
        .bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| u16::from_le_bytes(c))
        .collect();
    let (mut diff, mut mean, mut n) = (0.0f64, 0.0f64, 0usize);
    for (i, (&o, &s)) in out.iter().zip(&input).enumerate() {
        if i % 4 == 3 {
            continue;
        }
        let (o, s) = (f64::from(f32_of(o)), f64::from(f32_of(s)));
        diff += (o - s).abs();
        mean += o;
        n += 1;
    }
    log!(
        "nr-host: output mean {:.4}, mean |output - input| {:.4} (0 would mean untouched or empty)",
        mean / n as f64,
        diff / n as f64
    );
    let checksum = out.iter().fold(0u64, |h, &v| {
        h.rotate_left(5) ^ u64::from(v).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    });

    match output.fd {
        Some(fd) => log!(
            "nr-host: EXPORT memory fd={fd} size={} format=R16G16B16A16_SFLOAT {}x{} checksum={checksum:#018x} uuid={}",
            output.size,
            size.0,
            size.1,
            gpu.uuid_hex()
        ),
        None => log!("nr-host: no VK_KHR_external_memory_win32: cannot export to the native side"),
    }
    let semaphore = if gpu.has(c"VK_KHR_external_semaphore_win32") {
        match share::timeline_semaphore(&gpu) {
            Ok((s, fd)) => {
                log!("nr-host: EXPORT timeline semaphore fd={fd}");
                Some(s)
            }
            Err(e) => {
                log!("nr-host: semaphore export failed: {e}");
                None
            }
        }
    } else {
        None
    };
    if let (Some(sem), true) = (semaphore, a.hold_s > 0) {
        // Signal 1..=5 from the GPU, then wait for the native side's 10.
        for v in 1..=5u64 {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let values = [v];
            let sems = [sem];
            let mut timeline =
                vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
            // SAFETY: an empty batch signalling our semaphore.
            unsafe {
                gpu.device
                    .queue_submit(
                        gpu.queue,
                        &[vk::SubmitInfo::default()
                            .signal_semaphores(&sems)
                            .push_next(&mut timeline)],
                        vk::Fence::null(),
                    )
                    .map_err(|e| format!("signal submit: {e:?}"))?;
            }
        }
        log!("nr-host: GPU signalled 1..=5");
        let t = Instant::now();
        let sems = [sem];
        // SAFETY: valid semaphore.
        let r = unsafe {
            gpu.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&sems)
                    .values(&[10]),
                a.hold_s * 1_000_000_000,
            )
        };
        log!(
            "nr-host: waited for native signal 10: {r:?} after {} ms",
            t.elapsed().as_millis()
        );
    } else if a.hold_s > 0 {
        log!("nr-host: holding for {} s", a.hold_s);
        std::thread::sleep(std::time::Duration::from_secs(a.hold_s));
    }

    // SAFETY: nothing is in flight.
    unsafe {
        let _ = gpu.device.device_wait_idle();
    }
    if let Some(s) = semaphore {
        // SAFETY: unused by any pending work.
        unsafe { gpu.device.destroy_semaphore(s, None) };
    }
    drop(feature);
    bridge.shutdown(&gpu);
    log!("nr-host: done");
    Ok(())
}
