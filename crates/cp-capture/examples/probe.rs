//! Lists capture devices and modes, then captures a few seconds into a CPU
//! sink and reports throughput, drops and the detected content frame rate.
//!
//! `cargo run -p cp-capture --example probe -- [width height fps format]`

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cp_capture::{
    CaptureConfig, CaptureEvent, CaptureSession, FrameDest, FrameMeta, FrameSink, MfThread,
    PlaneDest,
};
use cp_core::VideoMode;
use cp_core::cadence::{CadenceConfig, CadenceTracker};
use cp_core::format::ModeRequest;
use cp_core::settings::ColorSettings;

struct CpuSink {
    buf: Mutex<Vec<u8>>,
    cadence: Mutex<(CadenceTracker, u64, i64, i64)>,
    /// Device timestamps of all frames and of unique frames.
    times: Mutex<(Vec<i64>, Vec<i64>)>,
}

/// Histogram of successive differences, in 0.5 ms buckets.
fn print_intervals(label: &str, times: &[i64]) {
    let mut buckets = std::collections::BTreeMap::<i64, u32>::new();
    for w in times.windows(2) {
        *buckets
            .entry(((w[1] - w[0]) as f64 / 500_000.0).round() as i64)
            .or_default() += 1;
    }
    let total: u32 = buckets.values().sum();
    println!("{label} intervals ({total}):");
    for (b, n) in buckets {
        println!("  {:6.1} ms  {n:5}", b as f64 * 0.5);
    }
}

impl FrameSink for CpuSink {
    fn acquire(&self, mode: &VideoMode) -> Option<FrameDest> {
        let planes = mode.format.planes(mode.width, mode.height);
        let pitch = planes.planes[0].0.next_multiple_of(256);
        let rows0 = planes.planes[0].1;
        let len = pitch * (rows0 + planes.planes[1].1);
        let mut b = self.buf.lock().ok()?;
        b.resize(len, 0);
        Some(FrameDest {
            slot: 0,
            base: b.as_mut_ptr(),
            len,
            planes: [
                PlaneDest { offset: 0, pitch },
                PlaneDest {
                    offset: pitch * rows0,
                    pitch,
                },
            ],
        })
    }
    fn publish(&self, _dest: FrameDest, meta: FrameMeta) {
        let mut g = self.cadence.lock().expect("lock");
        let iv = meta.mode.fps.frame_duration_hns() * 100;
        let info = g.0.push(meta.signature, meta.device_time.0, iv);
        if info.unique {
            g.1 += 1;
        }
        if let Ok(mut t) = self.times.lock() {
            t.0.push(meta.device_time.0);
            if info.unique {
                t.1.push(meta.device_time.0);
            }
        }
        let lat = meta.arrival - meta.device_time;
        g.2 = g.2.max(lat);
        if meta.seq.is_multiple_of(240) {
            g.3 = info.content_interval_ns.unwrap_or(0);
            println!(
                "seq {:5} color {:?} content interval {:.2} ms",
                meta.seq,
                meta.color.transfer,
                g.3 as f64 / 1e6
            );
        }
    }
    fn cancel(&self, _dest: FrameDest) {}
}

fn main() -> anyhow::Result<()> {
    eprintln!("init MF");
    let _mf = MfThread::init()?;
    eprintln!("enumerating");
    let devices = cp_capture::list_devices()?;
    for d in &devices {
        println!("{}  [{}]", d.name, d.symbolic_link);
    }
    let dev = devices
        .iter()
        .find(|d| !d.name.contains("OBS"))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no device"))?;
    eprintln!("opening {}", dev.name);
    let modes = cp_capture::list_modes(&dev)?;
    println!("{} modes (≥1080p):", modes.len());
    for m in modes.iter().filter(|m| m.height >= 1080) {
        println!("  {m}");
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let request = if args.len() >= 4 {
        ModeRequest {
            width: args[0].parse().ok(),
            height: args[1].parse().ok(),
            fps: args[2].parse().ok(),
            format: match args[3].to_ascii_lowercase().as_str() {
                "p010" => Some(cp_core::PixelFormat::P010),
                "nv12" => Some(cp_core::PixelFormat::Nv12),
                "yuy2" => Some(cp_core::PixelFormat::Yuy2),
                _ => None,
            },
        }
    } else {
        ModeRequest {
            width: Some(1920),
            height: Some(1080),
            fps: Some(60),
            format: None,
        }
    };
    let sink = Arc::new(CpuSink {
        buf: Mutex::new(Vec::new()),
        cadence: Mutex::new((CadenceTracker::new(CadenceConfig::default()), 0, 0, 0)),
        times: Mutex::new((Vec::new(), Vec::new())),
    });
    let cfg = CaptureConfig {
        device: dev,
        request,
        color: ColorSettings::new(),
        max_buffers: 2,
    };
    let session = CaptureSession::start(cfg, sink.clone(), |e| match e {
        CaptureEvent::Started { mode, color, .. } => println!("started {mode} {color:?}"),
        CaptureEvent::Stopped { error } => println!("stopped {error:?}"),
    })?;
    let secs = std::env::var("PROBE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    std::thread::sleep(Duration::from_secs(secs));
    let frames = session
        .stats()
        .frames
        .load(std::sync::atomic::Ordering::Relaxed);
    let dropped = session
        .stats()
        .dropped
        .load(std::sync::atomic::Ordering::Relaxed);
    session.stop();
    let g = sink.cadence.lock().expect("lock");
    println!(
        "frames {frames} dropped {dropped} unique {} max(arrival-device) {:.2} ms",
        g.1,
        g.2 as f64 / 1e6
    );
    let t = sink.times.lock().expect("lock");
    print_intervals("signal", &t.0);
    print_intervals("unique", &t.1);
    Ok(())
}
