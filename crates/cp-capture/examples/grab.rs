//! Grabs raw frames from the first capture device, for offline tests.
//!
//! `cargo run -p cp-capture --example grab -- <out_dir> [width height fps format] [count]`
//!
//! Writes `frame_<n>.raw` (tightly packed planes) and `frame.json`
//! (mode + detected color) to `out_dir`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cp_capture::{
    CaptureConfig, CaptureEvent, CaptureSession, FrameDest, FrameMeta, FrameSink, MfThread,
    PlaneDest,
};
use cp_core::VideoMode;
use cp_core::format::ModeRequest;
use cp_core::settings::ColorSettings;

struct DumpSink {
    buf: Mutex<Vec<u8>>,
    out: std::path::PathBuf,
    want: u64,
    saved: Mutex<u64>,
}

impl FrameSink for DumpSink {
    fn acquire(&self, mode: &VideoMode) -> Option<FrameDest> {
        let len = mode.format.frame_bytes(mode.width, mode.height);
        let planes = mode.format.planes(mode.width, mode.height);
        let mut b = self.buf.lock().ok()?;
        b.resize(len, 0);
        let (row0, rows0) = planes.planes[0];
        Some(FrameDest {
            slot: 0,
            base: b.as_mut_ptr(),
            len,
            planes: [
                PlaneDest {
                    offset: 0,
                    pitch: row0,
                },
                PlaneDest {
                    offset: row0 * rows0,
                    pitch: planes.planes[1].0,
                },
            ],
        })
    }

    fn publish(&self, _dest: FrameDest, meta: FrameMeta) {
        let mut saved = self.saved.lock().expect("lock");
        // Skip the first frames while the device settles.
        if meta.seq < 30 || *saved >= self.want {
            return;
        }
        let b = self.buf.lock().expect("lock");
        let _ = std::fs::write(self.out.join(format!("frame_{}.raw", *saved)), &*b);
        let m = meta.mode;
        let json = format!(
            "{{\"width\":{},\"height\":{},\"format\":\"{}\",\"color\":\"{:?}\"}}",
            m.width, m.height, m.format, meta.color
        );
        let _ = std::fs::write(self.out.join("frame.json"), json);
        *saved += 1;
        println!("saved frame {} ({m})", *saved);
    }

    fn cancel(&self, _dest: FrameDest) {}
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(
        args.first()
            .ok_or_else(|| anyhow::anyhow!("usage: grab <out_dir> [w h fps format] [count]"))?,
    );
    std::fs::create_dir_all(&out)?;
    let request = if args.len() >= 5 {
        ModeRequest {
            width: args[1].parse().ok(),
            height: args[2].parse().ok(),
            fps: args[3].parse().ok(),
            format: match args[4].to_ascii_lowercase().as_str() {
                "p010" => Some(cp_core::PixelFormat::P010),
                "nv12" => Some(cp_core::PixelFormat::Nv12),
                _ => None,
            },
        }
    } else {
        ModeRequest::default()
    };
    let want = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    let _mf = MfThread::init()?;
    let dev = cp_capture::list_devices()?
        .into_iter()
        .find(|d| !d.name.contains("Virtual"))
        .ok_or_else(|| anyhow::anyhow!("no device"))?;
    let sink = Arc::new(DumpSink {
        buf: Mutex::new(Vec::new()),
        out,
        want,
        saved: Mutex::new(0),
    });
    let session = CaptureSession::start(
        CaptureConfig {
            device: dev,
            request,
            color: ColorSettings::new(),
            max_buffers: 2,
        },
        sink.clone(),
        |e| {
            if let CaptureEvent::Stopped { error: Some(e) } = e {
                eprintln!("capture error: {e}");
            }
        },
    )?;
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(100));
        if *sink.saved.lock().expect("lock") >= want {
            break;
        }
    }
    session.stop();
    Ok(())
}
