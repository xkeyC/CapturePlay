//! Upscaler ground-truth benchmark on real captured frames.
//!
//! A native 4K HDR frame (P010 PQ BT.2020, grabbed with
//! `cargo run -p cp-capture --example grab`) is the reference. It is
//! downscaled to 1440p in linear light, upscaled back to 4K by each
//! candidate, and compared with the original in the PQ (perceptual) domain.
//!
//! `CAPTUREPLAY_TEST_FRAMES=<dir with frame_N.raw> cargo test -p cp-render
//! --release upscale -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use cp_core::color::{nits_to_pq, pq_to_nits, yuv_to_rgb_matrix};
use cp_core::{Range, YuvMatrix};
use cp_nvidia::vsr::{VsrQuality, VsrSession};
use windows::Win32::Graphics::Direct3D12::*;

use crate::testutil::Harness;

const W: usize = 3840;
const H: usize = 2160;
const LW: usize = 2560;
const LH: usize = 1440;

type Img = Vec<[f32; 3]>;

/// P010 (limited, BT.2020 NCL) → PQ-encoded R'G'B' in [0,1].
fn decode_p010(raw: &[u8]) -> Img {
    let m = yuv_to_rgb_matrix(YuvMatrix::Bt2020Ncl, Range::Limited, 10, 16);
    let px = |i: usize| f32::from(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]])) / 65535.0;
    let uv0 = W * H;
    let mut out = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let yv = px(y * W + x);
            // Chroma: nearest sample of the 4:2:0 plane (co-sited left).
            let ci = uv0 + (y / 2) * W + (x / 2) * 2;
            let (u, v) = (px(ci), px(ci + 1));
            let f = |r: &[f32; 4]| (r[0] * yv + r[1] * u + r[2] * v + r[3]).clamp(0.0, 1.0);
            out.push([f(&m[0]), f(&m[1]), f(&m[2])]);
        }
    }
    out
}

fn to_linear(img: &Img) -> Img {
    img.iter().map(|p| p.map(pq_to_nits)).collect()
}

fn to_pq(img: &Img) -> Img {
    img.iter()
        .map(|p| p.map(|c| nits_to_pq(c.max(0.0))))
        .collect()
}

/// Separable resampling with a kernel `k(x)` of `support` (in output-scale
/// units for downscaling, source units for upscaling).
fn resample(
    src: &Img,
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    kernel: &dyn Fn(f32) -> f32,
    support: f32,
) -> Img {
    let pass = |src: &Img, sw: usize, sh: usize, dw: usize, horizontal: bool| -> Img {
        let (n_out, n_in, lines) = if horizontal {
            (dw, sw, sh)
        } else {
            (dh, sh, sw)
        };
        let scale = n_in as f32 / n_out as f32;
        let fscale = scale.max(1.0); // widen the kernel when downscaling
        let mut out = vec![[0f32; 3]; if horizontal { dw * sh } else { sw * dh }];
        for o in 0..n_out {
            let center = (o as f32 + 0.5) * scale - 0.5;
            let lo = (center - support * fscale).floor() as i64;
            let hi = (center + support * fscale).ceil() as i64;
            let mut taps = Vec::new();
            let mut sum = 0.0;
            for i in lo..=hi {
                let w = kernel((i as f32 - center) / fscale);
                if w != 0.0 {
                    taps.push((i.clamp(0, n_in as i64 - 1) as usize, w));
                    sum += w;
                }
            }
            for l in 0..lines {
                let mut acc = [0f32; 3];
                for &(i, w) in &taps {
                    let p = if horizontal {
                        src[l * sw + i]
                    } else {
                        src[i * sw + l]
                    };
                    for c in 0..3 {
                        acc[c] += p[c] * w / sum;
                    }
                }
                if horizontal {
                    out[l * dw + o] = acc;
                } else {
                    out[o * sw + l] = acc;
                }
            }
        }
        out
    };
    let h = pass(src, sw, sh, dw, true);
    pass(&h, dw, sh, dw, false)
}

fn box_kernel(x: f32) -> f32 {
    if x.abs() <= 0.5 { 1.0 } else { 0.0 }
}

fn triangle(x: f32) -> f32 {
    (1.0 - x.abs()).max(0.0)
}

fn catmull_rom(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        1.5 * x * x * x - 2.5 * x * x + 1.0
    } else if x < 2.0 {
        -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0
    } else {
        0.0
    }
}

fn lanczos3(x: f32) -> f32 {
    let x = x.abs();
    if x < 1e-6 {
        1.0
    } else if x < 3.0 {
        let px = std::f32::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    } else {
        0.0
    }
}

/// PSNR in the PQ domain (peak 1.0), excluding a border.
fn psnr(a: &Img, b: &Img) -> f64 {
    let (mut se, mut n) = (0f64, 0f64);
    for y in 8..H - 8 {
        for x in 8..W - 8 {
            let i = y * W + x;
            for c in 0..3 {
                let d = f64::from(a[i][c] - b[i][c]);
                se += d * d;
            }
            n += 3.0;
        }
    }
    10.0 * (1.0 / (se / n)).log10()
}

/// Encodings fed to VSR (10-bit).
#[derive(Clone, Copy, Debug)]
enum Encoding {
    /// HDR10 code values as captured.
    Pq,
    /// sRGB curve over 0..1000 nits (SDR-like distribution).
    Srgb1000,
}

fn encode(enc: Encoding, pq: [f32; 3]) -> [f32; 3] {
    match enc {
        Encoding::Pq => pq,
        Encoding::Srgb1000 => {
            pq.map(|c| cp_core::color::linear_to_srgb((pq_to_nits(c) / 1000.0).clamp(0.0, 1.0)))
        }
    }
}

fn decode(enc: Encoding, e: [f32; 3]) -> [f32; 3] {
    match enc {
        Encoding::Pq => e,
        Encoding::Srgb1000 => e.map(|c| nits_to_pq(cp_core::color::srgb_to_linear(c) * 1000.0)),
    }
}

fn pack(img: &Img, w: usize, h: usize, pitch: usize, enc: Encoding) -> Vec<u8> {
    let mut out = vec![0u8; pitch * h];
    for y in 0..h {
        for x in 0..w {
            let q =
                encode(enc, img[y * w + x]).map(|c| (c.clamp(0.0, 1.0) * 1023.0).round() as u32);
            let v = q[0] | (q[1] << 10) | (q[2] << 20) | (3 << 30);
            out[y * pitch + x * 4..y * pitch + x * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn unpack(bytes: &[u8], w: usize, h: usize, pitch: usize, enc: Encoding) -> Img {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let o = y * pitch + x * 4;
            let v = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            let e = [v & 1023, (v >> 10) & 1023, (v >> 20) & 1023].map(|c| c as f32 / 1023.0);
            out.push(decode(enc, e));
        }
    }
    out
}

/// Runs VSR once on `low` and returns the 4K result plus mean run time (ms).
fn run_vsr(
    hs: &mut Harness,
    dir: &std::path::Path,
    low: &Img,
    q: VsrQuality,
    enc: Encoding,
) -> Option<(Img, f64)> {
    let mut s = match VsrSession::new(
        &hs.gpu.device,
        hs.gpu.adapter_luid,
        dir,
        (LW as u32, LH as u32),
        (W as u32, H as u32),
        q,
        1.0,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("  VSR {q:?}: unavailable ({e})");
            return None;
        }
    };
    let (ip, op) = (s.input().pitch as usize, s.output().pitch as usize);
    let data = pack(low, LW, LH, ip, enc);
    let upload = hs
        .gpu
        .create_buffer(
            data.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
        .expect("upload");
    let readback = hs
        .gpu
        .create_buffer(
            (op * H) as u64,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )
        .expect("readback");
    // SAFETY: mapped upload buffer of data.len() bytes.
    unsafe {
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).expect("map");
        std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast(), data.len());
    }
    hs.reopen();
    hs.copy_buffer(s.input_resource(), &upload, data.len() as u64);
    hs.finish();
    // Warm-up + timed runs (CPU wall time around enqueue + stream sync).
    let mut value = 0u64;
    let mut times = Vec::new();
    for i in 0..12 {
        value += 1;
        // SAFETY: valid queue and shared fence.
        unsafe { hs.gpu.queue.Signal(s.producer(), value).expect("signal") };
        let t0 = Instant::now();
        s.run(value).expect("run");
        s.synchronize();
        if i >= 2 {
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    // SAFETY: valid queue and shared fence.
    unsafe { hs.gpu.queue.Wait(s.completion(), value).expect("wait") };
    hs.reopen();
    hs.copy_buffer(&readback, s.output_resource(), (op * H) as u64);
    hs.finish();
    let mut bytes = vec![0u8; op * H];
    // SAFETY: GPU finished; buffer holds op*H bytes.
    unsafe {
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).expect("map");
        std::ptr::copy_nonoverlapping(p.cast::<u8>(), bytes.as_mut_ptr(), bytes.len());
    }
    times.sort_by(f64::total_cmp);
    Some((unpack(&bytes, W, H, op, enc), times[times.len() / 2]))
}

/// SDR preview of a crop (nits/200 → sRGB) for visual comparison.
fn crop_ppm(images: &[(&str, &Img)], x0: usize, y0: usize, cw: usize, ch: usize) -> Vec<u8> {
    let mut out = format!("P6 {} {} 255\n", cw * images.len(), ch).into_bytes();
    for y in y0..y0 + ch {
        for (_, img) in images {
            for x in x0..x0 + cw {
                for c in img[y * W + x] {
                    let v = cp_core::color::linear_to_srgb((pq_to_nits(c) / 200.0).clamp(0.0, 1.0));
                    out.push((v * 255.0) as u8);
                }
            }
        }
    }
    out
}

#[test]
#[ignore = "needs captured frames (CAPTUREPLAY_TEST_FRAMES) and an RTX GPU with the VFX runtime"]
fn upscale_1440p_to_4k() {
    let frames =
        PathBuf::from(std::env::var("CAPTUREPLAY_TEST_FRAMES").expect("CAPTUREPLAY_TEST_FRAMES"));
    let dir = std::env::var("CAPTUREPLAY_VFX_RUNTIME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runtime/nvidia-vfx")
        });
    let mut hs = Harness::new();
    // The harness list starts open; close it so `reopen` is valid.
    hs.finish();
    for n in 0.. {
        let Ok(raw) = std::fs::read(frames.join(format!("frame_{n}.raw"))) else {
            break;
        };
        println!("frame_{n}:");
        let gt = decode_p010(&raw);
        let lin = to_linear(&gt);
        let low = to_pq(&resample(&lin, W, H, LW, LH, &box_kernel, 0.5));
        let up =
            |k: &dyn Fn(f32) -> f32, s: f32| to_pq(&resample(&to_linear(&low), LW, LH, W, H, k, s));
        let bilinear = up(&triangle, 1.0);
        let catmull = up(&catmull_rom, 2.0);
        let lanczos = up(&lanczos3, 3.0);
        println!(
            "  {:<34} {:6.2} dB",
            "bilinear (linear light)",
            psnr(&bilinear, &gt)
        );
        println!(
            "  {:<34} {:6.2} dB",
            "Catmull-Rom (current)",
            psnr(&catmull, &gt)
        );
        println!("  {:<34} {:6.2} dB", "Lanczos3", psnr(&lanczos, &gt));
        // Fallback in the output pass: Catmull-Rom on PQ-encoded values.
        let catmull_pq = resample(&low, LW, LH, W, H, &catmull_rom, 2.0);
        println!(
            "  {:<34} {:6.2} dB",
            "Catmull-Rom (PQ domain, fallback)",
            psnr(&catmull_pq, &gt)
        );
        let mut best: Option<(String, Img, f64)> = None;
        let mut keep: Vec<(String, Img)> = Vec::new();
        for enc in [Encoding::Pq, Encoding::Srgb1000] {
            for q in [
                VsrQuality::Bicubic,
                VsrQuality::Low,
                VsrQuality::Medium,
                VsrQuality::High,
                VsrQuality::Ultra,
                VsrQuality::HighBitrateHigh,
                VsrQuality::HighBitrateUltra,
                VsrQuality::StreamingMedium,
                VsrQuality::StreamingUltra,
            ] {
                if let Some((img, ms)) = run_vsr(&mut hs, &dir, &low, q, enc) {
                    let db = psnr(&img, &gt);
                    println!(
                        "  {:<34} {:6.2} dB  {:5.2} ms",
                        format!("VSR {q:?} ({enc:?})"),
                        db,
                        ms
                    );
                    if matches!(enc, Encoding::Pq)
                        && matches!(q, VsrQuality::Medium | VsrQuality::High)
                    {
                        keep.push((format!("VSR {q:?}"), img.clone()));
                    }
                    if best.as_ref().is_none_or(|b| db > b.2) {
                        best = Some((format!("{q:?} {enc:?}"), img, db));
                    }
                }
            }
        }
        if let Ok(out) = std::env::var("CAPTUREPLAY_TEST_DUMP") {
            // Detail-rich crops: leaves/flowers and moss/rock.
            let mut cols: Vec<(&str, &Img)> = vec![("truth", &gt), ("catmull-pq", &catmull_pq)];
            cols.extend(keep.iter().map(|(n, i)| (n.as_str(), i)));
            for (k, (x, y)) in [(1700usize, 1350usize), (2600, 300)]
                .into_iter()
                .enumerate()
            {
                let _ = std::fs::write(
                    format!("{out}/upscale_{n}_{k}.ppm"),
                    crop_ppm(&cols, x, y, 480, 300),
                );
            }
            let names: Vec<&str> = cols.iter().map(|c| c.0).collect();
            println!("  crops: {}", names.join(" | "));
        }
        let _ = (&best, &catmull);
    }
}

/// VSR runs on an SDK-internal CUDA stream: a fence signaled on our stream
/// after `NvVFX_Run` fires before the upscale is done. Integrations must not
/// rely on stream ordering for VSR (see the README).
#[test]
#[ignore = "requires an RTX GPU and the VFX runtime"]
fn vsr_completion_fence_is_not_stream_ordered() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runtime/nvidia-vfx");
    let hs = Harness::new();
    let mut s = VsrSession::new(
        &hs.gpu.device,
        hs.gpu.adapter_luid,
        &dir,
        (LW as u32, LH as u32),
        (W as u32, H as u32),
        VsrQuality::High,
        1.0,
    )
    .expect("VSR");
    // SAFETY: plain event creation.
    let event =
        unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
            .expect("event");
    let (mut fence_ms, mut ctx_ms) = (Vec::new(), Vec::new());
    for v in 1..=12u64 {
        // SAFETY: valid queue/fence/event.
        unsafe { hs.gpu.queue.Signal(s.producer(), v).expect("signal") };
        let t0 = Instant::now();
        s.run(v).expect("run");
        // SAFETY: as above.
        unsafe {
            s.completion()
                .SetEventOnCompletion(v, event)
                .expect("event");
            windows::Win32::System::Threading::WaitForSingleObject(event, 5000);
        }
        let t1 = Instant::now();
        s.synchronize();
        let t2 = Instant::now();
        if v > 2 {
            fence_ms.push((t1 - t0).as_secs_f64() * 1e3);
            ctx_ms.push((t2 - t0).as_secs_f64() * 1e3);
        }
    }
    fence_ms.sort_by(f64::total_cmp);
    ctx_ms.sort_by(f64::total_cmp);
    println!(
        "VSR High 1440p->4K: completion fence after {:.2} ms, all CUDA work after {:.2} ms",
        fence_ms[5], ctx_ms[5]
    );
}
