//! Upscaler ground-truth benchmarks on real captured frames.
//!
//! A native 4K HDR frame (P010 PQ BT.2020, grabbed with
//! `cargo run -p cp-capture --example grab`) is the reference. It is
//! downscaled to 1440p in linear light, upscaled back to 4K by each
//! candidate, and compared with the original in the PQ (perceptual) domain.
//! NVIDIA RTX VSR runs through the real path: [`NgxVsrUpscaler`] (encode →
//! NGX D3D12 → decode) on working-format textures.
//!
//! `CAPTUREPLAY_TEST_FRAMES=<dir with frame_N.raw> cargo test -p cp-render
//! --release upscale_bench -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::{Duration, Instant};

use cp_core::color::{
    BT709_TO_BT2020, BT2020_TO_BT709, SCRGB_UNIT_NITS, mul3, nits_to_pq, pq_to_nits,
    yuv_to_rgb_matrix,
};
use cp_core::{Range, YuvMatrix};
use cp_nvidia::ngx::VsrQuality;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;

use crate::renderer::Upscaler;
use crate::testutil::Harness;
use crate::texture::{Texture, TextureDesc};
use crate::upscale_ngx::NgxVsrUpscaler;

const W: usize = 3840;
const H: usize = 2160;
const LW: usize = 2560;
const LH: usize = 1440;

const QUALITIES: [VsrQuality; 5] = [
    VsrQuality::Bicubic,
    VsrQuality::Low,
    VsrQuality::Medium,
    VsrQuality::High,
    VsrQuality::Ultra,
];

type Img = Vec<[f32; 3]>;

fn frames_dir() -> PathBuf {
    PathBuf::from(std::env::var("CAPTUREPLAY_TEST_FRAMES").expect("CAPTUREPLAY_TEST_FRAMES"))
}

fn vfx_dir() -> PathBuf {
    std::env::var("CAPTUREPLAY_VFX_RUNTIME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runtime/nvidia-vfx")
        })
}

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

/// PQ R'G'B' -> packed RGB10A2 rows (R in the low bits).
fn pack(img: &Img, w: usize, h: usize, pitch: usize) -> Vec<u8> {
    let mut out = vec![0u8; pitch * h];
    for y in 0..h {
        for x in 0..w {
            let q = img[y * w + x].map(|c| (c.clamp(0.0, 1.0) * 1023.0).round() as u32);
            let v = q[0] | (q[1] << 10) | (q[2] << 20) | (3 << 30);
            out[y * pitch + x * 4..y * pitch + x * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn unpack(bytes: &[u8], w: usize, h: usize, pitch: usize) -> Img {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let o = y * pitch + x * 4;
            let v = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            out.push([v & 1023, (v >> 10) & 1023, (v >> 20) & 1023].map(|c| c as f32 / 1023.0));
        }
    }
    out
}

/// SDR preview of a crop (nits/200 -> sRGB) for visual comparison.
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

/// PSNR in the PQ domain for a `w × h` image.
fn psnr_wh(a: &Img, b: &Img, w: usize, h: usize) -> f64 {
    let (mut se, mut n) = (0f64, 0f64);
    for y in 8..h - 8 {
        for x in 8..w - 8 {
            let i = y * w + x;
            for c in 0..3 {
                let d = f64::from(a[i][c] - b[i][c]);
                se += d * d;
            }
            n += 3.0;
        }
    }
    10.0 * (1.0 / (se / n)).log10()
}

/// Uploads `data` into `dst` through a temporary upload buffer.
fn upload_to(hs: &mut Harness, dst: &ID3D12Resource, data: &[u8]) {
    let upload = hs
        .gpu
        .create_buffer(
            data.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
        .expect("upload");
    // SAFETY: mapped upload buffer of data.len() bytes.
    unsafe {
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).expect("map");
        std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast(), data.len());
    }
    hs.reopen();
    hs.copy_buffer(dst, &upload, data.len() as u64);
    hs.finish();
}

/// Reads `len` bytes of `src` back to the CPU.
fn read_from(hs: &mut Harness, src: &ID3D12Resource, len: usize) -> Vec<u8> {
    let readback = hs
        .gpu
        .create_buffer(
            len as u64,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )
        .expect("readback");
    hs.reopen();
    hs.copy_buffer(&readback, src, len as u64);
    hs.finish();
    let mut bytes = vec![0u8; len];
    // SAFETY: GPU finished; buffer holds len bytes.
    unsafe {
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).expect("map");
        std::ptr::copy_nonoverlapping(p.cast::<u8>(), bytes.as_mut_ptr(), len);
    }
    bytes
}

/// VFG (medium) frame at `t` between `a` and `b` (`w × h`, PQ).
fn run_vfg(
    hs: &mut Harness,
    dir: &std::path::Path,
    a: &Img,
    b: &Img,
    (w, h): (usize, usize),
    t: f32,
) -> Img {
    let mut s = cp_nvidia::vfg::VfgSession::new(
        &hs.gpu.device,
        hs.gpu.adapter_luid,
        dir,
        w as u32,
        h as u32,
        1,
    )
    .expect("VFG session");
    let pitch = s.pitch as usize;
    upload_to(hs, s.input(0), &pack(a, w, h, pitch));
    upload_to(hs, s.input(1), &pack(b, w, h, pitch));
    // SAFETY: valid queue and shared fence.
    unsafe { hs.gpu.queue.Signal(s.producer(), 1).expect("signal") };
    s.run(0, 1, t, false, 1).expect("run");
    s.synchronize_context();
    // SAFETY: valid queue and shared fence.
    unsafe { hs.gpu.queue.Wait(s.completion(), 1).expect("wait") };
    let bytes = read_from(hs, s.output(), pitch * h);
    unpack(&bytes, w, h, pitch)
}

/// Horizontal sub-pixel shift by `dx` pixels (Lanczos3), PQ domain.
fn shift_x(img: &Img, w: usize, h: usize, dx: f32) -> Img {
    let mut out = vec![[0f32; 3]; w * h];
    for x in 0..w {
        let c = x as f32 - dx;
        let lo = c.floor() as i64 - 2;
        let mut taps = Vec::with_capacity(6);
        let mut sum = 0.0;
        for i in lo..lo + 6 {
            let k = lanczos3(c - i as f32);
            taps.push((i.clamp(0, w as i64 - 1) as usize, k));
            sum += k;
        }
        for y in 0..h {
            let mut acc = [0f32; 3];
            for &(i, k) in &taps {
                let p = img[y * w + i];
                for ch in 0..3 {
                    acc[ch] += p[ch] * k / sum;
                }
            }
            out[y * w + x] = acc.map(|v| v.clamp(0.0, 1.0));
        }
    }
    out
}

fn diff(a: &Img, b: &Img) -> Img {
    a.iter()
        .zip(b)
        .map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
        .collect()
}

/// PQ BT.2020 R'G'B' → linear scRGB (BT.709, 1.0 = 80 nits) for upload.
fn to_scrgb(img: &Img) -> Vec<[f32; 4]> {
    img.iter()
        .map(|p| {
            let l = mul3(&BT2020_TO_BT709, p.map(pq_to_nits)).map(|c| c / SCRGB_UNIT_NITS);
            [l[0], l[1], l[2], 1.0]
        })
        .collect()
}

fn from_scrgb(px: &[[f32; 4]]) -> Img {
    px.iter()
        .map(|p| {
            mul3(
                &BT709_TO_BT2020,
                [p[0], p[1], p[2]].map(|c| c * SCRGB_UNIT_NITS),
            )
            .map(|n| nits_to_pq(n.max(0.0)))
        })
        .collect()
}

/// RTX VSR through the app's upscaler, ready for 1440p → 4K.
fn ngx_upscaler(hs: &Harness, q: VsrQuality) -> NgxVsrUpscaler {
    let mut dirs: Vec<PathBuf> = std::env::var_os("CAPTUREPLAY_RTX_VIDEO_RUNTIME")
        .map(Into::into)
        .into_iter()
        .collect();
    dirs.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../runtime/rtx-video-sdk/bin/Windows/x64/rel"),
    );
    let mut up = NgxVsrUpscaler::new(
        &hs.gpu,
        dirs,
        std::env::temp_dir().join("capture-play-ngx-test"),
        q,
    )
    .expect("RTX VSR (needs the RTX Video SDK, see runtime/README.md)");
    let t0 = Instant::now();
    while !up.prepare(
        &hs.gpu,
        &hs.heap,
        (LW as u32, LH as u32),
        (W as u32, H as u32),
    ) {
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "RTX VSR did not become ready"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    up
}

fn working(hs: &Harness, w: usize, h: usize) -> Texture {
    Texture::new(
        &hs.gpu,
        &hs.heap,
        TextureDesc::new(w as u32, h as u32, DXGI_FORMAT_R16G16B16A16_FLOAT).uav(),
    )
    .expect("texture")
}

/// Upscales one 1440p frame to 4K and reads it back.
fn ngx_run(hs: &mut Harness, up: &mut NgxVsrUpscaler, low: &Img) -> Img {
    hs.reopen();
    let src = hs.upload(LW as u32, LH as u32, &to_scrgb(low));
    let out = working(hs, W, H);
    up.upscale(&mut hs.ctx(), &src, &out).expect("upscale");
    let rb = hs.readback(&out);
    hs.finish();
    from_scrgb(&rb.read())
}

/// Mean GPU-bound time of one upscale (encode + VSR + decode), ms.
fn ngx_ms(hs: &mut Harness, up: &mut NgxVsrUpscaler, low: &Img) -> f64 {
    const N: u32 = 20;
    hs.reopen();
    let src = hs.upload(LW as u32, LH as u32, &to_scrgb(low));
    let out = working(hs, W, H);
    hs.finish();
    hs.reopen();
    for _ in 0..N {
        up.upscale(&mut hs.ctx(), &src, &out).expect("upscale");
    }
    let t0 = Instant::now();
    hs.finish();
    t0.elapsed().as_secs_f64() * 1e3 / f64::from(N)
}

#[test]
#[ignore = "needs captured frames (CAPTUREPLAY_TEST_FRAMES), an RTX GPU and the RTX Video SDK"]
fn upscale_1440p_to_4k() {
    let frames = frames_dir();
    let mut hs = Harness::new();
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
        let row = |name: &str, img: &Img| println!("  {name:<34} {:6.2} dB", psnr(img, &gt));
        row("bilinear (linear light)", &up(&triangle, 1.0));
        row("Catmull-Rom (linear light)", &up(&catmull_rom, 2.0));
        row("Lanczos3", &up(&lanczos3, 3.0));
        // Fallback in the output pass: Catmull-Rom on PQ-encoded values.
        let catmull_pq = resample(&low, LW, LH, W, H, &catmull_rom, 2.0);
        row("Catmull-Rom (PQ domain, fallback)", &catmull_pq);
        let mut keep: Vec<(String, Img)> = Vec::new();
        // One feature at a time: with several alive at once, later ones
        // never became ready here.
        for q in QUALITIES {
            let mut u = ngx_upscaler(&hs, q);
            let img = ngx_run(&mut hs, &mut u, &low);
            let ms = ngx_ms(&mut hs, &mut u, &low);
            println!(
                "  {:<34} {:6.2} dB  {ms:5.2} ms",
                format!("RTX VSR {q:?}"),
                psnr(&img, &gt)
            );
            if matches!(q, VsrQuality::Medium | VsrQuality::High) {
                keep.push((format!("VSR {q:?}"), img));
            }
        }
        if let Ok(out) = std::env::var("CAPTUREPLAY_TEST_DUMP") {
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
        }
    }
}

/// Where to interpolate when upscaling 1440p → 4K: on the upscaled frames
/// or before upscaling. Truth is the real middle frame of three consecutive
/// native-4K captures.
#[test]
#[ignore = "needs 3 consecutive captured frames (CAPTUREPLAY_TEST_FRAMES), an RTX GPU, the VFX runtime and the RTX Video SDK"]
fn interpolate_before_or_after_upscale() {
    let frames = frames_dir();
    let dir = vfx_dir();
    let gt: Vec<Img> = (0..3)
        .map(|n| decode_p010(&std::fs::read(frames.join(format!("frame_{n}.raw"))).expect("frame")))
        .collect();
    let low: Vec<Img> = gt
        .iter()
        .map(|g| to_pq(&resample(&to_linear(g), W, H, LW, LH, &box_kernel, 0.5)))
        .collect();
    let mut hs = Harness::new();
    hs.finish();
    println!(
        "motion: frame0↔1 {:.2} dB, frame0↔2 {:.2} dB (lower = more motion)",
        psnr(&gt[0], &gt[1]),
        psnr(&gt[0], &gt[2])
    );
    let vfg1440 = run_vfg(&mut hs, &dir, &low[0], &low[2], (LW, LH), 0.5);
    println!(
        "VFG 1440p vs truth 1440p         {:6.2} dB",
        psnr_wh(&vfg1440, &low[1], LW, LH)
    );
    let mut up = ngx_upscaler(&hs, VsrQuality::Medium);
    let ups: Vec<Img> = low.iter().map(|l| ngx_run(&mut hs, &mut up, l)).collect();
    let x = run_vfg(&mut hs, &dir, &ups[0], &ups[2], (W, H), 0.5);
    let y = ngx_run(&mut hs, &mut up, &vfg1440);
    println!(
        "VSR(true middle) [upper bound]   {:6.2} dB",
        psnr(&ups[1], &gt[1])
    );
    println!(
        "X: VSR → VFG@4K                  {:6.2} dB",
        psnr(&x, &gt[1])
    );
    println!(
        "Y: VFG@1440p → VSR               {:6.2} dB",
        psnr(&y, &gt[1])
    );
    println!(
        "repeat VSR(frame0)               {:6.2} dB",
        psnr(&ups[0], &gt[1])
    );
}

/// Temporal stability under a slow pan: a native 4K frame is shifted by a
/// fraction of a pixel per frame, downscaled to 1440p and upscaled back.
/// "temporal" compares frame-to-frame changes of the output with those of
/// the truth (higher = steadier). Also guards against output lag: each
/// output must match its own frame better than the previous one (the Video
/// Effects SDK route returned the previous call's frame).
#[test]
#[ignore = "needs a captured frame (CAPTUREPLAY_TEST_FRAMES), an RTX GPU and the RTX Video SDK"]
fn slow_pan_temporal_stability() {
    let step: f32 = std::env::var("PAN_STEP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.6);
    let gt = decode_p010(&std::fs::read(frames_dir().join("frame_0.raw")).expect("frame"));
    let truth: Vec<Img> = (0..6)
        .map(|k| shift_x(&gt, W, H, k as f32 * step))
        .collect();
    let lows: Vec<Img> = truth
        .iter()
        .map(|t| to_pq(&resample(&to_linear(t), W, H, LW, LH, &box_kernel, 0.5)))
        .collect();
    let mut hs = Harness::new();
    hs.finish();
    println!("pan {step} px/frame at 4K ({:.2} px at 1440p)", step / 1.5);
    let report = |name: &str, ups: &[Img]| -> bool {
        let spatial: f64 =
            ups.iter().zip(&truth).map(|(u, t)| psnr(u, t)).sum::<f64>() / ups.len() as f64;
        let mut temporal = 0.0;
        for k in 0..ups.len() - 1 {
            temporal += psnr(&diff(&ups[k + 1], &ups[k]), &diff(&truth[k + 1], &truth[k]));
        }
        temporal /= (ups.len() - 1) as f64;
        println!("  {name:<28} spatial {spatial:6.2} dB   temporal {temporal:6.2} dB");
        (1..ups.len()).all(|k| psnr(&ups[k], &truth[k]) > psnr(&ups[k], &truth[k - 1]))
    };
    let cr: Vec<Img> = lows
        .iter()
        .map(|l| resample(l, LW, LH, W, H, &catmull_rom, 2.0))
        .collect();
    assert!(report("Catmull-Rom (PQ)", &cr));
    for q in QUALITIES {
        let mut up = ngx_upscaler(&hs, q);
        let ups: Vec<Img> = lows.iter().map(|l| ngx_run(&mut hs, &mut up, l)).collect();
        assert!(
            report(&format!("RTX VSR {q:?}"), &ups),
            "RTX VSR {q:?} output lags its input"
        );
    }
}

/// Per-frame GPU cost of each VFG / RTX VSR quality.
#[test]
#[ignore = "needs a captured frame (CAPTUREPLAY_TEST_FRAMES), an RTX GPU, the VFX runtime and the RTX Video SDK"]
fn effect_costs() {
    let dir = vfx_dir();
    let gt = decode_p010(&std::fs::read(frames_dir().join("frame_0.raw")).expect("frame"));
    let low = to_pq(&resample(&to_linear(&gt), W, H, LW, LH, &box_kernel, 0.5));
    let shifted = shift_x(&low, LW, LH, 1.0);
    let mut hs = Harness::new();
    hs.finish();
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for (w, h, a, b) in [
        (LW, LH, &low, &shifted),
        (W, H, &gt, &shift_x(&gt, W, H, 1.5)),
    ] {
        for mode in 0..3u32 {
            let mut s = cp_nvidia::vfg::VfgSession::new(
                &hs.gpu.device,
                hs.gpu.adapter_luid,
                &dir,
                w as u32,
                h as u32,
                mode,
            )
            .expect("VFG");
            let pitch = s.pitch as usize;
            upload_to(&mut hs, s.input(0), &pack(a, w, h, pitch));
            upload_to(&mut hs, s.input(1), &pack(b, w, h, pitch));
            let mut times = Vec::new();
            for i in 0..24u64 {
                // SAFETY: valid queue and shared fence.
                unsafe { hs.gpu.queue.Signal(s.producer(), i + 1).expect("signal") };
                let t0 = Instant::now();
                s.run(0, 1, (i % 3 + 1) as f32 / 4.0, false, i + 1)
                    .expect("run");
                s.synchronize_context();
                if i >= 4 {
                    times.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            println!("VFG {w}x{h} mode {mode}: {:.2} ms", median(times));
        }
    }
    for q in QUALITIES {
        let mut up = ngx_upscaler(&hs, q);
        let _ = ngx_ms(&mut hs, &mut up, &low);
        println!(
            "RTX VSR {q:?} (encode + NGX + decode): {:.2} ms",
            ngx_ms(&mut hs, &mut up, &low)
        );
    }
}

/// RTX VSR at 1:1 (artifact reduction without upscaling): does NGX accept
/// it, how much does it change the frame and what does it cost?
#[test]
#[ignore = "needs a captured frame (CAPTUREPLAY_TEST_FRAMES), an RTX GPU and the RTX Video SDK"]
fn vsr_same_size() {
    let gt = decode_p010(&std::fs::read(frames_dir().join("frame_0.raw")).expect("frame"));
    let low = to_pq(&resample(&to_linear(&gt), W, H, LW, LH, &box_kernel, 0.5));
    let mut hs = Harness::new();
    hs.finish();
    for q in QUALITIES {
        let mut up = ngx_upscaler(&hs, q);
        let size = (LW as u32, LH as u32);
        let t0 = Instant::now();
        while !up.prepare(&hs.gpu, &hs.heap, size, size) {
            assert!(t0.elapsed() < Duration::from_secs(30));
            std::thread::sleep(Duration::from_millis(20));
        }
        hs.reopen();
        let src = hs.upload(LW as u32, LH as u32, &to_scrgb(&low));
        let out = working(&hs, LW, LH);
        let r = up.upscale(&mut hs.ctx(), &src, &out);
        let rb = hs.readback(&out);
        hs.finish();
        match r {
            Ok(()) => {
                let img = from_scrgb(&rb.read());
                println!(
                    "RTX VSR {q:?} 1:1: vs input {:.2} dB",
                    psnr_wh(&img, &low, LW, LH)
                );
            }
            Err(e) => println!("RTX VSR {q:?} 1:1: {e}"),
        }
    }
}

/// Does 1:1 RTX VSR ("artifact reduction") bring an interpolated frame
/// closer to the real one? Compares VFG(frame0, frame2) at t = 0.5 with the
/// real frame 1 before and after VSR, against the same VSR on the real
/// frame (its change when there is nothing to repair).
#[test]
#[ignore = "needs 3 consecutive captured frames (CAPTUREPLAY_TEST_FRAMES), an RTX GPU, the VFX runtime and the RTX Video SDK"]
fn vsr_on_interpolated_frame() {
    let frames = frames_dir();
    let low: Vec<Img> = (0..3)
        .map(|n| {
            let gt =
                decode_p010(&std::fs::read(frames.join(format!("frame_{n}.raw"))).expect("frame"));
            to_pq(&resample(&to_linear(&gt), W, H, LW, LH, &box_kernel, 0.5))
        })
        .collect();
    let mut hs = Harness::new();
    hs.finish();
    let size = (LW as u32, LH as u32);
    let mut results = Vec::new();
    for mode in 0..3u32 {
        let dir = vfx_dir();
        let interp = {
            let mut s = cp_nvidia::vfg::VfgSession::new(
                &hs.gpu.device,
                hs.gpu.adapter_luid,
                &dir,
                LW as u32,
                LH as u32,
                mode,
            )
            .expect("VFG");
            let pitch = s.pitch as usize;
            upload_to(&mut hs, s.input(0), &pack(&low[0], LW, LH, pitch));
            upload_to(&mut hs, s.input(1), &pack(&low[2], LW, LH, pitch));
            // SAFETY: valid queue and shared fence.
            unsafe { hs.gpu.queue.Signal(s.producer(), 1).expect("signal") };
            s.run(0, 1, 0.5, false, 1).expect("run");
            s.synchronize_context();
            // SAFETY: as above.
            unsafe { hs.gpu.queue.Wait(s.completion(), 1).expect("wait") };
            let bytes = read_from(&mut hs, s.output(), pitch * LH);
            unpack(&bytes, LW, LH, pitch)
        };
        results.push((mode, interp));
    }
    println!(
        "motion frame0↔2 (1440p): {:.2} dB",
        psnr_wh(&low[0], &low[2], LW, LH)
    );
    for q in QUALITIES {
        let mut up = ngx_upscaler(&hs, q);
        let t0 = Instant::now();
        while !up.prepare(&hs.gpu, &hs.heap, size, size) {
            assert!(t0.elapsed() < Duration::from_secs(30));
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut run = |img: &Img| {
            hs.reopen();
            let src = hs.upload(LW as u32, LH as u32, &to_scrgb(img));
            let out = working(&hs, LW, LH);
            up.upscale(&mut hs.ctx(), &src, &out).expect("vsr");
            let rb = hs.readback(&out);
            hs.finish();
            from_scrgb(&rb.read())
        };
        let real = run(&low[1]);
        print!(
            "VSR {q:?} 1:1 changes the real frame to {:.2} dB |",
            psnr_wh(&real, &low[1], LW, LH)
        );
        // Interpolation error before (vs the real frame) and after (both
        // through VSR): higher after = VSR repaired some of it.
        for (mode, interp) in &results {
            let before = psnr_wh(interp, &low[1], LW, LH);
            let after = psnr_wh(&run(interp), &real, LW, LH);
            print!("  VFG mode {mode}: {before:.2} → {after:.2} dB");
        }
        println!();
    }
}
