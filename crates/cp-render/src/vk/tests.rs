//! GPU tests of the Vulkan passes against the CPU reference math in
//! `cp_core::color`. Skipped (with a note) when no Vulkan device exists.

use ash::vk;
use cp_core::color::{decode_to_scrgb, yuv_to_rgb_matrix};
use cp_core::{ColorDesc, PixelFormat, Rational, VideoMode};

use super::arena::Arena;
use super::convert::ConvertPass;
use super::device::Gpu;
use super::output::OutputPass;
use super::pipeline::Layouts;
use super::texture::{Buffer, Texture, WORKING_USAGE, barrier};
use crate::common::{DecodeParams, OutputParams};
use crate::half::{f16_to_f32, f32_to_f16};
use crate::upload::SlotLayout;

struct Harness {
    layouts: Layouts,
    arena: Arena,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    keep: Vec<Buffer>,
    gpu: Gpu,
}

impl Harness {
    fn new() -> Option<Self> {
        // Validation messages arrive through `tracing`.
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        let gpu = match Gpu::headless(std::env::var_os("CAPTUREPLAY_VK_DEBUG").is_some()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return None;
            }
        };
        let layouts = Layouts::new(&gpu).expect("layouts");
        let arena = Arena::new(&gpu.device, &gpu.memory_properties, 4 << 20).expect("arena");
        // SAFETY: plain object creation on a valid device.
        let (pool, cmd) = unsafe {
            let pool = gpu
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family),
                    None,
                )
                .expect("pool");
            let cmd = gpu
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("cmd")[0];
            (pool, cmd)
        };
        let mut h = Self {
            layouts,
            arena,
            pool,
            cmd,
            keep: Vec::new(),
            gpu,
        };
        h.begin();
        Some(h)
    }

    fn begin(&mut self) {
        self.arena.reset();
        // SAFETY: the previous submission completed (submit_wait).
        unsafe {
            self.gpu
                .device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .expect("reset");
            self.gpu
                .device
                .begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())
                .expect("begin");
        }
    }

    fn submit_wait(&mut self) {
        // SAFETY: the buffer is open and recorded.
        unsafe { self.gpu.device.end_command_buffer(self.cmd).expect("end") };
        let v = self.gpu.submit(self.cmd, &[], &[]).expect("submit");
        self.gpu.wait(v).expect("wait");
        self.keep.clear();
        self.begin();
    }

    fn host_buffer(&self, size: usize, usage: vk::BufferUsageFlags) -> Buffer {
        Buffer::host(
            &self.gpu.device,
            &self.gpu.memory_properties,
            size as u64,
            usage,
        )
        .expect("buffer")
    }

    fn copy_region(w: u32, h: u32) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            })
    }

    /// RGBA16F texture with `rgba` (row-major) uploaded.
    fn upload(&mut self, w: u32, h: u32, rgba: &[[f32; 4]]) -> Texture {
        let tex = Texture::new(
            &self.gpu,
            self.cmd,
            w,
            h,
            vk::Format::R16G16B16A16_SFLOAT,
            WORKING_USAGE,
        )
        .expect("texture");
        let buf = self.host_buffer(rgba.len() * 8, vk::BufferUsageFlags::TRANSFER_SRC);
        let halves: Vec<u16> = rgba.iter().flatten().map(|&v| f32_to_f16(v)).collect();
        // SAFETY: the buffer holds exactly `halves`.
        unsafe { std::ptr::copy_nonoverlapping(halves.as_ptr().cast(), buf.ptr, halves.len() * 2) };
        barrier(&self.gpu.device, self.cmd);
        // SAFETY: recording a copy within both resources.
        unsafe {
            self.gpu.device.cmd_copy_buffer_to_image(
                self.cmd,
                buf.buffer,
                tex.image,
                vk::ImageLayout::GENERAL,
                &[Self::copy_region(w, h)],
            );
        }
        barrier(&self.gpu.device, self.cmd);
        self.keep.push(buf);
        tex
    }

    /// Submits everything recorded and reads `tex` (RGBA16F) back.
    fn readback(&mut self, tex: &Texture) -> Vec<[f32; 4]> {
        let n = (tex.width * tex.height) as usize;
        let buf = self.host_buffer(n * 8, vk::BufferUsageFlags::TRANSFER_DST);
        barrier(&self.gpu.device, self.cmd);
        // SAFETY: recording a copy within both resources.
        unsafe {
            self.gpu.device.cmd_copy_image_to_buffer(
                self.cmd,
                tex.image,
                vk::ImageLayout::GENERAL,
                buf.buffer,
                &[Self::copy_region(tex.width, tex.height)],
            );
        }
        barrier(&self.gpu.device, self.cmd);
        self.submit_wait();
        // SAFETY: the GPU finished writing; the mapping is coherent.
        let halves = unsafe { std::slice::from_raw_parts(buf.ptr.cast::<u16>(), n * 4) };
        halves
            .chunks(4)
            .map(|c| std::array::from_fn(|i| f16_to_f32(c[i])))
            .collect()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = self.gpu.wait_idle();
        // SAFETY: the GPU is idle.
        unsafe { self.gpu.device.destroy_command_pool(self.pool, None) };
    }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 + 5e-3 * b.abs()
}

/// Signal value of a plane at a (fractional) sample position.
type PlaneFn = fn(f32, f32) -> f32;

/// Writes a 4:2:0 frame (Y and interleaved CbCr in `storage` bits, values
/// given at the signal's bit depth) into an upload slot laid out like the
/// capture thread does.
#[allow(clippy::too_many_arguments)]
fn fill_420(
    buf: &Buffer,
    layout: &SlotLayout,
    w: u32,
    h: u32,
    sixteen: bool,
    shift: u32,
    y: impl Fn(u32, u32) -> u32,
    c: impl Fn(u32, u32) -> (u32, u32),
) {
    let put = |off: usize, v: u32| {
        // SAFETY: offsets lie within the slot (`layout.total`).
        unsafe {
            if sixteen {
                buf.ptr
                    .add(off)
                    .cast::<u16>()
                    .write_unaligned((v << shift) as u16);
            } else {
                *buf.ptr.add(off) = v as u8;
            }
        }
    };
    let bpc = if sixteen { 2 } else { 1 };
    for row in 0..h {
        for x in 0..w {
            put(
                layout.planes[0].offset + row as usize * layout.planes[0].pitch + x as usize * bpc,
                y(x, row),
            );
        }
    }
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            let (cb, cr) = c(i, j);
            let base = layout.planes[1].offset
                + j as usize * layout.planes[1].pitch
                + i as usize * 2 * bpc;
            put(base, cb);
            put(base + bpc, cr);
        }
    }
}

/// Converts a synthetic 4:2:0 frame on the GPU and compares every interior
/// pixel with the CPU reference. Chroma is linear in the texel grid, so the
/// shader's bilinear chroma fetch has an exact expected value.
fn check_420(format: PixelFormat, color: ColorDesc) {
    let Some(mut h) = Harness::new() else { return };
    let (w, ht) = (64u32, 32u32);
    let mode = VideoMode {
        width: w,
        height: ht,
        fps: Rational::new(60, 1),
        format,
    };
    let layout = SlotLayout::for_mode(format, w, ht);
    let upload = h.host_buffer(layout.total, vk::BufferUsageFlags::TRANSFER_SRC);
    let (sixteen, shift, max) = match format {
        PixelFormat::P010 => (true, 6u32, 65535.0f32),
        _ => (false, 0, 255.0),
    };
    let (yv, cbv, crv): (PlaneFn, PlaneFn, PlaneFn) = match format {
        PixelFormat::P010 => (
            |x, y| 64.0 + 6.0 * x + 8.0 * y,
            |i, j| 512.0 + 8.0 * i - 6.0 * j,
            |i, j| 512.0 - 5.0 * i + 7.0 * j,
        ),
        _ => (
            |x, y| 16.0 + 2.0 * x + 3.0 * y,
            |i, j| 128.0 + 2.0 * i - 1.0 * j,
            |i, j| 128.0 - 1.0 * i + 2.0 * j,
        ),
    };
    fill_420(
        &upload,
        &layout,
        w,
        ht,
        sixteen,
        shift,
        |x, y| yv(x as f32, y as f32) as u32,
        |i, j| {
            (
                cbv(i as f32, j as f32) as u32,
                crv(i as f32, j as f32) as u32,
            )
        },
    );
    let out = Texture::new(
        &h.gpu,
        h.cmd,
        w,
        ht,
        vk::Format::R16G16B16A16_SFLOAT,
        WORKING_USAGE,
    )
    .expect("out");
    let params = DecodeParams {
        sdr_white_nits: 203.0,
        hlg_peak_nits: 1000.0,
    };
    let mut convert = ConvertPass::new(&h.gpu, &h.layouts).expect("convert");
    let mut retire = Vec::new();
    convert
        .record(
            &h.gpu,
            &h.layouts,
            &mut h.arena,
            h.cmd,
            &upload,
            &layout,
            &mode,
            &color,
            params,
            &out,
            &mut retire,
        )
        .expect("record");
    let got = h.readback(&out);

    let rows = yuv_to_rgb_matrix(
        color.matrix,
        color.range,
        format.bit_depth(),
        format.storage_bits(),
    );
    let scale = (1u32 << shift) as f32 / max;
    let mut checked = 0;
    for y in 1..ht - 1 {
        for x in 0..w - 2 {
            // Shader chroma position in texel units (see convert.hlsl).
            let (i, j) = (x as f32 * 0.5, (y as f32 + 0.5) * 0.5 - 0.5);
            let ycc = [
                yv(x as f32, y as f32) * scale,
                cbv(i, j) * scale,
                crv(i, j) * scale,
            ];
            let rgbp: [f32; 3] = std::array::from_fn(|k| {
                rows[k][0] * ycc[0] + rows[k][1] * ycc[1] + rows[k][2] * ycc[2] + rows[k][3]
            });
            let want = decode_to_scrgb(rgbp, &color, params.sdr_white_nits, params.hlg_peak_nits);
            let px = got[(y * w + x) as usize];
            for k in 0..3 {
                assert!(
                    close(px[k], want[k]),
                    "{format:?} ({x},{y}) channel {k}: gpu {} cpu {}",
                    px[k],
                    want[k]
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 1000);
}

#[test]
fn convert_p010_hdr10_matches_cpu_reference() {
    check_420(PixelFormat::P010, ColorDesc::HDR10);
}

#[test]
fn convert_nv12_sdr_matches_cpu_reference() {
    check_420(PixelFormat::Nv12, ColorDesc::SDR_709);
}

/// Same-size output with no tone mapping must reproduce the source pixel
/// for pixel: catches a flipped clip space (`-fvk-invert-y`) or a wrong
/// letterbox / constant layout.
#[test]
fn output_pass_is_upright_and_identity() {
    let Some(mut h) = Harness::new() else { return };
    let (w, ht) = (16u32, 8u32);
    let src: Vec<[f32; 4]> = (0..ht)
        .flat_map(|y| (0..w).map(move |x| [(y + 1) as f32 * 0.5, (x + 1) as f32 * 0.25, 0.1, 1.0]))
        .collect();
    let tex = h.upload(w, ht, &src);
    let target = Texture::new(
        &h.gpu,
        h.cmd,
        w,
        ht,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    )
    .expect("target");
    let pass =
        OutputPass::new(&h.gpu, &h.layouts, vk::Format::R16G16B16A16_SFLOAT).expect("output");
    let attachments = [vk::RenderingAttachmentInfo::default()
        .image_view(target.view)
        .image_layout(vk::ImageLayout::GENERAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)];
    let rendering = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: w,
                height: ht,
            },
        })
        .layer_count(1)
        .color_attachments(&attachments);
    barrier(&h.gpu.device, h.cmd);
    // SAFETY: recording dynamic rendering into the offscreen target.
    unsafe { h.gpu.device.cmd_begin_rendering(h.cmd, &rendering) };
    let params = OutputParams {
        display_hdr: true,
        display_peak_nits: 10_000.0,
        sdr_white_nits: 203.0,
        content_peak_nits: 1000.0,
        catmull_rom: false,
    };
    pass.record(
        &h.gpu,
        &h.layouts,
        &mut h.arena,
        h.cmd,
        Some(&tex),
        (w, ht),
        &params,
    )
    .expect("record");
    // SAFETY: closes the rendering begun above.
    unsafe { h.gpu.device.cmd_end_rendering(h.cmd) };
    let got = h.readback(&target);
    for (i, (g, s)) in got.iter().zip(&src).enumerate() {
        for k in 0..3 {
            assert!(
                close(g[k], s[k]),
                "pixel ({}, {}) channel {k}: got {} want {}",
                i as u32 % w,
                i as u32 / w,
                g[k],
                s[k]
            );
        }
    }
}

/// Letterboxing (16×8 into 16×16: black bars of 4 rows) and the BT.2390
/// tone map to a 400-nit display against the CPU reference.
#[test]
fn output_pass_letterbox_and_tone_map() {
    use cp_core::color::{SCRGB_UNIT_NITS, bt2390_eetf};
    let Some(mut h) = Harness::new() else { return };
    let (w, sh, th) = (16u32, 8u32, 16u32);
    let src: Vec<[f32; 4]> = (0..sh)
        .flat_map(|y| {
            (0..w).map(move |x| [(x * 8 + y + 1) as f32 * 0.1, 0.5, (y + 1) as f32 * 0.3, 1.0])
        })
        .collect();
    let tex = h.upload(w, sh, &src);
    let target = Texture::new(
        &h.gpu,
        h.cmd,
        w,
        th,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    )
    .expect("target");
    let pass =
        OutputPass::new(&h.gpu, &h.layouts, vk::Format::R16G16B16A16_SFLOAT).expect("output");
    let attachments = [vk::RenderingAttachmentInfo::default()
        .image_view(target.view)
        .image_layout(vk::ImageLayout::GENERAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [9.0, 9.0, 9.0, 1.0],
            },
        })];
    let rendering = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: w,
                height: th,
            },
        })
        .layer_count(1)
        .color_attachments(&attachments);
    barrier(&h.gpu.device, h.cmd);
    // SAFETY: recording dynamic rendering into the offscreen target.
    unsafe { h.gpu.device.cmd_begin_rendering(h.cmd, &rendering) };
    let (src_peak, dst_peak) = (1000.0, 400.0);
    let params = OutputParams {
        display_hdr: true,
        display_peak_nits: dst_peak,
        sdr_white_nits: 203.0,
        content_peak_nits: src_peak,
        catmull_rom: false,
    };
    pass.record(
        &h.gpu,
        &h.layouts,
        &mut h.arena,
        h.cmd,
        Some(&tex),
        (w, th),
        &params,
    )
    .expect("record");
    // SAFETY: closes the rendering begun above.
    unsafe { h.gpu.device.cmd_end_rendering(h.cmd) };
    let got = h.readback(&target);
    for y in 0..th {
        for x in 0..w {
            let g = got[(y * w + x) as usize];
            let want = if (4..12).contains(&y) {
                let s = src[((y - 4) * w + x) as usize];
                let nits = [
                    s[0] * SCRGB_UNIT_NITS,
                    s[1] * SCRGB_UNIT_NITS,
                    s[2] * SCRGB_UNIT_NITS,
                ];
                let m = nits[0].max(nits[1]).max(nits[2]);
                let k = bt2390_eetf(m, src_peak, dst_peak) / m;
                nits.map(|n| n * k / SCRGB_UNIT_NITS)
            } else {
                [0.0; 3] // the shader paints the bars black over the clear
            };
            for k in 0..3 {
                assert!(
                    close(g[k], want[k]),
                    "({x},{y}) channel {k}: got {} want {}",
                    g[k],
                    want[k]
                );
            }
        }
    }
}

/// RTX VSR through NGX on Vulkan, end to end: a smooth 640×360 pattern
/// upscaled 2× must reproduce the pattern. Needs the NVIDIA Video Effects
/// runtime (`libnvidia-ngx-vsr.so`), so it only runs on request:
/// `CAPTUREPLAY_VFX_RUNTIME=<dir> cargo test -p cp-render vsr -- --ignored`.
#[test]
#[ignore = "needs the NVIDIA Video Effects runtime (CAPTUREPLAY_VFX_RUNTIME)"]
fn vsr_ngx_upscales_a_smooth_pattern() {
    use super::Upscaler;
    use super::renderer::RecordCtx;
    use super::upscale_ngx::NgxVsrUpscaler;

    let Some(mut h) = Harness::new() else { return };
    let dir = std::env::var_os("CAPTUREPLAY_VFX_RUNTIME").expect("CAPTUREPLAY_VFX_RUNTIME");
    let dirs: Vec<std::path::PathBuf> = [
        std::path::PathBuf::from(&dir).join("nvvfx").join("libs"),
        std::path::PathBuf::from(&dir),
    ]
    .into_iter()
    .filter(|d| cp_nvidia::ngx_vk::runtime_present(d))
    .collect();
    let data = std::env::temp_dir().join("captureplay-ngx-test");
    let mut up = NgxVsrUpscaler::new(
        &h.gpu,
        &h.layouts,
        dirs,
        data,
        cp_nvidia::ngx_vk::VsrQuality::High,
    )
    .expect("VSR available");
    let (sw, sh, dw, dh) = (640u32, 360u32, 1280u32, 720u32);
    // Smooth in both axes: the upscale has a well-defined expected value.
    let pattern = |u: f32, v: f32| [0.2 + 1.2 * u, 0.2 + 0.8 * v, 0.6 - 0.4 * u * v];
    let src: Vec<[f32; 4]> = (0..sh)
        .flat_map(|y| {
            (0..sw).map(move |x| {
                let p = pattern((x as f32 + 0.5) / sw as f32, (y as f32 + 0.5) / sh as f32);
                [p[0], p[1], p[2], 1.0]
            })
        })
        .collect();
    let t0 = std::time::Instant::now();
    while !up.prepare(&h.gpu, (sw, sh), (dw, dh)) {
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(60),
            "VSR did not load"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let tex = h.upload(sw, sh, &src);
    let out = Texture::new(
        &h.gpu,
        h.cmd,
        dw,
        dh,
        vk::Format::R16G16B16A16_SFLOAT,
        WORKING_USAGE,
    )
    .expect("out");
    let mut retire = Vec::new();
    let mut ctx = RecordCtx {
        gpu: &h.gpu,
        layouts: &h.layouts,
        arena: &mut h.arena,
        cmd: h.cmd,
        retire: &mut retire,
    };
    up.upscale(&mut ctx, &tex, &out).expect("upscale");
    let got = h.readback(&out);
    let mut err = 0.0f64;
    for y in 8..dh - 8 {
        for x in 8..dw - 8 {
            let want = pattern((x as f32 + 0.5) / dw as f32, (y as f32 + 0.5) / dh as f32);
            let g = got[(y * dw + x) as usize];
            err += (0..3)
                .map(|k| f64::from((g[k] - want[k]).abs()))
                .sum::<f64>();
        }
    }
    let mean = err / f64::from((dw - 16) * (dh - 16) * 3);
    eprintln!("VSR mean abs error {mean:.4} (scRGB units)");
    assert!(mean < 0.02, "VSR output does not match the pattern: {mean}");
}
