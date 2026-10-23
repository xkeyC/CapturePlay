//! GPU test harness: a device + command list without a window, texture
//! upload/readback in RGBA16F, and half-float conversion. Test-only.

use std::rc::Rc;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use crate::descriptors::{GpuDescriptorRing, StagingHeap};
use crate::gpu::{Gpu, borrow_com};
use crate::renderer::{FrameInput, Interpolator, RecordCtx};
use crate::texture::{Texture, TextureDesc, flush_barriers};

pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let mant = x & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    // Round to nearest.
    let m = (mant + 0x1000) >> 13;
    sign | (((exp as u16) << 10) + m as u16)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f32::from(h & 0x3ff);
    if exp == 0 {
        return sign * mant * 2f32.powi(-24);
    }
    if exp == 31 {
        return sign * f32::INFINITY;
    }
    sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
}

pub struct Harness {
    pub gpu: Gpu,
    pub heap: Rc<StagingHeap>,
    pub ring: GpuDescriptorRing,
    pub allocator: ID3D12CommandAllocator,
    pub list: ID3D12GraphicsCommandList,
    keep: Vec<ID3D12Resource>,
}

impl Harness {
    pub fn new() -> Self {
        let gpu = Gpu::new(false).expect("D3D12 device");
        let heap = Rc::new(
            StagingHeap::new(&gpu.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 256)
                .expect("heap"),
        );
        let ring = GpuDescriptorRing::new(&gpu.device, 1, 1024).expect("ring");
        // SAFETY: plain object creation.
        let (allocator, list) = unsafe {
            let a: ID3D12CommandAllocator = gpu
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("allocator");
            let l: ID3D12GraphicsCommandList = gpu
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)
                .expect("list");
            l.SetDescriptorHeaps(&[Some(ring.heap().clone())]);
            (a, l)
        };
        Self {
            gpu,
            heap,
            ring,
            allocator,
            list,
            keep: Vec::new(),
        }
    }

    pub fn ctx(&mut self) -> RecordCtx<'_> {
        RecordCtx {
            gpu: &self.gpu,
            heap: &self.heap,
            ring: &mut self.ring,
            list: &self.list,
            allocator: &self.allocator,
        }
    }

    /// Creates an RGBA16F texture filled from `rgba` (row-major).
    pub fn upload(&mut self, w: u32, h: u32, rgba: &[[f32; 4]]) -> Texture {
        let tex = Texture::new(
            &self.gpu,
            &self.heap,
            TextureDesc::new(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT).uav(),
        )
        .expect("texture");
        let pitch = (w as usize * 8).next_multiple_of(256);
        let buf = self
            .gpu
            .create_buffer(
                (pitch * h as usize) as u64,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .expect("upload");
        // SAFETY: mapped upload buffer of pitch*h bytes.
        unsafe {
            let mut p = std::ptr::null_mut();
            buf.Map(0, None, Some(&mut p)).expect("map");
            for y in 0..h as usize {
                let row = p.cast::<u8>().add(y * pitch).cast::<u16>();
                for x in 0..w as usize {
                    for (c, v) in rgba[y * w as usize + x].iter().enumerate() {
                        *row.add(x * 4 + c) = f32_to_f16(*v);
                    }
                }
            }
        }
        let mut b = Vec::new();
        tex.transition(&mut b, D3D12_RESOURCE_STATE_COPY_DEST);
        flush_barriers(&self.list, &mut b);
        self.copy(&buf, &tex, pitch, true);
        self.keep.push(buf);
        tex
    }

    fn copy(&self, buf: &ID3D12Resource, tex: &Texture, pitch: usize, to_texture: bool) {
        let tloc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrow_com(&tex.resource),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let bloc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrow_com(buf),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: tex.format,
                        Width: tex.width,
                        Height: tex.height,
                        Depth: 1,
                        RowPitch: pitch as u32,
                    },
                },
            },
        };
        // SAFETY: both locations reference live resources of matching size.
        unsafe {
            if to_texture {
                self.list.CopyTextureRegion(&tloc, 0, 0, 0, &bloc, None);
            } else {
                self.list.CopyTextureRegion(&bloc, 0, 0, 0, &tloc, None);
            }
        }
    }

    /// Records a readback of an RGBA16F texture; call `finish` then `read`.
    pub fn readback(&mut self, tex: &Texture) -> Readback {
        let pitch = (tex.width as usize * 8).next_multiple_of(256);
        let buf = self
            .gpu
            .create_buffer(
                (pitch * tex.height as usize) as u64,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .expect("readback");
        let mut b = Vec::new();
        tex.transition(&mut b, D3D12_RESOURCE_STATE_COPY_SOURCE);
        flush_barriers(&self.list, &mut b);
        self.copy(&buf, tex, pitch, false);
        Readback {
            buf,
            pitch,
            w: tex.width,
            h: tex.height,
        }
    }

    /// Submits everything recorded and waits for the GPU.
    pub fn finish(&mut self) {
        // SAFETY: list is open; wait for completion before reading back.
        unsafe {
            self.list.Close().expect("close");
            self.gpu
                .queue
                .ExecuteCommandLists(&[Some(self.list.cast().expect("cast"))]);
        }
        self.gpu.wait_idle().expect("wait");
    }
}

pub struct Readback {
    buf: ID3D12Resource,
    pitch: usize,
    pub w: u32,
    pub h: u32,
}

impl Readback {
    pub fn read(&self) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity((self.w * self.h) as usize);
        // SAFETY: GPU finished (Harness::finish); buffer is pitch*h bytes.
        unsafe {
            let mut p = std::ptr::null_mut();
            self.buf.Map(0, None, Some(&mut p)).expect("map");
            for y in 0..self.h as usize {
                let row = p.cast::<u8>().add(y * self.pitch).cast::<u16>();
                for x in 0..self.w as usize {
                    out.push(std::array::from_fn(|c| f16_to_f32(*row.add(x * 4 + c))));
                }
            }
        }
        out
    }
}

#[test]
fn half_float_round_trip() {
    for v in [0.0f32, 0.25, 1.0, 3.5, 12.5, -2.0] {
        assert!((f16_to_f32(f32_to_f16(v)) - v).abs() < 1e-3 * v.abs().max(1.0));
    }
}

// ------------------------------------------------- ground-truth test scenes

pub(crate) fn hash(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374_761_393) ^ (y as u32).wrapping_mul(668_265_263);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (h ^ (h >> 16)) as f32 / u32::MAX as f32
}

/// Smooth value noise, continuous in (x, y).
pub(crate) fn noise(x: f32, y: f32, cell: f32) -> f32 {
    let (fx, fy) = (x / cell, y / cell);
    let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (tx, ty) = (s(fx - fx.floor()), s(fy - fy.floor()));
    let a = hash(ix, iy) + (hash(ix + 1, iy) - hash(ix, iy)) * tx;
    let b = hash(ix, iy + 1) + (hash(ix + 1, iy + 1) - hash(ix, iy + 1)) * tx;
    a + (b - a) * ty
}

/// Textured test image in scRGB (0..~2, i.e. up to ~160 nits).
pub(crate) fn pattern(x: f32, y: f32) -> f32 {
    2.0 * (0.5 * noise(x, y, 31.0) + 0.3 * noise(x, y, 9.0) + 0.2 * noise(x, y, 3.0))
}

pub(crate) fn image(w: u32, h: u32, f: impl Fn(f32, f32) -> f32) -> Vec<[f32; 4]> {
    let mut v = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = f(x as f32 + 0.5, y as f32 + 0.5);
            v.push([p, p, p, 1.0]);
        }
    }
    v
}

pub(crate) fn psnr(a: &[[f32; 4]], b: &[[f32; 4]], w: u32, h: u32, margin: u32) -> f64 {
    let (mut se, mut n) = (0f64, 0f64);
    for y in margin..h - margin {
        for x in margin..w - margin {
            let i = (y * w + x) as usize;
            let d = f64::from(a[i][0] - b[i][0]) / 2.0; // normalize peak 2 → 1
            se += d * d;
            n += 1.0;
        }
    }
    10.0 * (1.0 / (se / n)).log10()
}

/// Static background, textured square of side 320 moving by (dx, dy):
/// the disocclusion case that produces ghosting.
pub(crate) fn occlusion_scene(x: f32, y: f32, ox: f32, oy: f32) -> f32 {
    let (cx, cy) = (800.0 + ox, 380.0 + oy);
    if (cx..cx + 320.0).contains(&x) && (cy..cy + 320.0).contains(&y) {
        // Foreground texture moves with the square; brighter than background.
        0.6 + 1.4 * noise(x - ox + 1000.0, y - oy + 1000.0, 11.0)
    } else {
        0.6 * pattern(x, y)
    }
}

/// PSNR restricted to the band around the moving square's edges.
pub(crate) fn psnr_band(a: &[[f32; 4]], b: &[[f32; 4]], w: u32, dx: f32, dy: f32) -> f64 {
    let (mut se, mut n) = (0f64, 0f64);
    let (x0, y0) = (800.0 + dx / 2.0, 380.0 + dy / 2.0);
    let band = dx.abs().max(dy.abs()) + 8.0;
    for y in 300..800u32 {
        for x in 700..1300u32 {
            let (fx, fy) = (x as f32, y as f32);
            let near_x = (fx - x0).abs() < band || (fx - x0 - 320.0).abs() < band;
            let near_y = (fy - y0).abs() < band || (fy - y0 - 320.0).abs() < band;
            let inside = fx > x0 - band
                && fx < x0 + 320.0 + band
                && fy > y0 - band
                && fy < y0 + 320.0 + band;
            if inside && (near_x || near_y) {
                let i = (y * w + x) as usize;
                let d = f64::from(a[i][0] - b[i][0]) / 2.0;
                se += d * d;
                n += 1.0;
            }
        }
    }
    10.0 * (1.0 / (se / n)).log10()
}

pub(crate) fn occlusion_with(
    make: &dyn Fn(&Harness) -> Box<dyn Interpolator>,
    w: u32,
    h: u32,
    dx: f32,
    dy: f32,
) -> (f64, f64, f64, f64) {
    let mut hs = Harness::new();
    let a_img = image(w, h, |x, y| occlusion_scene(x, y, 0.0, 0.0));
    let b_img = image(w, h, |x, y| occlusion_scene(x, y, dx, dy));
    let truth = image(w, h, |x, y| occlusion_scene(x, y, dx / 2.0, dy / 2.0));
    let a = hs.upload(w, h, &a_img);
    let b = hs.upload(w, h, &b_img);
    let out = Texture::new(
        &hs.gpu,
        &hs.heap,
        TextureDesc::new(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT).uav(),
    )
    .expect("out");
    let mut interp = make(&hs);
    interp
        .interpolate(
            &mut hs.ctx(),
            FrameInput { id: 1, tex: &a },
            FrameInput { id: 2, tex: &b },
            0.5,
            &out,
        )
        .expect("interpolate");
    let rb = hs.readback(&out);
    hs.finish();
    let img = rb.read();
    if let Ok(dir) = std::env::var("CAPTUREPLAY_TEST_DUMP") {
        // Side-by-side crop (interp | truth) around the object, as PGM.
        let (x0, y0, cw, ch) = (700u32, 300u32, 600u32, 500u32);
        let mut pgm = format!(
            "P5 {} {} 255
",
            cw * 2,
            ch
        )
        .into_bytes();
        for y in y0..y0 + ch {
            for src in [&img, &truth] {
                for x in x0..x0 + cw {
                    pgm.push((src[(y * w + x) as usize][0] / 2.0 * 255.0).clamp(0.0, 255.0) as u8);
                }
            }
        }
        let tag: String = interp
            .name()
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect();
        let _ = std::fs::write(format!("{dir}/occ_{tag}_{w}_{dx}_{dy}.pgm"), pgm);
    }
    let blend: Vec<[f32; 4]> = a_img
        .iter()
        .zip(&b_img)
        .map(|(p, q)| std::array::from_fn(|c| 0.5 * (p[c] + q[c])))
        .collect();
    let r = (
        psnr(&img, &truth, w, h, 16),
        psnr(&blend, &truth, w, h, 16),
        psnr_band(&img, &truth, w, dx, dy),
        psnr_band(&blend, &truth, w, dx, dy),
    );
    drop(interp);
    drop((out, a, b, rb));
    r
}
