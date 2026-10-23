//! Frame interpolation with NVIDIA's AI Video Frame Generation (RTX 40/50).
//!
//! Per output frame: encode the frames of the pair that are not cached yet
//! straight into the CUDA-shared buffers, hand over via a shared fence, let
//! VFG generate the frame at `t` on its CUDA stream, and decode the result
//! back into the working frame once the completion fence is reached — all
//! GPU-side, no CPU waits.

use std::path::PathBuf;
use std::rc::Rc;

use cp_nvidia::vfg::VfgSession;
use cp_nvidia::vfx::VfxError;
use windows::Win32::Foundation::{E_FAIL, E_NOTIMPL};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

use crate::descriptors::{CpuDescriptor, StagingHeap, raw_buffer_view};
use crate::gpu::{Gpu, transition};
use crate::renderer::{FrameInput, Interpolator, RecordCtx};
use crate::shaders::{self, ComputePipeline, dispatch_2d, set_compute_constants};
use crate::texture::{Texture, flush_barriers};

/// Root constants of `vfg.hlsl` (also used by the VSR upscaler).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Constants {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    /// 1 = PQ / BT.2020 encoding, 0 = sRGB relative to `sdr_white`.
    pub hdr: u32,
    pub sdr_white: f32,
    pub pad: [f32; 3],
}

fn vfg_err(e: VfxError) -> windows::core::Error {
    windows::core::Error::new(E_FAIL, e.to_string())
}

struct Resources {
    /// Raw UAVs of the two input buffers and a raw SRV of the output.
    input_uav: [CpuDescriptor; 2],
    output_srv: CpuDescriptor,
    /// Frame id held by each input slot.
    slot_ids: [Option<u64>; 2],
    fence_value: u64,
    heap: Rc<StagingHeap>,
    session: VfgSession,
}

impl Drop for Resources {
    fn drop(&mut self) {
        for d in self
            .input_uav
            .iter()
            .chain(std::iter::once(&self.output_srv))
        {
            self.heap.free(*d);
        }
    }
}

pub struct VfgInterpolator {
    encode: ComputePipeline,
    decode: ComputePipeline,
    runtime_dir: PathBuf,
    mode: u32,
    hdr: bool,
    sdr_white: f32,
    res: Option<Resources>,
}

impl std::fmt::Debug for VfgInterpolator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VfgInterpolator")
            .field("runtime_dir", &self.runtime_dir)
            .field("mode", &self.mode)
            .finish()
    }
}

impl VfgInterpolator {
    /// `mode`: 0 low, 1 medium, 2 high. Fails fast if the runtime is absent.
    pub fn new(gpu: &Gpu, runtime_dir: PathBuf, mode: u32) -> Result<Self> {
        if !gpu.is_nvidia() || !cp_nvidia::vfg::runtime_present(&runtime_dir) {
            return Err(windows::core::Error::new(
                E_NOTIMPL,
                format!("VFG runtime not found in {}", runtime_dir.display()),
            ));
        }
        Ok(Self {
            encode: ComputePipeline::new(&gpu.device, shaders::VFG_ENCODE)?,
            decode: ComputePipeline::new(&gpu.device, shaders::VFG_DECODE)?,
            runtime_dir,
            mode,
            hdr: false,
            sdr_white: 1.0,
            res: None,
        })
    }

    fn ensure(&mut self, gpu: &Gpu, heap: &Rc<StagingHeap>, w: u32, h: u32) -> Result<()> {
        if self
            .res
            .as_ref()
            .is_some_and(|r| r.session.width == w && r.session.height == h)
        {
            return Ok(());
        }
        self.res = None;
        let session = VfgSession::new(
            &gpu.device,
            gpu.adapter_luid,
            &self.runtime_dir,
            w,
            h,
            self.mode,
        )
        .map_err(vfg_err)?;
        let words = session.pitch / 4 * h;
        let raw = |r: &ID3D12Resource, uav: bool| raw_buffer_view(&gpu.device, heap, r, words, uav);
        let input_uav = [raw(session.input(0), true)?, raw(session.input(1), true)?];
        let output_srv = raw(session.output(), false)?;
        self.res = Some(Resources {
            input_uav,
            output_srv,
            slot_ids: [None, None],
            fence_value: 0,
            heap: heap.clone(),
            session,
        });
        Ok(())
    }

    fn constants(&self, res: &Resources) -> Constants {
        Constants {
            width: res.session.width,
            height: res.session.height,
            pitch: res.session.pitch,
            hdr: u32::from(self.hdr),
            sdr_white: self.sdr_white,
            pad: [0.0; 3],
        }
    }

    fn record_encode(
        &self,
        ctx: &mut RecordCtx<'_>,
        res: &Resources,
        frame: &Texture,
        slot: usize,
    ) {
        let buffer = res.session.input(slot);
        let mut b = Vec::new();
        frame.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        b.push(transition(
            buffer,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ));
        flush_barriers(ctx.list, &mut b);
        self.encode.bind(ctx.list);
        let srv = ctx.ring.table(&ctx.gpu.device, &[frame.srv.handle]);
        let uav = ctx
            .ring
            .table(&ctx.gpu.device, &[res.input_uav[slot].handle]);
        set_compute_constants(ctx.list, &self.constants(res));
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, res.session.width, res.session.height);
        // CUDA reads it next: hand over in COMMON.
        b.push(transition(
            buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        ));
        flush_barriers(ctx.list, &mut b);
    }
}

impl Interpolator for VfgInterpolator {
    fn name(&self) -> &'static str {
        "NVIDIA VFG (AI)"
    }

    fn set_content(&mut self, hdr: bool, sdr_white: f32) {
        if hdr != self.hdr || (sdr_white - self.sdr_white).abs() > 1e-3 {
            self.hdr = hdr;
            self.sdr_white = sdr_white;
            // Cached encodings used the old mapping.
            if let Some(r) = &mut self.res {
                r.slot_ids = [None, None];
                r.session.invalidate_inputs();
            }
        }
    }

    fn interpolate(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        prev: FrameInput<'_>,
        next: FrameInput<'_>,
        t: f32,
        out: &Texture,
    ) -> Result<()> {
        self.ensure(ctx.gpu, ctx.heap, next.tex.width, next.tex.height)?;
        let mut res = self.res.take().ok_or_else(windows::core::Error::empty)?;
        let result = (|| {
            let slot_of = |id: u64| res.slot_ids.iter().position(|s| *s == Some(id));
            let prev_slot =
                slot_of(prev.id).unwrap_or_else(|| if slot_of(next.id) == Some(0) { 1 } else { 0 });
            let next_slot = 1 - prev_slot;
            let mut encoded = false;
            if res.slot_ids[prev_slot] != Some(prev.id) {
                self.record_encode(ctx, &res, prev.tex, prev_slot);
                res.slot_ids[prev_slot] = Some(prev.id);
                encoded = true;
            }
            if res.slot_ids[next_slot] != Some(next.id) {
                self.record_encode(ctx, &res, next.tex, next_slot);
                res.slot_ids[next_slot] = Some(next.id);
                encoded = true;
            }
            if encoded {
                res.session.invalidate_inputs();
            }

            // Hand the encoded inputs to CUDA, and wait for its result.
            ctx.submit_and_continue()?;
            res.fence_value += 1;
            let v = res.fence_value;
            // SAFETY: valid queue and shared fence.
            unsafe { ctx.gpu.queue.Signal(res.session.producer(), v)? };
            res.session
                .run(prev_slot, next_slot, t, false, v)
                .map_err(vfg_err)?;
            ctx.queue_wait(res.session.completion(), v)?;

            // Decode into the working-format output.
            let mut b = Vec::new();
            b.push(transition(
                res.session.output(),
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            ));
            out.transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            flush_barriers(ctx.list, &mut b);
            self.decode.bind(ctx.list);
            let srv = ctx.ring.table(&ctx.gpu.device, &[res.output_srv.handle]);
            let uav = ctx.ring.table(&ctx.gpu.device, &[out.uav_handle()]);
            set_compute_constants(ctx.list, &self.constants(&res));
            // SAFETY: tables live in the bound heap.
            unsafe {
                ctx.list.SetComputeRootDescriptorTable(1, srv);
                ctx.list.SetComputeRootDescriptorTable(2, uav);
            }
            dispatch_2d(ctx.list, res.session.width, res.session.height);
            b.push(transition(
                res.session.output(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            flush_barriers(ctx.list, &mut b);
            Ok(())
        })();
        self.res = Some(res);
        result
    }
}

/// Ground-truth occlusion scene (moving object over a static background,
/// see `testutil`); needs the
/// VFG runtime in `runtime/nvidia-vfx`. Run with
/// `cargo test -p cp-render -- --ignored --nocapture`.
#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::testutil::{Harness, occlusion_with};

    #[test]
    #[ignore = "requires an RTX 40/50 GPU and the VFG runtime"]
    fn occlusion_moving_object_vfg() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runtime/nvidia-vfx");
        for mode in 0..3u32 {
            for (w, h) in [(1920, 1080), (2560, 1440)] {
                for (dx, dy) in [(24.0, 0.0), (60.0, 20.0)] {
                    let make = |hs: &Harness| -> Box<dyn Interpolator> {
                        let mut i =
                            VfgInterpolator::new(&hs.gpu, dir.clone(), mode).expect("VFG runtime");
                        // Test images span 0..2 scRGB: use that as SDR white.
                        i.set_content(false, 2.0);
                        Box::new(i)
                    };
                    let (all, all_blend, edge, edge_blend) = occlusion_with(&make, w, h, dx, dy);
                    println!(
                        "VFG mode {mode} {w}x{h} object motion ({dx},{dy}): full {all:.1} dB (blend {all_blend:.1}) | edges {edge:.1} dB (blend {edge_blend:.1})"
                    );
                }
            }
        }
    }
}

/// Is `completion` (signaled on our stream after `NvVFX_Run`) really ordered
/// after the effect's GPU work? Compare the stream-ordered completion time
/// with a full context synchronize.
#[cfg(test)]
mod sync_tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use cp_nvidia::vfg::VfgSession;

    use crate::testutil::Harness;

    #[test]
    #[ignore = "requires an RTX 40/50 GPU and the VFX runtime"]
    fn completion_fence_covers_vfg_work() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runtime/nvidia-vfx");
        let hs = Harness::new();
        let mut s =
            VfgSession::new(&hs.gpu.device, hs.gpu.adapter_luid, &dir, 2560, 1440, 1).expect("VFG");
        // SAFETY: plain event creation.
        let event =
            unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
                .expect("event");
        let (mut fence_ms, mut ctx_ms) = (Vec::new(), Vec::new());
        for v in 1..=12u64 {
            // SAFETY: valid queue/fence/event.
            unsafe { hs.gpu.queue.Signal(s.producer(), v).expect("signal") };
            let t0 = Instant::now();
            s.run(0, 1, 0.5, false, v).expect("run");
            // Time until the completion fence fires (what D3D12 waits on)…
            // SAFETY: valid fence and event.
            unsafe {
                s.completion()
                    .SetEventOnCompletion(v, event)
                    .expect("event");
                windows::Win32::System::Threading::WaitForSingleObject(event, 5000);
            }
            let t1 = Instant::now();
            // …versus until all CUDA work is really done.
            s.synchronize_context();
            let t2 = Instant::now();
            if v > 2 {
                fence_ms.push((t1 - t0).as_secs_f64() * 1e3);
                ctx_ms.push((t2 - t0).as_secs_f64() * 1e3);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!(
            "VFG 1440p medium: completion fence after {:.2} ms, all CUDA work after {:.2} ms",
            med(&mut fence_ms),
            med(&mut ctx_ms)
        );
    }
}
