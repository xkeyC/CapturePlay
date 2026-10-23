//! Frame interpolation on NVIDIA Optical Flow hardware.
//!
//! Per new frame pair `(A, B)` (once, however many in-between frames):
//! 1. `luma`: B → 8-bit perceptual luma at flow resolution (A's is cached
//!    from the previous pair).
//! 2. NVOFA (own engine, fence-synchronized) computes A→B and B→A flow plus
//!    cost in a single bidirectional execute.
//! 3. `densify`: vector-median cleanup + confidence for both directions.
//!
//! Per output frame: one `warp` dispatch at time `t`.
//!
//! GPU cost at 1440p on an RTX 4090 is ~1-2 ms, so the interpolated frame is
//! ready almost immediately after B arrives.

use std::rc::Rc;

use cp_nvidia::nvof::{self, Buffer, FencePoint, PerfLevel, Session, SessionDesc};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Result;

use crate::descriptors::StagingHeap;
use crate::gpu::Gpu;
use crate::renderer::{FrameInput, Interpolator, RecordCtx};
use crate::shaders::{self, ComputePipeline, dispatch_2d, set_compute_constants};
use crate::texture::{Texture, TextureDesc, flush_barriers};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Constants {
    dst_size: [u32; 2],
    src_size: [u32; 2],
    t: f32,
    flow_scale: f32,
    inv_grid: [f32; 2],
    inv_src: [f32; 2],
    grid_size: u32,
    max_cost: f32,
    sigma: f32,
    pad: [f32; 3],
}

/// Flow textures registered with NVOFA.
struct FlowSet {
    raw: Texture,
    cost: Texture,
    dense: Texture,
    raw_buf: Buffer,
    cost_buf: Buffer,
}

struct Resources {
    work: (u32, u32),
    flow_res: (u32, u32),
    grid: u32,
    grid_extent: (u32, u32),
    /// Two luma slots; `luma_ids[i]` is the frame id each holds.
    luma: [Texture; 2],
    luma_buf: [Buffer; 2],
    luma_ids: [Option<u64>; 2],
    /// Pair whose flow is currently in `fwd`/`bwd`.
    flow_pair: Option<(u64, u64)>,
    fwd: FlowSet,
    bwd: FlowSet,
    of_fence: ID3D12Fence,
    of_value: u64,
    // Last: unregisters everything before the textures above are released.
    session: Session,
}

pub struct NvofInterpolator {
    luma_pass: ComputePipeline,
    densify_pass: ComputePipeline,
    warp_pass: ComputePipeline,
    res: Option<Resources>,
    perf: PerfLevel,
}

impl std::fmt::Debug for NvofInterpolator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvofInterpolator")
            .field("ready", &self.res.is_some())
            .finish()
    }
}

impl NvofInterpolator {
    /// Fails when the NVOFA runtime is unavailable (non-NVIDIA GPU, old driver).
    pub fn new(gpu: &Gpu) -> Result<Self> {
        if !gpu.is_nvidia() || !nvof::is_available() {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_NOTIMPL,
                "NVIDIA Optical Flow not available",
            ));
        }
        Ok(Self {
            luma_pass: ComputePipeline::new(&gpu.device, shaders::INTERP_LUMA)?,
            densify_pass: ComputePipeline::new(&gpu.device, shaders::INTERP_DENSIFY)?,
            warp_pass: ComputePipeline::new(&gpu.device, shaders::INTERP_WARP)?,
            res: None,
            perf: PerfLevel::Fast,
        })
    }

    fn ensure_resources(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        work: (u32, u32),
    ) -> Result<()> {
        if self.res.as_ref().is_some_and(|r| r.work == work) {
            return Ok(());
        }
        self.res = None;
        // Flow at ≤1920 px wide: 1440p/4K content is matched at half resolution,
        // which NVOFA handles better (larger search range per pixel) and
        // twice as fast.
        let ds = if work.0 > 1920 { 2 } else { 1 };
        let flow_res = (work.0 / ds, work.1 / ds);
        let err = |e: nvof::NvofError| {
            windows::core::Error::new(windows::Win32::Foundation::E_FAIL, e.to_string())
        };
        let mut session = Session::new(
            &gpu.device,
            SessionDesc {
                width: flow_res.0,
                height: flow_res.1,
                grid: 2,
                perf: self.perf,
            },
        )
        .map_err(err)?;
        let grid = session.grid();
        let grid_extent = (flow_res.0.div_ceil(grid), flow_res.1.div_ceil(grid));

        // SAFETY: valid device.
        let of_fence: ID3D12Fence = unsafe { gpu.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let mut of_value = 0u64;
        let ready = gpu.signal()?;
        let mut reg = |s: &mut Session, r: &ID3D12Resource| -> Result<Buffer> {
            of_value += 1;
            s.register(
                r,
                FencePoint {
                    fence: &gpu.fence,
                    value: ready,
                },
                FencePoint {
                    fence: &of_fence,
                    value: of_value,
                },
            )
            .map_err(err)
        };

        let luma_desc = TextureDesc::new(flow_res.0, flow_res.1, DXGI_FORMAT_R8_UNORM).uav();
        let luma = [
            Texture::new(gpu, heap, luma_desc)?,
            Texture::new(gpu, heap, luma_desc)?,
        ];
        let luma_buf = [
            reg(&mut session, &luma[0].resource)?,
            reg(&mut session, &luma[1].resource)?,
        ];
        let mut flow_set = |s: &mut Session| -> Result<FlowSet> {
            let raw = Texture::new(
                gpu,
                heap,
                TextureDesc::new(grid_extent.0, grid_extent.1, DXGI_FORMAT_R16G16_SINT),
            )?;
            let cost = Texture::new(
                gpu,
                heap,
                TextureDesc::new(grid_extent.0, grid_extent.1, DXGI_FORMAT_R8_UINT),
            )?;
            let dense = Texture::new(
                gpu,
                heap,
                TextureDesc::new(grid_extent.0, grid_extent.1, DXGI_FORMAT_R16G16B16A16_FLOAT)
                    .uav(),
            )?;
            let raw_buf = reg(s, &raw.resource)?;
            let cost_buf = reg(s, &cost.resource)?;
            Ok(FlowSet {
                raw,
                cost,
                dense,
                raw_buf,
                cost_buf,
            })
        };
        let fwd = flow_set(&mut session)?;
        let bwd = flow_set(&mut session)?;
        // Our queue must not touch the textures before NVOFA finished
        // registering them.
        // SAFETY: valid queue and fence.
        unsafe { gpu.queue.Wait(&of_fence, of_value)? };
        tracing::info!(?work, ?flow_res, grid, ?grid_extent, perf = ?self.perf, "NvOF interpolator ready");
        self.res = Some(Resources {
            work,
            flow_res,
            grid,
            grid_extent,
            luma,
            luma_buf,
            luma_ids: [None, None],
            flow_pair: None,
            fwd,
            bwd,
            of_fence,
            of_value,
            session,
        });
        Ok(())
    }
}

impl Resources {
    /// Slot holding the luma of frame `id`, if cached.
    fn luma_slot(&self, id: u64) -> Option<usize> {
        self.luma_ids.iter().position(|x| *x == Some(id))
    }
}

impl NvofInterpolator {
    fn record_luma(&self, ctx: &mut RecordCtx<'_>, res: &Resources, src: &Texture, slot: usize) {
        let mut b = Vec::new();
        src.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        res.luma[slot].transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.luma_pass.bind(ctx.list);
        let h = src.srv.handle;
        let srv = ctx.ring.table(&ctx.gpu.device, &[h, h, h, h]);
        let uav = ctx
            .ring
            .table(&ctx.gpu.device, &[res.luma[slot].uav_handle()]);
        set_compute_constants(
            ctx.list,
            &Constants {
                dst_size: [res.flow_res.0, res.flow_res.1],
                src_size: [res.work.0, res.work.1],
                ..Default::default()
            },
        );
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, res.flow_res.0, res.flow_res.1);
        // NVOFA reads it from its own queue: hand over in COMMON.
        res.luma[slot].transition(&mut b, D3D12_RESOURCE_STATE_COMMON);
        flush_barriers(ctx.list, &mut b);
    }

    fn record_densify(&self, ctx: &mut RecordCtx<'_>, res: &Resources, set: &FlowSet, ds: f32) {
        let mut b = Vec::new();
        set.raw
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        set.cost
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        set.dense
            .transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.densify_pass.bind(ctx.list);
        let srv = ctx.ring.table(
            &ctx.gpu.device,
            &[
                set.raw.srv.handle,
                set.cost.srv.handle,
                set.raw.srv.handle,
                set.raw.srv.handle,
            ],
        );
        let uav = ctx.ring.table(&ctx.gpu.device, &[set.dense.uav_handle()]);
        let ext = [res.grid_extent.0, res.grid_extent.1];
        set_compute_constants(
            ctx.list,
            &Constants {
                dst_size: ext,
                src_size: ext,
                flow_scale: ds,
                grid_size: res.grid,
                max_cost: 192.0,
                ..Default::default()
            },
        );
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, ext[0], ext[1]);
        set.raw.transition(&mut b, D3D12_RESOURCE_STATE_COMMON);
        set.cost.transition(&mut b, D3D12_RESOURCE_STATE_COMMON);
        set.dense
            .transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        flush_barriers(ctx.list, &mut b);
    }
}

impl Interpolator for NvofInterpolator {
    fn name(&self) -> &'static str {
        "NVIDIA Optical Flow"
    }

    fn interpolate(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        prev: FrameInput<'_>,
        next: FrameInput<'_>,
        t: f32,
        out: &Texture,
    ) -> Result<()> {
        self.ensure_resources(ctx.gpu, ctx.heap, (next.tex.width, next.tex.height))?;
        let mut res = self.res.take().ok_or_else(windows::core::Error::empty)?;
        let result = self.interpolate_with(ctx, &mut res, prev, next, t, out);
        self.res = Some(res);
        result
    }
}

impl NvofInterpolator {
    #[allow(clippy::too_many_arguments)]
    fn interpolate_with(
        &self,
        ctx: &mut RecordCtx<'_>,
        res: &mut Resources,
        prev: FrameInput<'_>,
        next: FrameInput<'_>,
        t: f32,
        out: &Texture,
    ) -> Result<()> {
        let ds = res.work.0 as f32 / res.flow_res.0 as f32;
        if res.flow_pair != Some((prev.id, next.id)) {
            // Luma for both frames; `prev` is normally cached from last pair.
            let prev_slot = match res.luma_slot(prev.id) {
                Some(s) => s,
                None => {
                    self.record_luma(ctx, res, prev.tex, 0);
                    res.luma_ids[0] = Some(prev.id);
                    0
                }
            };
            let next_slot = 1 - prev_slot;
            if res.luma_slot(next.id) != Some(next_slot) {
                self.record_luma(ctx, res, next.tex, next_slot);
                res.luma_ids[next_slot] = Some(next.id);
            }
            res.flow_pair = Some((prev.id, next.id));

            // Hand over to NVOFA and make our queue wait for its result.
            let ready = ctx.submit_and_continue()?;
            res.of_value += 1;
            res.session
                .execute(
                    res.luma_buf[prev_slot],
                    res.luma_buf[next_slot],
                    res.fwd.raw_buf,
                    res.fwd.cost_buf,
                    res.bwd.raw_buf,
                    res.bwd.cost_buf,
                    FencePoint {
                        fence: &ctx.gpu.fence,
                        value: ready,
                    },
                    FencePoint {
                        fence: &res.of_fence,
                        value: res.of_value,
                    },
                )
                .map_err(|e| {
                    windows::core::Error::new(windows::Win32::Foundation::E_FAIL, e.to_string())
                })?;
            ctx.queue_wait(&res.of_fence, res.of_value)?;
            self.record_densify(ctx, res, &res.fwd, ds);
            self.record_densify(ctx, res, &res.bwd, ds);
        }

        let (prev, next) = (prev.tex, next.tex);
        // Warp.
        let mut b = Vec::new();
        prev.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        next.transition(&mut b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        out.transition(&mut b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(ctx.list, &mut b);
        self.warp_pass.bind(ctx.list);
        let srv = ctx.ring.table(
            &ctx.gpu.device,
            &[
                prev.srv.handle,
                next.srv.handle,
                res.fwd.dense.srv.handle,
                res.bwd.dense.srv.handle,
            ],
        );
        let uav = ctx.ring.table(&ctx.gpu.device, &[out.uav_handle()]);
        let cell = res.grid as f32 * ds;
        set_compute_constants(
            ctx.list,
            &Constants {
                dst_size: [res.work.0, res.work.1],
                src_size: [res.work.0, res.work.1],
                t,
                inv_grid: [
                    1.0 / (res.grid_extent.0 as f32 * cell),
                    1.0 / (res.grid_extent.1 as f32 * cell),
                ],
                inv_src: [1.0 / res.work.0 as f32, 1.0 / res.work.1 as f32],
                sigma: 1.5 * ds + 1.0,
                ..Default::default()
            },
        );
        // SAFETY: tables live in the bound heap.
        unsafe {
            ctx.list.SetComputeRootDescriptorTable(1, srv);
            ctx.list.SetComputeRootDescriptorTable(2, uav);
        }
        dispatch_2d(ctx.list, res.work.0, res.work.1);
        Ok(())
    }
}
