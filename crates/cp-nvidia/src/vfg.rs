//! NVIDIA Video Effects SDK "VideoFrameGeneration": AI frame interpolation
//! (RTX 40/50), bridged to D3D12 without CPU copies or CPU waits (see
//! [`crate::vfx`]).

use std::path::Path;

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource};

use crate::vfx::{self, Bridge, Effect, SharedImage, VfxError};

pub const EFFECT_DLL: &str = "nvVFXVideoFrameGeneration.dll";

/// Whether the VFG runtime is present in `dir`.
pub fn runtime_present(dir: &Path) -> bool {
    vfx::runtime_present(dir, &[EFFECT_DLL])
}

/// One VFG effect instance with two input slots and one output.
pub struct VfgSession {
    // Drop order: effect, images, then the bridge (stream, context).
    effect: Effect,
    inputs: [SharedImage; 2],
    output: SharedImage,
    /// Inputs currently bound as (previous, current) slots.
    bound: Option<(usize, usize)>,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    bridge: Bridge,
}

impl std::fmt::Debug for VfgSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VfgSession({}x{})", self.width, self.height)
    }
}

impl VfgSession {
    /// Creates the session. `mode`: 0 low, 1 medium, 2 high quality.
    pub fn new(
        device: &ID3D12Device,
        luid: LUID,
        runtime_dir: &Path,
        width: u32,
        height: u32,
        mode: u32,
    ) -> Result<Self, VfxError> {
        // Ada (8.9) or newer.
        let bridge = Bridge::new(device, luid, runtime_dir, EFFECT_DLL, (8, 9))?;
        let inputs = [
            bridge.image(device, width, height)?,
            bridge.image(device, width, height)?,
        ];
        let output = bridge.image(device, width, height)?;
        let effect = bridge.effect("VideoFrameGeneration")?;
        {
            let _g = bridge.push()?;
            effect.set_u32("InputWidth", width)?;
            effect.set_u32("InputHeight", height)?;
            effect.set_u32("Mode", mode.min(2))?;
            // The SDK's own detector forces a CPU wait per Run; cuts are
            // signaled explicitly instead (see `run`).
            effect.set_u32("AutomaticShotChangeDetectionEnabled", 0)?;
            effect.set_image("SrcImage0", &inputs[0])?;
            effect.set_image("SrcImage1", &inputs[1])?;
            effect.set_image("DstImage0", &output)?;
            effect.load()?;
        }
        tracing::info!(width, height, mode, "NVIDIA VFG session ready");
        Ok(Self {
            effect,
            pitch: output.pitch,
            inputs,
            output,
            bound: None,
            width,
            height,
            bridge,
        })
    }

    /// D3D12 buffer of input slot `i` (packed RGB10A2 rows of `pitch` bytes).
    pub fn input(&self, i: usize) -> &ID3D12Resource {
        self.inputs[i].resource()
    }

    pub fn output(&self) -> &ID3D12Resource {
        self.output.resource()
    }

    /// Fence the D3D12 queue signals when inputs are ready.
    pub fn producer(&self) -> &ID3D12Fence {
        self.bridge.producer()
    }

    /// Fence CUDA signals when the output is written.
    pub fn completion(&self) -> &ID3D12Fence {
        self.bridge.completion()
    }

    /// Enqueues one interpolation on the CUDA stream: waits for
    /// `producer >= value`, generates the frame at `t` between input slots
    /// `prev` and `next`, then signals `completion = value`. Never blocks the
    /// CPU. `shot_change` outputs `next` unchanged.
    pub fn run(
        &mut self,
        prev: usize,
        next: usize,
        t: f32,
        shot_change: bool,
        value: u64,
    ) -> Result<(), VfxError> {
        let _g = self.bridge.push()?;
        self.bridge.wait(value)?;
        let result = (|| {
            // Inputs must be rebound for every new pair, even if the
            // pointers are unchanged; rebinding within a pair restarts it.
            if self.bound != Some((prev, next)) {
                self.effect.set_image("SrcImage0", &self.inputs[prev])?;
                self.effect.set_image("SrcImage1", &self.inputs[next])?;
                self.effect.set_u32("ShotChange", u32::from(shot_change))?;
                self.bound = Some((prev, next));
            }
            self.effect.set_f32("Timestep", t.clamp(0.001, 0.999))?;
            self.effect.run()
        })();
        // Always signal so the D3D12 queue never deadlocks on failure.
        let signaled = self.bridge.signal(value);
        if result.is_err() {
            self.bound = None;
        }
        result?;
        signaled
    }

    /// Blocks until all CUDA work of the session is idle (tests/benchmarks).
    pub fn synchronize_context(&self) {
        self.bridge.synchronize_context();
    }

    /// Forces the next `run` to rebind its inputs (slot contents changed).
    pub fn invalidate_inputs(&mut self) {
        self.bound = None;
    }
}

impl Drop for VfgSession {
    fn drop(&mut self) {
        // Finish in-flight work before the effect and images go away.
        self.bridge.synchronize();
    }
}
