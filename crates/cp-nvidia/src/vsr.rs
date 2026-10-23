//! NVIDIA Video Effects SDK "VideoSuperRes": RTX Video Super Resolution
//! (AI upscaling with compression-artifact reduction), bridged to D3D12 the
//! same way as VFG (see [`crate::vfx`]). Purely spatial: no frame history,
//! no motion vectors.

use std::path::Path;

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource};

use crate::vfx::{self, Bridge, Effect, SharedImage, VfxError};

/// Effect DLL; it loads the model runtime `nvngx_vsr.dll` from the same folder.
pub const EFFECT_DLL: &str = "nvVFXVideoSuperRes.dll";
pub const MODEL_DLL: &str = "nvngx_vsr.dll";

/// Quality levels accepted by the effect (`QualityLevel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsrQuality {
    /// Non-AI bicubic (reference).
    Bicubic = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Ultra = 4,
    /// Tuned for clean, high-bitrate sources.
    HighBitrateLow = 16,
    HighBitrateMedium = 17,
    HighBitrateHigh = 18,
    HighBitrateUltra = 19,
    /// Tuned for low-latency streaming (Ampere+).
    StreamingMedium = 21,
    StreamingUltra = 23,
}

/// Whether the VSR runtime is present in `dir`.
pub fn runtime_present(dir: &Path) -> bool {
    vfx::runtime_present(dir, &[EFFECT_DLL, MODEL_DLL])
}

/// One upscaler instance: `in_w × in_h` → `out_w × out_h`, 10-bit.
pub struct VsrSession {
    // Drop order: effect, images, then the bridge.
    effect: Effect,
    input: SharedImage,
    output: SharedImage,
    bridge: Bridge,
}

impl std::fmt::Debug for VsrSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VsrSession({}x{} -> {}x{})",
            self.input.width, self.input.height, self.output.width, self.output.height
        )
    }
}

impl VsrSession {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ID3D12Device,
        luid: LUID,
        runtime_dir: &Path,
        (in_w, in_h): (u32, u32),
        (out_w, out_h): (u32, u32),
        quality: VsrQuality,
        strength: f32,
    ) -> Result<Self, VfxError> {
        // Turing (7.5) or newer; the streaming models need Ampere (8.6).
        let min = if matches!(
            quality,
            VsrQuality::StreamingMedium | VsrQuality::StreamingUltra
        ) {
            (8, 6)
        } else {
            (7, 5)
        };
        if !runtime_present(runtime_dir) {
            return Err(VfxError::MissingRuntime(runtime_dir.join(MODEL_DLL)));
        }
        let bridge = Bridge::new(device, luid, runtime_dir, EFFECT_DLL, min)?;
        let input = bridge.image(device, in_w, in_h)?;
        let output = bridge.image(device, out_w, out_h)?;
        let effect = bridge.effect("VideoSuperRes")?;
        {
            let _g = bridge.push()?;
            effect.set_u32("ImageEncodingMode", 1)?; // 10-bit RGB10A2
            effect.set_u32("QualityLevel", quality as u32)?;
            effect.set_f32("Strength", strength.clamp(0.0, 1.0))?;
            effect.set_image("SrcImage0", &input)?;
            effect.set_image("DstImage0", &output)?;
            effect.load()?;
        }
        tracing::info!(
            in_w,
            in_h,
            out_w,
            out_h,
            ?quality,
            "NVIDIA VSR session ready"
        );
        Ok(Self {
            effect,
            input,
            output,
            bridge,
        })
    }

    pub fn input(&self) -> &SharedImage {
        &self.input
    }

    pub fn output(&self) -> &SharedImage {
        &self.output
    }

    pub fn input_resource(&self) -> &ID3D12Resource {
        self.input.resource()
    }

    pub fn output_resource(&self) -> &ID3D12Resource {
        self.output.resource()
    }

    pub fn producer(&self) -> &ID3D12Fence {
        self.bridge.producer()
    }

    pub fn completion(&self) -> &ID3D12Fence {
        self.bridge.completion()
    }

    /// Enqueues one upscale: waits for `producer >= value`, runs, signals
    /// `completion = value`. Never blocks the CPU.
    pub fn run(&mut self, value: u64) -> Result<(), VfxError> {
        let _g = self.bridge.push()?;
        self.bridge.wait(value)?;
        let result = self.effect.run();
        let signaled = self.bridge.signal(value);
        result?;
        signaled
    }

    /// Blocks until all CUDA work of the session is idle (benchmarks/tests).
    pub fn synchronize(&self) {
        self.bridge.synchronize_context();
    }
}

impl Drop for VsrSession {
    fn drop(&mut self) {
        self.bridge.synchronize();
    }
}
