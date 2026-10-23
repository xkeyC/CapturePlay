//! Upload slot → plane textures → linear scRGB working frame.

use std::rc::Rc;

use cp_core::color::{TransferFunction, yuv_to_rgb_matrix};
use cp_core::{ColorDesc, PixelFormat, Primaries, VideoMode};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Result;

use crate::descriptors::{GpuDescriptorRing, StagingHeap};
use crate::gpu::{Gpu, borrow_com};
use crate::shaders::{self, ComputePipeline, dispatch_2d, set_compute_constants};
use crate::texture::{Texture, TextureDesc, flush_barriers};
use crate::upload::SlotLayout;

/// Root constants of `convert.hlsl` (24 dwords).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Constants {
    rows: [[f32; 4]; 3],
    transfer: u32,
    primaries_2020: u32,
    sdr_white: f32,
    hlg_peak: f32,
    hlg_gamma: f32,
    width: u32,
    height: u32,
    pad0: u32,
    chroma_texel: [f32; 2],
    pad1: [f32; 2],
}

/// Display-dependent parameters of the decode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeParams {
    /// Nits at which SDR white is placed.
    pub sdr_white_nits: f32,
    /// Display peak used for the HLG OOTF.
    pub hlg_peak_nits: f32,
}

struct PlaneTextures {
    mode: VideoMode,
    planes: Vec<Texture>,
}

pub struct ConvertPass {
    planar: ComputePipeline,
    yuyv: ComputePipeline,
    uyvy: ComputePipeline,
    bgra: ComputePipeline,
    inputs: Option<PlaneTextures>,
}

impl std::fmt::Debug for ConvertPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvertPass").finish_non_exhaustive()
    }
}

/// Texture formats/extents of each plane as stored in the upload slot.
fn plane_formats(format: PixelFormat, w: u32, h: u32) -> Vec<(DXGI_FORMAT, u32, u32)> {
    match format {
        PixelFormat::Nv12 => vec![
            (DXGI_FORMAT_R8_UNORM, w, h),
            (DXGI_FORMAT_R8G8_UNORM, w / 2, h.div_ceil(2)),
        ],
        PixelFormat::P010 => vec![
            (DXGI_FORMAT_R16_UNORM, w, h),
            (DXGI_FORMAT_R16G16_UNORM, w / 2, h.div_ceil(2)),
        ],
        PixelFormat::Yuy2 | PixelFormat::Uyvy => vec![(DXGI_FORMAT_R8G8B8A8_UNORM, w / 2, h)],
        PixelFormat::Bgra8 => vec![(DXGI_FORMAT_B8G8R8A8_UNORM, w, h)],
    }
}

impl ConvertPass {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            planar: ComputePipeline::new(&gpu.device, shaders::CONVERT_PLANAR)?,
            yuyv: ComputePipeline::new(&gpu.device, shaders::CONVERT_YUYV)?,
            uyvy: ComputePipeline::new(&gpu.device, shaders::CONVERT_UYVY)?,
            bgra: ComputePipeline::new(&gpu.device, shaders::CONVERT_BGRA)?,
            inputs: None,
        })
    }

    fn ensure_inputs(&mut self, gpu: &Gpu, heap: &Rc<StagingHeap>, mode: &VideoMode) -> Result<()> {
        if self.inputs.as_ref().is_some_and(|i| i.mode == *mode) {
            return Ok(());
        }
        let planes = plane_formats(mode.format, mode.width, mode.height)
            .into_iter()
            .map(|(fmt, w, h)| Texture::new(gpu, heap, TextureDesc::new(w, h, fmt)))
            .collect::<Result<Vec<_>>>()?;
        self.inputs = Some(PlaneTextures {
            mode: *mode,
            planes,
        });
        Ok(())
    }

    /// Records upload-slot copy + conversion into `out` (RGBA16F, UAV).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        gpu: &Gpu,
        heap: &Rc<StagingHeap>,
        ring: &mut GpuDescriptorRing,
        list: &ID3D12GraphicsCommandList,
        upload: &ID3D12Resource,
        layout: &SlotLayout,
        mode: &VideoMode,
        color: &ColorDesc,
        params: DecodeParams,
        out: &Texture,
    ) -> Result<()> {
        self.ensure_inputs(gpu, heap, mode)?;
        let inputs = self
            .inputs
            .as_ref()
            .ok_or_else(windows::core::Error::empty)?;
        let mut barriers = Vec::new();

        for p in &inputs.planes {
            p.transition(&mut barriers, D3D12_RESOURCE_STATE_COPY_DEST);
        }
        flush_barriers(list, &mut barriers);
        for (i, p) in inputs.planes.iter().enumerate() {
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: borrow_com(&p.resource),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: borrow_com(upload),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: layout.planes[i].offset as u64,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: p.format,
                            Width: p.width,
                            Height: p.height,
                            Depth: 1,
                            RowPitch: layout.planes[i].pitch as u32,
                        },
                    },
                },
            };
            // SAFETY: both locations reference live resources; the footprint
            // matches the slot layout written by the capture thread.
            unsafe { list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        }
        for p in &inputs.planes {
            p.transition(
                &mut barriers,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
        }
        out.transition(&mut barriers, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush_barriers(list, &mut barriers);

        let pipeline = match mode.format {
            PixelFormat::Nv12 | PixelFormat::P010 => &self.planar,
            PixelFormat::Yuy2 => &self.yuyv,
            PixelFormat::Uyvy => &self.uyvy,
            PixelFormat::Bgra8 => &self.bgra,
        };
        pipeline.bind(list);
        let mut srvs: Vec<_> = inputs.planes.iter().map(|p| p.srv.handle).collect();
        if srvs.len() < 2 {
            // Table has two slots; repeat the first view for the unused one.
            srvs.push(srvs[0]);
        }
        let srv_table = ring.table(&gpu.device, &srvs);
        let uav_table = ring.table(&gpu.device, &[out.uav_handle()]);
        let chroma = inputs.planes.get(1).unwrap_or(&inputs.planes[0]);
        let c = Constants {
            rows: yuv_to_rgb_matrix(
                color.matrix,
                color.range,
                mode.format.bit_depth(),
                mode.format.storage_bits(),
            ),
            transfer: match color.transfer {
                TransferFunction::Srgb => 0,
                TransferFunction::Gamma22 => 1,
                TransferFunction::Bt1886 => 2,
                TransferFunction::Pq => 3,
                TransferFunction::Hlg => 4,
            },
            primaries_2020: u32::from(color.primaries == Primaries::Bt2020),
            sdr_white: params.sdr_white_nits / cp_core::color::SCRGB_UNIT_NITS,
            hlg_peak: params.hlg_peak_nits,
            hlg_gamma: 1.2 + 0.42 * (params.hlg_peak_nits / 1000.0).log10(),
            width: mode.width,
            height: mode.height,
            pad0: 0,
            chroma_texel: [1.0 / chroma.width as f32, 1.0 / chroma.height as f32],
            pad1: [0.0; 2],
        };
        set_compute_constants(list, &c);
        // SAFETY: tables live in the bound shader-visible heap.
        unsafe {
            list.SetComputeRootDescriptorTable(1, srv_table);
            list.SetComputeRootDescriptorTable(2, uav_table);
        }
        dispatch_2d(list, mode.width, mode.height);
        Ok(())
    }
}
