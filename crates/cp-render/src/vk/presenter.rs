//! Swapchain and the output contract.
//!
//! Preferred contract: FP16 scRGB (`R16G16B16A16_SFLOAT` +
//! `EXTENDED_SRGB_LINEAR`), the same as the D3D12 backend; the compositor
//! (KWin through `wp_color_manager_v1`'s Windows-scRGB description) maps it
//! to SDR or HDR outputs. Without that color space (no color management, X11)
//! the swapchain is 8-bit sRGB and the output pass maps to SDR white.
//!
//! The present mode follows `vsync`: MAILBOX (the compositor takes the
//! newest image each refresh: no queueing latency) or IMMEDIATE (tearing),
//! falling back to FIFO. Changing it recreates the swapchain.

use ash::vk;

use super::device::Gpu;
use super::{Result, VkError};
use crate::common::PresentResult;

/// `VK_PRESENT_MODE_FIFO_LATEST_READY_EXT` (newer than the ash headers).
const FIFO_LATEST_READY: vk::PresentModeKHR = vk::PresentModeKHR::from_raw(1_000_361_000);

/// What the swapchain images hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputEncoding {
    /// Linear scRGB, 1.0 = 80 nits (compositor color-managed).
    Scrgb,
    /// sRGB-encoded SDR (the format encodes; shaders write linear 0..1).
    Sdr,
}

pub struct Presenter {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    /// Signaled when rendering into image `i` finished (waited by present).
    render_done: Vec<vk::Semaphore>,
    pub format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    pub encoding: OutputEncoding,
    width: u32,
    height: u32,
    vsync: bool,
    mode: vk::PresentModeKHR,
    tearing_supported: bool,
    /// Swapchain must be recreated before the next acquire.
    dirty: bool,
    current: Option<u32>,
    present_id: u64,
}

impl std::fmt::Debug for Presenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Presenter")
            .field("size", &(self.width, self.height))
            .field("format", &self.format)
            .field("mode", &self.mode)
            .finish()
    }
}

fn choose_format(
    formats: &[vk::SurfaceFormatKHR],
    scrgb_allowed: bool,
) -> Option<(vk::SurfaceFormatKHR, OutputEncoding)> {
    let find = |f: vk::Format, cs: vk::ColorSpaceKHR| {
        formats
            .iter()
            .copied()
            .find(|s| s.format == f && s.color_space == cs)
    };
    if scrgb_allowed
        && let Some(f) = find(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        )
    {
        return Some((f, OutputEncoding::Scrgb));
    }
    [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB]
        .into_iter()
        .find_map(|f| find(f, vk::ColorSpaceKHR::SRGB_NONLINEAR))
        .map(|f| (f, OutputEncoding::Sdr))
}

impl Presenter {
    pub fn new(
        gpu: &Gpu,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Self> {
        // SAFETY: queries on a valid physical device and surface.
        let (formats, modes) = unsafe {
            (
                gpu.surface_fn
                    .get_physical_device_surface_formats(gpu.physical, surface)?,
                gpu.surface_fn
                    .get_physical_device_surface_present_modes(gpu.physical, surface)?,
            )
        };
        let (format, encoding) = choose_format(&formats, gpu.features.swapchain_colorspace)
            .ok_or(VkError::NoSurfaceFormat)?;
        let mut p = Self {
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            render_done: Vec::new(),
            format: format.format,
            color_space: format.color_space,
            encoding,
            width: width.max(1),
            height: height.max(1),
            vsync,
            mode: vk::PresentModeKHR::FIFO,
            tearing_supported: modes.contains(&vk::PresentModeKHR::IMMEDIATE),
            dirty: true,
            current: None,
            present_id: 0,
        };
        p.recreate(gpu)?;
        tracing::info!(
            width,
            height,
            format = ?p.format,
            color_space = ?p.color_space,
            mode = ?p.mode,
            "swapchain created"
        );
        Ok(p)
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn tearing_supported(&self) -> bool {
        self.tearing_supported
    }

    fn present_mode(&self, gpu: &Gpu) -> Result<vk::PresentModeKHR> {
        // SAFETY: query on a valid surface.
        let modes = unsafe {
            gpu.surface_fn
                .get_physical_device_surface_present_modes(gpu.physical, self.surface)?
        };
        let order: &[vk::PresentModeKHR] = if self.vsync {
            &[vk::PresentModeKHR::MAILBOX, FIFO_LATEST_READY]
        } else {
            &[vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX]
        };
        Ok(order
            .iter()
            .copied()
            .find(|m| modes.contains(m))
            .unwrap_or(vk::PresentModeKHR::FIFO))
    }

    fn destroy_views(&mut self, gpu: &Gpu) {
        // SAFETY: the caller waited for the GPU; views/semaphores unused.
        unsafe {
            for v in self.views.drain(..) {
                gpu.device.destroy_image_view(v, None);
            }
            for s in self.render_done.drain(..) {
                gpu.device.destroy_semaphore(s, None);
            }
        }
        self.images.clear();
    }

    /// (Re)creates the swapchain for the current size and present mode.
    fn recreate(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.wait_idle()?;
        // SAFETY: surface queries and swapchain creation with the previous
        // swapchain as `old_swapchain`; the GPU is idle, so the old images
        // and views are unused when destroyed.
        unsafe {
            let caps = gpu
                .surface_fn
                .get_physical_device_surface_capabilities(gpu.physical, self.surface)?;
            let extent = if caps.current_extent.width == u32::MAX {
                vk::Extent2D {
                    width: self.width.clamp(
                        caps.min_image_extent.width,
                        caps.max_image_extent.width.max(1),
                    ),
                    height: self.height.clamp(
                        caps.min_image_extent.height,
                        caps.max_image_extent.height.max(1),
                    ),
                }
            } else {
                caps.current_extent
            };
            if extent.width == 0 || extent.height == 0 {
                // Minimized: keep the old swapchain, retry later.
                return Ok(());
            }
            let mode = self.present_mode(gpu)?;
            let mut count = caps.min_image_count.max(3);
            if caps.max_image_count > 0 {
                count = count.min(caps.max_image_count);
            }
            let alpha = [
                vk::CompositeAlphaFlagsKHR::OPAQUE,
                vk::CompositeAlphaFlagsKHR::INHERIT,
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            ]
            .into_iter()
            .find(|a| caps.supported_composite_alpha.contains(*a))
            .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);
            let old = self.swapchain;
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(self.format)
                .image_color_space(self.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(alpha)
                .present_mode(mode)
                .clipped(true)
                .old_swapchain(old);
            let swapchain = gpu.swapchain_fn.create_swapchain(&info, None)?;
            self.destroy_views(gpu);
            if old != vk::SwapchainKHR::null() {
                gpu.swapchain_fn.destroy_swapchain(old, None);
            }
            self.swapchain = swapchain;
            self.images = gpu.swapchain_fn.get_swapchain_images(swapchain)?;
            for &image in &self.images {
                let view = gpu.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format)
                        .subresource_range(super::texture::COLOR_RANGE),
                    None,
                )?;
                self.views.push(view);
                self.render_done.push(
                    gpu.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
            }
            self.width = extent.width;
            self.height = extent.height;
            self.mode = mode;
            self.dirty = false;
            self.current = None;
            tracing::debug!(
                width = extent.width,
                height = extent.height,
                ?mode,
                images = self.images.len(),
                "swapchain (re)created"
            );
        }
        Ok(())
    }

    /// Requests new back buffers of `width × height` (applied before the
    /// next acquire).
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) != (self.width, self.height) {
            self.width = width;
            self.height = height;
            self.dirty = true;
        }
    }

    /// Acquires the next image, signaling `acquired` when it is ready.
    /// `None` when no image is available (minimized, timeout).
    pub fn acquire(
        &mut self,
        gpu: &Gpu,
        acquired: vk::Semaphore,
        timeout_ms: u32,
    ) -> Result<Option<u32>> {
        for _ in 0..2 {
            if self.dirty {
                self.recreate(gpu)?;
                if self.dirty {
                    return Ok(None);
                }
            }
            // SAFETY: valid swapchain; `acquired` is unsignaled (the previous
            // wait on it was submitted before this frame context reused it).
            let r = unsafe {
                gpu.swapchain_fn.acquire_next_image(
                    self.swapchain,
                    u64::from(timeout_ms) * 1_000_000,
                    acquired,
                    vk::Fence::null(),
                )
            };
            match r {
                Ok((index, suboptimal)) => {
                    self.dirty |= suboptimal;
                    self.current = Some(index);
                    return Ok(Some(index));
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.dirty = true,
                Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => return Ok(None),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(None)
    }

    /// Image + view + render-done semaphore of the acquired image.
    pub fn back_buffer(&self) -> Option<(vk::Image, vk::ImageView, vk::Semaphore)> {
        let i = self.current? as usize;
        Some((self.images[i], self.views[i], self.render_done[i]))
    }

    /// Presents the acquired image after its render-done semaphore. A
    /// changed `vsync` takes effect from the next acquire.
    pub fn present(&mut self, gpu: &Gpu, vsync: bool) -> PresentResult {
        if vsync != self.vsync {
            self.vsync = vsync;
            self.dirty = true;
        }
        let Some(index) = self.current.take() else {
            return PresentResult::Occluded;
        };
        self.present_id += 1;
        let swapchains = [self.swapchain];
        let indices = [index];
        let waits = [self.render_done[index as usize]];
        let ids = [self.present_id];
        let mut present_id = vk::PresentIdKHR::default().present_ids(&ids);
        let mut info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        if gpu.features.present_wait {
            info = info.push_next(&mut present_id);
        }
        // SAFETY: the image was acquired and its rendering submitted with a
        // signal of `render_done[index]`.
        match unsafe { gpu.swapchain_fn.queue_present(gpu.queue, &info) } {
            Ok(suboptimal) => {
                self.dirty |= suboptimal;
                PresentResult::Ok
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty = true;
                PresentResult::Ok
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => PresentResult::DeviceLost,
            Err(e) => {
                tracing::warn!("present failed: {e}");
                self.dirty = true;
                PresentResult::Ok
            }
        }
    }

    /// Id of the most recent present (`VK_KHR_present_id`).
    pub fn last_present_id(&self) -> u64 {
        self.present_id
    }

    pub fn destroy(&mut self, gpu: &Gpu) {
        let _ = gpu.wait_idle();
        self.destroy_views(gpu);
        // SAFETY: the GPU is idle; the swapchain is destroyed before its
        // surface.
        unsafe {
            if self.swapchain != vk::SwapchainKHR::null() {
                gpu.swapchain_fn.destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
            if self.surface != vk::SurfaceKHR::null() {
                gpu.surface_fn.destroy_surface(self.surface, None);
                self.surface = vk::SurfaceKHR::null();
            }
        }
    }
}
