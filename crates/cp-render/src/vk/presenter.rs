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
//!
//! # Present timing
//! With `VK_EXT_present_timing` every present queries when its first pixel
//! went out to the display: the equivalent of DXGI frame statistics, fed to
//! [`crate::present_timing`]. Times are on `CLOCK_MONOTONIC` (the host
//! clock) when the swapchain offers it; otherwise (NVIDIA on Wayland) on the
//! presentation engine's own clock, calibrated against the host clock every
//! second with `VK_KHR_calibrated_timestamps`.
//! `CAPTUREPLAY_VK_TIMED_PRESENT=1` additionally hands each frame over right
//! after rendering with its absolute target time (FIFO, so queued frames are
//! not replaced) instead of presenting at the target from the render thread.

use std::ffi::c_void;

use ash::vk;

use super::device::Gpu;
use super::present_timing_ext::{
    self as pt, PastPresentationTiming, PastPresentationTimingInfo,
    PastPresentationTimingProperties, PresentId2, PresentStageTime, PresentTimingInfo,
    PresentTimingSurfaceCapabilities, PresentTimingsInfo, SurfaceCapabilitiesPresentId2,
    SwapchainCalibratedTimestampInfo, SwapchainTimeDomainProperties, SwapchainTimingProperties,
};
use super::{Result, VkError};
use crate::common::PresentResult;

/// Timing results the presentation engine may hold for us.
const TIMING_QUEUE: u32 = 64;
/// Re-calibration interval of a presentation-engine clock.
const CALIBRATE_EVERY: std::time::Duration = std::time::Duration::from_secs(1);

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
    /// The surface supports present timing; `Some(absolute)` tells whether
    /// absolute-time presents are possible as well.
    timing_support: Option<bool>,
    /// `CAPTUREPLAY_VK_TIMED_PRESENT=1`.
    timed_requested: bool,
    timing: Option<Timing>,
    /// Host time of recent present calls, by present id.
    sent: std::collections::VecDeque<(u64, i64)>,
    /// Present call → first pixel out of the latest timed result (ns).
    pub present_to_display_ns: Option<i64>,
    /// Light levels last handed to the compositor (re-applied to new
    /// swapchains).
    hdr_metadata: Option<HdrLevels>,
}

/// Content light levels of the output in the swapchain's encoding (what
/// the compositor should assume the frames span).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrLevels {
    pub max_nits: f32,
    pub max_frame_average_nits: f32,
    pub min_nits: f32,
}

/// Present timing state of the current swapchain.
#[derive(Debug)]
struct Timing {
    domain_id: u64,
    /// `None`: the domain is `CLOCK_MONOTONIC`. Otherwise the calibration
    /// of a presentation-engine domain: `host = engine + offset`.
    calibration: Option<Calibration>,
    /// Frames are presented at absolute target times.
    absolute: bool,
    properties_counter: u64,
    /// `(refreshDuration, refreshInterval)`, see [`Presenter::refresh_info`].
    refresh: Option<(u64, u64)>,
}

#[derive(Debug, Clone, Copy)]
struct Calibration {
    domain: i32,
    offset_ns: i64,
    at: std::time::Instant,
}

impl Timing {
    /// Engine time → host time.
    fn to_host(&self, t: u64) -> i64 {
        t as i64 + self.calibration.map_or(0, |c| c.offset_ns)
    }

    /// Host time → engine time.
    fn to_engine(&self, ns: i64) -> u64 {
        (ns - self.calibration.map_or(0, |c| c.offset_ns)).max(1) as u64
    }
}

/// `host - engine` for a presentation-engine domain of `swapchain`.
fn calibrate(gpu: &Gpu, swapchain: vk::SwapchainKHR, domain: i32, domain_id: u64) -> Option<i64> {
    let f = gpu.calibrated_fn.as_ref()?;
    let swapchain_info = SwapchainCalibratedTimestampInfo {
        s_type: pt::ST_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO,
        p_next: std::ptr::null(),
        swapchain,
        present_stage: pt::STAGE_IMAGE_FIRST_PIXEL_OUT,
        time_domain_id: domain_id,
    };
    let engine = vk::CalibratedTimestampInfoKHR {
        p_next: std::ptr::from_ref(&swapchain_info).cast(),
        time_domain: vk::TimeDomainKHR::from_raw(domain),
        ..Default::default()
    };
    let host =
        vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::CLOCK_MONOTONIC);
    // SAFETY: valid swapchain; the chained struct outlives the call.
    let (ts, deviation) = unsafe { f.get_calibrated_timestamps(&[engine, host]) }
        .inspect_err(|e| tracing::warn!("present clock calibration failed: {e}"))
        .ok()?;
    tracing::trace!(deviation, "present clock calibrated");
    Some(ts[1] as i64 - ts[0] as i64)
}

/// Present-timing support of `surface`: `Some(absolute_time_supported)`
/// when per-present `IMAGE_FIRST_PIXEL_OUT` queries with present ids work.
fn surface_timing(gpu: &Gpu, surface: vk::SurfaceKHR) -> Option<bool> {
    let caps2 = gpu.surface_caps2_fn.as_ref()?;
    gpu.present_timing_fn?;
    let mut id2 = SurfaceCapabilitiesPresentId2 {
        s_type: pt::ST_SURFACE_CAPABILITIES_PRESENT_ID_2,
        p_next: std::ptr::null_mut(),
        present_id2_supported: 0,
    };
    let mut timing = PresentTimingSurfaceCapabilities {
        s_type: pt::ST_PRESENT_TIMING_SURFACE_CAPABILITIES,
        p_next: std::ptr::from_mut(&mut id2).cast(),
        present_timing_supported: 0,
        present_at_absolute_time_supported: 0,
        present_at_relative_time_supported: 0,
        present_stage_queries: 0,
    };
    let info = vk::PhysicalDeviceSurfaceInfo2KHR::default().surface(surface);
    let mut caps = vk::SurfaceCapabilities2KHR {
        p_next: std::ptr::from_mut(&mut timing).cast(),
        ..Default::default()
    };
    // SAFETY: valid surface; the chained structs outlive the call.
    unsafe { caps2.get_physical_device_surface_capabilities2(gpu.physical, &info, &mut caps) }
        .ok()?;
    let usable = timing.present_timing_supported != 0
        && id2.present_id2_supported != 0
        && timing.present_stage_queries & pt::STAGE_IMAGE_FIRST_PIXEL_OUT != 0;
    usable.then_some(
        timing.present_at_absolute_time_supported != 0 && gpu.features.present_at_absolute_time,
    )
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
            timing_support: surface_timing(gpu, surface),
            timed_requested: std::env::var("CAPTUREPLAY_VK_TIMED_PRESENT").as_deref() == Ok("1"),
            timing: None,
            sent: std::collections::VecDeque::new(),
            present_to_display_ns: None,
            hdr_metadata: None,
        };
        p.recreate(gpu)?;
        tracing::info!(
            width,
            height,
            format = ?p.format,
            color_space = ?p.color_space,
            mode = ?p.mode,
            timing = ?p.timing,
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

    /// Absolute-time presents requested and possible (vsync only).
    fn wants_timed(&self) -> bool {
        self.timed_requested && self.vsync && self.timing_support == Some(true)
    }

    fn present_mode(&self, gpu: &Gpu) -> Result<vk::PresentModeKHR> {
        // SAFETY: query on a valid surface.
        let modes = unsafe {
            gpu.surface_fn
                .get_physical_device_surface_present_modes(gpu.physical, self.surface)?
        };
        let order: &[vk::PresentModeKHR] = if self.wants_timed() {
            &[vk::PresentModeKHR::FIFO]
        } else if self.vsync {
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
            let flags = if self.timing_support.is_some() {
                pt::SWAPCHAIN_CREATE_PRESENT_TIMING | pt::SWAPCHAIN_CREATE_PRESENT_ID_2
            } else {
                vk::SwapchainCreateFlagsKHR::empty()
            };
            let info = vk::SwapchainCreateInfoKHR::default()
                .flags(flags)
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
            if let Some(levels) = self.hdr_metadata {
                self.apply_hdr_metadata(gpu, levels);
            }
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
            self.timing = if self.timing_support.is_some() {
                self.setup_timing(gpu)
            } else {
                None
            };
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

    /// Results queue + the `CLOCK_MONOTONIC` time domain of a new swapchain.
    fn setup_timing(&self, gpu: &Gpu) -> Option<Timing> {
        let f = gpu.present_timing_fn?;
        let dev = gpu.device.handle();
        // SAFETY: valid swapchain created with PRESENT_TIMING; the arrays
        // passed to the second query hold `count` elements.
        unsafe {
            let r = (f.set_queue_size)(dev, self.swapchain, TIMING_QUEUE);
            if r != vk::Result::SUCCESS {
                tracing::warn!("present timing queue size failed: {r}");
                return None;
            }
            let mut props = SwapchainTimeDomainProperties {
                s_type: pt::ST_SWAPCHAIN_TIME_DOMAIN_PROPERTIES,
                p_next: std::ptr::null_mut(),
                time_domain_count: 0,
                p_time_domains: std::ptr::null_mut(),
                p_time_domain_ids: std::ptr::null_mut(),
            };
            let mut counter = 0u64;
            let _ = (f.get_time_domain_properties)(dev, self.swapchain, &mut props, &mut counter);
            let n = props.time_domain_count as usize;
            let mut domains = vec![0i32; n];
            let mut ids = vec![0u64; n];
            props.p_time_domains = domains.as_mut_ptr();
            props.p_time_domain_ids = ids.as_mut_ptr();
            let r = (f.get_time_domain_properties)(dev, self.swapchain, &mut props, &mut counter);
            if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                tracing::warn!("present timing domains failed: {r}");
                return None;
            }
            let domains = &domains[..props.time_domain_count as usize];
            let pick = |want: i32| domains.iter().position(|&d| d == want);
            let (i, calibration) = if let Some(i) = pick(pt::TIME_DOMAIN_CLOCK_MONOTONIC) {
                (i, None)
            } else {
                let Some(i) = pick(pt::TIME_DOMAIN_PRESENT_STAGE_LOCAL)
                    .or_else(|| pick(pt::TIME_DOMAIN_SWAPCHAIN_LOCAL))
                else {
                    tracing::warn!(?domains, "present timing offers no usable time domain");
                    return None;
                };
                let offset_ns = calibrate(gpu, self.swapchain, domains[i], ids[i])?;
                let c = Calibration {
                    domain: domains[i],
                    offset_ns,
                    at: std::time::Instant::now(),
                };
                (i, Some(c))
            };
            Some(Timing {
                domain_id: ids[i],
                calibration,
                absolute: self.wants_timed(),
                properties_counter: u64::MAX,
                refresh: None,
            })
        }
    }

    /// Frames are handed over early with their target time.
    pub fn timed(&self) -> bool {
        self.timing.as_ref().is_some_and(|t| t.absolute)
    }

    /// When presented frames reached the display: `(present id, host ns of
    /// the first pixel out)`, in present order. Call regularly: the results
    /// queue is bounded.
    pub fn poll_timing(&mut self, gpu: &Gpu) -> Vec<(u64, i64)> {
        let (Some(t), Some(f)) = (self.timing.as_mut(), gpu.present_timing_fn) else {
            return Vec::new();
        };
        if let Some(c) = t.calibration.filter(|c| c.at.elapsed() >= CALIBRATE_EVERY)
            && let Some(offset_ns) = calibrate(gpu, self.swapchain, c.domain, t.domain_id)
        {
            t.calibration = Some(Calibration {
                offset_ns,
                at: std::time::Instant::now(),
                ..c
            });
        }
        const N: usize = 32;
        let mut stages = [PresentStageTime::default(); N];
        let mut timings: Vec<PastPresentationTiming> = stages
            .iter_mut()
            .map(|s| PastPresentationTiming {
                s_type: pt::ST_PAST_PRESENTATION_TIMING,
                p_next: std::ptr::null_mut(),
                present_id: 0,
                target_time: 0,
                present_stage_count: 1,
                p_present_stages: std::ptr::from_mut(s),
                time_domain: 0,
                time_domain_id: 0,
                report_complete: 0,
            })
            .collect();
        let mut props = PastPresentationTimingProperties {
            s_type: pt::ST_PAST_PRESENTATION_TIMING_PROPERTIES,
            p_next: std::ptr::null_mut(),
            timing_properties_counter: 0,
            time_domains_counter: 0,
            presentation_timing_count: N as u32,
            p_presentation_timings: timings.as_mut_ptr(),
        };
        let info = PastPresentationTimingInfo {
            s_type: pt::ST_PAST_PRESENTATION_TIMING_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            swapchain: self.swapchain,
        };
        let dev = gpu.device.handle();
        // SAFETY: every result has room for the one stage each present
        // queries; all arrays outlive the call.
        let r = unsafe { (f.get_past_presentation_timing)(dev, &info, &mut props) };
        if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
            if r == vk::Result::ERROR_OUT_OF_DATE_KHR {
                self.dirty = true;
            }
            return Vec::new();
        }
        let out: Vec<(u64, i64)> = timings[..props.presentation_timing_count as usize]
            .iter()
            .filter(|p| p.report_complete != 0 && p.present_stage_count > 0)
            .filter_map(|p| {
                // SAFETY: the implementation wrote `present_stage_count`
                // (≤ 1) stages into our array.
                let s = unsafe { *p.p_present_stages };
                (s.stage == pt::STAGE_IMAGE_FIRST_PIXEL_OUT && s.time > 0)
                    .then_some((p.present_id, t.to_host(s.time)))
            })
            .collect();
        if let Some(&(id, shown)) = out.last() {
            while self.sent.front().is_some_and(|&(i, _)| i < id) {
                self.sent.pop_front();
            }
            if let Some(&(_, sent)) = self.sent.front().filter(|&&(i, _)| i == id) {
                self.present_to_display_ns = Some(shown - sent);
                tracing::trace!(
                    id,
                    results = out.len(),
                    present_to_display_ns = shown - sent,
                    "present timing"
                );
            }
        }
        if props.timing_properties_counter != t.properties_counter {
            let mut tp = SwapchainTimingProperties {
                s_type: pt::ST_SWAPCHAIN_TIMING_PROPERTIES,
                p_next: std::ptr::null_mut(),
                refresh_duration: 0,
                refresh_interval: 0,
            };
            let mut counter = 0u64;
            // SAFETY: valid swapchain and out pointers.
            let r =
                unsafe { (f.get_timing_properties)(dev, self.swapchain, &mut tp, &mut counter) };
            if r == vk::Result::SUCCESS {
                t.properties_counter = props.timing_properties_counter;
                let refresh = (tp.refresh_duration, tp.refresh_interval);
                if t.refresh != Some(refresh) {
                    tracing::info!(
                        duration_ns = tp.refresh_duration,
                        interval_ns = tp.refresh_interval,
                        "presentation engine refresh"
                    );
                }
                t.refresh = Some(refresh);
            }
        }
        out
    }

    /// Refresh period reported by the presentation engine and whether it
    /// runs variable refresh (`None`: unknown).
    pub fn refresh_info(&self) -> Option<(i64, Option<bool>)> {
        let (duration, interval) = self.timing.as_ref()?.refresh?;
        if duration == 0 {
            return None;
        }
        let vrr = match interval {
            0 => None,
            u64::MAX => Some(true),
            _ => Some(false),
        };
        Some((duration as i64, vrr))
    }

    /// Tells the compositor how bright the frames get (`VK_EXT_hdr_metadata`;
    /// on Wayland the driver puts it into the surface's image description).
    /// Without it KWin assumes scRGB content peaks at the description's
    /// nominal 80 and clips or compresses everything above.
    pub fn set_hdr_metadata(&mut self, gpu: &Gpu, levels: Option<HdrLevels>) {
        if levels == self.hdr_metadata {
            return;
        }
        self.hdr_metadata = levels;
        if let Some(l) = levels
            && self.swapchain != vk::SwapchainKHR::null()
        {
            self.apply_hdr_metadata(gpu, l);
        }
    }

    fn apply_hdr_metadata(&self, gpu: &Gpu, l: HdrLevels) {
        let Some(f) = gpu.hdr_metadata_fn.as_ref() else {
            return;
        };
        let xy = |x: f32, y: f32| vk::XYColorEXT { x, y };
        // The frames may use the whole BT.2020 gamut (negative scRGB).
        let metadata = vk::HdrMetadataEXT::default()
            .display_primary_red(xy(0.708, 0.292))
            .display_primary_green(xy(0.170, 0.797))
            .display_primary_blue(xy(0.131, 0.046))
            .white_point(xy(0.3127, 0.3290))
            .max_luminance(l.max_nits)
            .min_luminance(l.min_nits)
            .max_content_light_level(l.max_nits)
            .max_frame_average_light_level(l.max_frame_average_nits);
        // SAFETY: a live swapchain of this device.
        unsafe { f.set_hdr_metadata(&[self.swapchain], &[metadata]) };
        tracing::debug!(?l, "HDR metadata");
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

    /// Presents the acquired image after its render-done semaphore; with
    /// [`Self::timed`], to be shown at `target_ns` (host time). A changed
    /// `vsync` takes effect from the next acquire.
    pub fn present(&mut self, gpu: &Gpu, vsync: bool, target_ns: Option<i64>) -> PresentResult {
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
        let id2 = PresentId2 {
            s_type: pt::ST_PRESENT_ID_2,
            p_next: std::ptr::null(),
            swapchain_count: 1,
            p_present_ids: ids.as_ptr(),
        };
        let target = match (&self.timing, target_ns) {
            (Some(t), Some(ns)) if t.absolute && ns > 0 => t.to_engine(ns),
            _ => 0,
        };
        let timing_info = self.timing.as_ref().map(|t| PresentTimingInfo {
            s_type: pt::ST_PRESENT_TIMING_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            target_time: target,
            time_domain_id: t.domain_id,
            present_stage_queries: pt::STAGE_IMAGE_FIRST_PIXEL_OUT,
            target_time_domain_present_stage: if target != 0 {
                pt::STAGE_IMAGE_FIRST_PIXEL_OUT
            } else {
                0
            },
        });
        let timings = timing_info.as_ref().map(|ti| PresentTimingsInfo {
            s_type: pt::ST_PRESENT_TIMINGS_INFO,
            p_next: std::ptr::from_ref(&id2).cast(),
            swapchain_count: 1,
            p_timing_infos: std::ptr::from_ref(ti),
        });
        let mut info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        if let Some(t) = &timings {
            info.p_next = std::ptr::from_ref(t).cast::<c_void>();
        }
        if self.timing.is_some() {
            if self.sent.len() == 2 * TIMING_QUEUE as usize {
                self.sent.pop_front();
            }
            self.sent
                .push_back((self.present_id, cp_core::time::HostTime::now().0));
        }
        // SAFETY: the image was acquired and its rendering submitted with a
        // signal of `render_done[index]`; the chained structs outlive the
        // call.
        let mut r = unsafe { gpu.swapchain_fn.queue_present(gpu.queue, &info) };
        if r == Err(pt::ERROR_PRESENT_TIMING_QUEUE_FULL) {
            // Results not collected in time: present without a query.
            tracing::debug!("present timing queue full");
            info.p_next = std::ptr::from_ref(&id2).cast::<c_void>();
            // SAFETY: as above.
            r = unsafe { gpu.swapchain_fn.queue_present(gpu.queue, &info) };
        }
        match r {
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

    /// Id of the most recent present (`VK_KHR_present_id2`).
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
