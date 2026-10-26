//! `VK_EXT_present_timing` + `VK_KHR_present_id2` (newer than the ash
//! headers): the structures and entry points the presenter uses, written
//! from `vulkan_core.h` 1.4.363.

#![allow(dead_code)]

use std::ffi::c_void;

use ash::vk;

pub const EXT_PRESENT_TIMING: &std::ffi::CStr = c"VK_EXT_present_timing";
pub const KHR_PRESENT_ID_2: &std::ffi::CStr = c"VK_KHR_present_id2";
pub const KHR_CALIBRATED_TIMESTAMPS: &std::ffi::CStr = c"VK_KHR_calibrated_timestamps";

const fn st(v: i32) -> vk::StructureType {
    vk::StructureType::from_raw(v)
}

pub const ST_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES: vk::StructureType = st(1_000_208_000);
pub const ST_SWAPCHAIN_TIMING_PROPERTIES: vk::StructureType = st(1_000_208_001);
pub const ST_SWAPCHAIN_TIME_DOMAIN_PROPERTIES: vk::StructureType = st(1_000_208_002);
pub const ST_PRESENT_TIMINGS_INFO: vk::StructureType = st(1_000_208_003);
pub const ST_PRESENT_TIMING_INFO: vk::StructureType = st(1_000_208_004);
pub const ST_PAST_PRESENTATION_TIMING_INFO: vk::StructureType = st(1_000_208_005);
pub const ST_PAST_PRESENTATION_TIMING_PROPERTIES: vk::StructureType = st(1_000_208_006);
pub const ST_PAST_PRESENTATION_TIMING: vk::StructureType = st(1_000_208_007);
pub const ST_PRESENT_TIMING_SURFACE_CAPABILITIES: vk::StructureType = st(1_000_208_008);
pub const ST_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO: vk::StructureType = st(1_000_208_009);
pub const ST_SURFACE_CAPABILITIES_PRESENT_ID_2: vk::StructureType = st(1_000_479_000);
pub const ST_PRESENT_ID_2: vk::StructureType = st(1_000_479_001);
pub const ST_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES: vk::StructureType = st(1_000_479_002);

pub const SWAPCHAIN_CREATE_PRESENT_TIMING: vk::SwapchainCreateFlagsKHR =
    vk::SwapchainCreateFlagsKHR::from_raw(0x200);
pub const SWAPCHAIN_CREATE_PRESENT_ID_2: vk::SwapchainCreateFlagsKHR =
    vk::SwapchainCreateFlagsKHR::from_raw(0x40);

pub const ERROR_PRESENT_TIMING_QUEUE_FULL: vk::Result = vk::Result::from_raw(-1_000_208_000);

/// `VkPresentStageFlagBitsEXT`.
pub const STAGE_QUEUE_OPERATIONS_END: u32 = 0x1;
pub const STAGE_REQUEST_DEQUEUED: u32 = 0x2;
pub const STAGE_IMAGE_FIRST_PIXEL_OUT: u32 = 0x4;
pub const STAGE_IMAGE_FIRST_PIXEL_VISIBLE: u32 = 0x8;

/// `VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR`: the host clock (`HostTime`).
pub const TIME_DOMAIN_CLOCK_MONOTONIC: i32 = 1;
/// `VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT` / `..._SWAPCHAIN_LOCAL_EXT`:
/// presentation-engine clocks, calibrated against the host clock.
pub const TIME_DOMAIN_PRESENT_STAGE_LOCAL: i32 = 1_000_208_000;
pub const TIME_DOMAIN_SWAPCHAIN_LOCAL: i32 = 1_000_208_001;

#[repr(C)]
#[derive(Debug)]
pub struct PhysicalDevicePresentTimingFeatures {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub present_timing: vk::Bool32,
    pub present_at_absolute_time: vk::Bool32,
    pub present_at_relative_time: vk::Bool32,
}

#[repr(C)]
#[derive(Debug)]
pub struct PhysicalDevicePresentId2Features {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub present_id2: vk::Bool32,
}

#[repr(C)]
#[derive(Debug)]
pub struct PresentTimingSurfaceCapabilities {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub present_timing_supported: vk::Bool32,
    pub present_at_absolute_time_supported: vk::Bool32,
    pub present_at_relative_time_supported: vk::Bool32,
    pub present_stage_queries: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct SurfaceCapabilitiesPresentId2 {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub present_id2_supported: vk::Bool32,
}

#[repr(C)]
#[derive(Debug)]
pub struct SwapchainTimingProperties {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub refresh_duration: u64,
    pub refresh_interval: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct SwapchainTimeDomainProperties {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub time_domain_count: u32,
    pub p_time_domains: *mut i32,
    pub p_time_domain_ids: *mut u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct PastPresentationTimingInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub swapchain: vk::SwapchainKHR,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PresentStageTime {
    pub stage: u32,
    pub time: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct PastPresentationTiming {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub present_id: u64,
    pub target_time: u64,
    pub present_stage_count: u32,
    pub p_present_stages: *mut PresentStageTime,
    pub time_domain: i32,
    pub time_domain_id: u64,
    pub report_complete: vk::Bool32,
}

#[repr(C)]
#[derive(Debug)]
pub struct PastPresentationTimingProperties {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub timing_properties_counter: u64,
    pub time_domains_counter: u64,
    pub presentation_timing_count: u32,
    pub p_presentation_timings: *mut PastPresentationTiming,
}

#[repr(C)]
#[derive(Debug)]
pub struct PresentTimingInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub target_time: u64,
    pub time_domain_id: u64,
    pub present_stage_queries: u32,
    pub target_time_domain_present_stage: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct PresentTimingsInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub swapchain_count: u32,
    pub p_timing_infos: *const PresentTimingInfo,
}

#[repr(C)]
#[derive(Debug)]
pub struct SwapchainCalibratedTimestampInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub swapchain: vk::SwapchainKHR,
    pub present_stage: u32,
    pub time_domain_id: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct PresentId2 {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub swapchain_count: u32,
    pub p_present_ids: *const u64,
}

const _: () = {
    assert!(size_of::<PhysicalDevicePresentTimingFeatures>() == 32);
    assert!(size_of::<PresentTimingSurfaceCapabilities>() == 32);
    assert!(size_of::<SwapchainTimingProperties>() == 32);
    assert!(size_of::<SwapchainTimeDomainProperties>() == 40);
    assert!(size_of::<PastPresentationTimingInfo>() == 32);
    assert!(size_of::<PresentStageTime>() == 16);
    assert!(size_of::<PastPresentationTiming>() == 72);
    assert!(size_of::<PastPresentationTimingProperties>() == 48);
    assert!(size_of::<PresentTimingInfo>() == 48);
    assert!(size_of::<PresentTimingsInfo>() == 32);
    assert!(size_of::<PresentId2>() == 32);
    assert!(size_of::<SwapchainCalibratedTimestampInfo>() == 40);
};

type PfnSetQueueSize = unsafe extern "system" fn(vk::Device, vk::SwapchainKHR, u32) -> vk::Result;
type PfnGetTimingProperties = unsafe extern "system" fn(
    vk::Device,
    vk::SwapchainKHR,
    *mut SwapchainTimingProperties,
    *mut u64,
) -> vk::Result;
type PfnGetTimeDomainProperties = unsafe extern "system" fn(
    vk::Device,
    vk::SwapchainKHR,
    *mut SwapchainTimeDomainProperties,
    *mut u64,
) -> vk::Result;
type PfnGetPastPresentationTiming = unsafe extern "system" fn(
    vk::Device,
    *const PastPresentationTimingInfo,
    *mut PastPresentationTimingProperties,
) -> vk::Result;

/// Device entry points of `VK_EXT_present_timing`.
#[derive(Clone, Copy)]
pub struct PresentTimingFn {
    pub set_queue_size: PfnSetQueueSize,
    pub get_timing_properties: PfnGetTimingProperties,
    pub get_time_domain_properties: PfnGetTimeDomainProperties,
    pub get_past_presentation_timing: PfnGetPastPresentationTiming,
}

impl std::fmt::Debug for PresentTimingFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PresentTimingFn")
    }
}

impl PresentTimingFn {
    /// Loads the entry points; `None` if any is missing.
    ///
    /// # Safety
    /// `device` must have been created with `VK_EXT_present_timing`.
    pub unsafe fn load(instance: &ash::Instance, device: &ash::Device) -> Option<Self> {
        let load = |name: &std::ffi::CStr| {
            // SAFETY: valid device handle and NUL-terminated name.
            unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }
        };
        // SAFETY: each pointer was returned for the named command, whose
        // signature matches the declared type (vulkan_core.h).
        unsafe {
            Some(Self {
                set_queue_size: std::mem::transmute::<
                    vk::PFN_vkVoidFunction,
                    Option<PfnSetQueueSize>,
                >(load(
                    c"vkSetSwapchainPresentTimingQueueSizeEXT",
                ))?,
                get_timing_properties: std::mem::transmute::<
                    vk::PFN_vkVoidFunction,
                    Option<PfnGetTimingProperties>,
                >(load(
                    c"vkGetSwapchainTimingPropertiesEXT",
                ))?,
                get_time_domain_properties: std::mem::transmute::<
                    vk::PFN_vkVoidFunction,
                    Option<PfnGetTimeDomainProperties>,
                >(load(
                    c"vkGetSwapchainTimeDomainPropertiesEXT",
                ))?,
                get_past_presentation_timing: std::mem::transmute::<
                    vk::PFN_vkVoidFunction,
                    Option<PfnGetPastPresentationTiming>,
                >(load(
                    c"vkGetPastPresentationTimingEXT",
                ))?,
            })
        }
    }
}
