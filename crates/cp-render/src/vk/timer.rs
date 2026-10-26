//! GPU timestamp pairs per frame-in-flight, read back without stalling.

use ash::vk;

use super::Result;
use super::device::Gpu;

pub struct GpuTimer {
    device: ash::Device,
    pool: vk::QueryPool,
    ns_per_tick: f64,
    pending: Vec<bool>,
}

impl std::fmt::Debug for GpuTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuTimer")
    }
}

impl GpuTimer {
    pub fn new(gpu: &Gpu, slots: usize) -> Result<Self> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count((slots * 2) as u32);
        // SAFETY: plain query pool creation.
        let pool = unsafe { gpu.device.create_query_pool(&info, None)? };
        Ok(Self {
            device: gpu.device.clone(),
            pool,
            ns_per_tick: f64::from(gpu.timestamp_period),
            pending: vec![false; slots],
        })
    }

    pub fn begin(&self, cmd: vk::CommandBuffer, slot: usize) {
        let first = (slot * 2) as u32;
        // SAFETY: indices within the pool; recording into an open buffer.
        unsafe {
            self.device.cmd_reset_query_pool(cmd, self.pool, first, 2);
            self.device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                first,
            );
        }
    }

    pub fn end(&mut self, cmd: vk::CommandBuffer, slot: usize) {
        // SAFETY: as above.
        unsafe {
            self.device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                (slot * 2 + 1) as u32,
            );
        }
        self.pending[slot] = true;
    }

    /// Elapsed ms of `slot`'s span. Call only after that frame's work
    /// completed.
    pub fn read(&mut self, slot: usize) -> Option<f64> {
        if !std::mem::take(&mut self.pending[slot]) {
            return None;
        }
        let mut ticks = [0u64; 2];
        // SAFETY: the queries were written by completed work; the result
        // array matches the query count.
        unsafe {
            self.device
                .get_query_pool_results(
                    self.pool,
                    (slot * 2) as u32,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
                .ok()?;
        }
        let [a, b] = ticks;
        (b > a).then(|| (b - a) as f64 * self.ns_per_tick / 1e6)
    }
}

impl Drop for GpuTimer {
    fn drop(&mut self) {
        // SAFETY: the renderer waits for the GPU before dropping timers.
        unsafe { self.device.destroy_query_pool(self.pool, None) };
    }
}
