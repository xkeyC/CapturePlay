//! One frame's command recording: a command pool whose buffers are
//! submitted in order, the frame arena, textures retired while recording,
//! and semaphore waits for the next submission.
//!
//! A frame normally records into one buffer. Backends that hand work to
//! another engine (CUDA for VFG) submit what they recorded so far with a
//! signal ([`Recording::submit_and_continue`]), let the engine work, and
//! make the rest of the frame wait for its result ([`Recording::wait_next`])
//! — the Vulkan form of the D3D12 backend's `submit_and_continue` and
//! `queue_wait`.

use ash::vk;

use super::Result;
use super::arena::Arena;
use super::device::Gpu;
use super::texture::Texture;

pub struct Recording {
    device: ash::Device,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    /// Index of the open buffer in `cmds`.
    open: usize,
    pub arena: Arena,
    /// Textures replaced while recording; destroyed once the recording's
    /// work completed (later than any earlier use: the timeline is
    /// monotonic).
    pub retire: Vec<Texture>,
    /// Timeline waits for the next submission.
    waits: Vec<(vk::Semaphore, u64)>,
    /// Device-timeline value of the latest submission.
    pub value: u64,
}

impl std::fmt::Debug for Recording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recording")
            .field("buffers", &self.cmds.len())
            .field("value", &self.value)
            .finish_non_exhaustive()
    }
}

impl Recording {
    pub fn new(gpu: &Gpu, arena_bytes: u64) -> Result<Self> {
        // SAFETY: plain object creation on a valid device.
        let pool = unsafe {
            gpu.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(gpu.queue_family),
                None,
            )?
        };
        let arena = match Arena::new(&gpu.device, &gpu.memory_properties, arena_bytes) {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: created above, unused.
                unsafe { gpu.device.destroy_command_pool(pool, None) };
                return Err(e);
            }
        };
        Ok(Self {
            device: gpu.device.clone(),
            pool,
            cmds: Vec::new(),
            open: 0,
            arena,
            retire: Vec::new(),
            waits: Vec::new(),
            value: 0,
        })
    }

    /// Waits until the previous use completed, recycles everything and
    /// opens the first buffer.
    pub fn begin(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.wait(self.value)?;
        self.retire.clear();
        self.arena.reset();
        self.waits.clear();
        // SAFETY: the GPU finished this pool's work (timeline wait above).
        unsafe {
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
        }
        self.open = 0;
        self.open_buffer()
    }

    fn open_buffer(&mut self) -> Result<()> {
        if self.open == self.cmds.len() {
            // SAFETY: allocating from our own pool.
            let cmd = unsafe {
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            self.cmds.push(cmd);
        }
        // SAFETY: the buffer was reset with the pool and is not pending.
        unsafe {
            self.device.begin_command_buffer(
                self.cmds[self.open],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(())
    }

    /// The open command buffer.
    pub fn cmd(&self) -> vk::CommandBuffer {
        self.cmds[self.open]
    }

    /// Ends and submits the open buffer, waiting on the pending waits plus
    /// `waits` and signaling `signal`. Returns its device-timeline value.
    pub fn submit(
        &mut self,
        gpu: &Gpu,
        waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        signal: &[(vk::Semaphore, u64)],
    ) -> Result<u64> {
        let cmd = self.cmd();
        // SAFETY: the buffer is open and fully recorded.
        unsafe { self.device.end_command_buffer(cmd)? };
        let all: Vec<_> = self
            .waits
            .drain(..)
            .map(|(s, v)| (s, v, vk::PipelineStageFlags2::ALL_COMMANDS))
            .chain(waits.iter().copied())
            .collect();
        self.value = gpu.submit(cmd, &all, signal)?;
        Ok(self.value)
    }

    /// Submits what was recorded (signaling `signal`) and opens a new
    /// buffer for the rest of the frame.
    pub fn submit_and_continue(
        &mut self,
        gpu: &Gpu,
        signal: &[(vk::Semaphore, u64)],
    ) -> Result<u64> {
        let v = self.submit(gpu, &[], signal)?;
        self.open += 1;
        self.open_buffer()?;
        Ok(v)
    }

    /// The next submission waits for `semaphore >= value` (timeline).
    pub fn wait_next(&mut self, semaphore: vk::Semaphore, value: u64) {
        self.waits.push((semaphore, value));
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        // SAFETY: owners wait for the GPU before dropping recordings.
        unsafe { self.device.destroy_command_pool(self.pool, None) };
    }
}
