//! The native end of the probe (Linux): `import-test` starts the Windows
//! build under Proton as its child, waits for the `EXPORT` lines in its
//! log, takes the exported descriptors with `pidfd_getfd` (allowed for a
//! descendant under yama's default `ptrace_scope` 1), imports them into a
//! native Vulkan device and checks the pixels against the host's checksum.

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ash::vk;

type Result<T> = std::result::Result<T, String>;

fn vk_err(what: &'static str) -> impl Fn(vk::Result) -> String {
    move |e| format!("{what}: {e:?}")
}

pub fn main() -> i32 {
    match run() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("import-test: FAILED: {e}");
            1
        }
    }
}

const USAGE: &str = "usage: captureplay-nr-host import-test --log <host log file> -- <command starting the host probe with --hold>";

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("import-test") {
        return Err(USAGE.into());
    }
    let split = args.iter().position(|a| a == "--").ok_or(USAGE)?;
    let log = match &args[1..split] {
        [k, v] if k == "--log" => PathBuf::from(v),
        _ => return Err(USAGE.into()),
    };
    let command = &args[split + 1..];
    let (program, rest) = command.split_first().ok_or(USAGE)?;

    let _ = std::fs::remove_file(&log);
    // Identifies the host among the processes Proton starts (its Linux pid
    // is not known to the host itself: under Wine, /proc/self resolves in
    // wineserver, which opens files on the client's behalf).
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let token = format!("cp-nr-{:x}-{nanos:x}", std::process::id());
    let mut child = Command::new(program)
        .args(rest)
        .args(["--token", &token])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("{program}: {e}"))?;
    let result = wait_and_import(&log, &token);
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// `key=value` fields of the first log line containing `marker`.
fn fields(text: &str, marker: &str) -> Option<HashMap<String, String>> {
    let line = text.lines().find(|l| l.contains(marker))?;
    Some(
        line.split_whitespace()
            .filter_map(|w| w.split_once('='))
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect(),
    )
}

/// The Linux pid of the host process started with `token` (not Proton's
/// `steam.exe` wrapper, which carries the same arguments).
fn find_host(token: &str) -> Option<i32> {
    std::fs::read_dir("/proc").ok()?.flatten().find_map(|e| {
        let pid: i32 = e.file_name().to_str()?.parse().ok()?;
        let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
        let mut args = cmdline.split(|&b| b == 0).map(String::from_utf8_lossy);
        let exe = args.next()?;
        (exe.ends_with("captureplay-nr-host.exe") && args.any(|a| a == token)).then_some(pid)
    })
}

fn wait_and_import(log: &PathBuf, token: &str) -> Result<()> {
    let start = Instant::now();
    let mut watched: Vec<(String, String)> = Vec::new();
    let (memory, semaphore) = loop {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        // Follow what each announced fd points to while the host runs on.
        let pid = find_host(token).map(|p| p.to_string());
        for fd in text
            .lines()
            .filter_map(|l| l.split("-> fd ").nth(1))
            .map(str::trim)
        {
            if let Some(pid) = &pid {
                let target = std::fs::read_link(format!("/proc/{pid}/fd/{fd}"))
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|e| e.to_string());
                let key = format!("{pid}/{fd}");
                if watched.iter().all(|(k, t)| k != &key || t != &target) {
                    println!(
                        "import-test: t={:>6} ms host fd {fd} -> {target}",
                        start.elapsed().as_millis()
                    );
                    watched.push((key, target));
                }
            }
        }
        if let Some(line) = text.lines().find(|l| l.contains("FAILED")) {
            return Err(format!("host: {line}"));
        }
        if let (Some(m), Some(s)) = (
            fields(&text, "EXPORT memory"),
            fields(&text, "EXPORT timeline semaphore"),
        ) {
            print!("{text}");
            break (m, s);
        }
        if start.elapsed() > Duration::from_secs(240) {
            return Err(format!("no EXPORT lines in {} after 240 s", log.display()));
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let get = |f: &HashMap<String, String>, k: &str| -> Result<String> {
        f.get(k)
            .cloned()
            .ok_or_else(|| format!("EXPORT line lacks {k}"))
    };
    let num = |s: String| -> Result<u64> {
        let s = s.trim_start_matches("0x");
        u64::from_str_radix(s, if s.len() == 16 { 16 } else { 10 }).map_err(|e| e.to_string())
    };
    let pid = find_host(token).ok_or("host process not found in /proc")?;
    let mem_fd = adopt(pid, num(get(&memory, "fd")?)? as i32)?;
    let sem_fd = adopt(pid, num(get(&semaphore, "fd")?)? as i32)?;
    let size = num(get(&memory, "size")?)?;
    let checksum = u64::from_str_radix(get(&memory, "checksum")?.trim_start_matches("0x"), 16)
        .map_err(|e| e.to_string())?;
    let uuid = get(&memory, "uuid")?;
    let extent = dims(&std::fs::read_to_string(log).unwrap_or_default())
        .ok_or("EXPORT memory line lacks WxH")?;
    println!("import-test: adopted memory fd {mem_fd} and semaphore fd {sem_fd} from pid {pid}");
    for fd in [mem_fd, sem_fd] {
        let target = std::fs::read_link(format!("/proc/self/fd/{fd}"))
            .map(|p| p.display().to_string())
            .unwrap_or_else(|e| e.to_string());
        println!("import-test:   fd {fd} -> {target}");
    }
    import(mem_fd, sem_fd, size, extent, &uuid, checksum)
}

/// `WxH` after `format=...` on the memory line.
fn dims(text: &str) -> Option<(u32, u32)> {
    let line = text.lines().find(|l| l.contains("EXPORT memory"))?;
    let word = line
        .split_whitespace()
        .skip_while(|w| !w.starts_with("format="))
        .nth(1)?;
    let (a, b) = word.split_once('x')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Our own descriptor for `fd` of process `pid`.
fn adopt(pid: i32, fd: i32) -> Result<i32> {
    // SAFETY: plain syscalls; the pidfd is closed below.
    unsafe {
        let pidfd = libc::syscall(libc::SYS_pidfd_open, pid, 0) as i32;
        if pidfd < 0 {
            return Err(format!(
                "pidfd_open({pid}): {}",
                std::io::Error::last_os_error()
            ));
        }
        let ours = libc::syscall(libc::SYS_pidfd_getfd, pidfd, fd, 0) as i32;
        let err = std::io::Error::last_os_error();
        libc::close(pidfd);
        if ours < 0 {
            return Err(format!("pidfd_getfd({pid}, {fd}): {err}"));
        }
        Ok(ours)
    }
}

fn checksum(values: &[u16]) -> u64 {
    values.iter().fold(0u64, |h, &v| {
        h.rotate_left(5) ^ u64::from(v).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    })
}

fn import(
    mem_fd: i32,
    sem_fd: i32,
    size: u64,
    (w, h): (u32, u32),
    uuid: &str,
    expected: u64,
) -> Result<()> {
    // SAFETY: plain Vulkan setup and teardown on this thread; every object
    // is destroyed before the device.
    unsafe {
        let entry = ash::Entry::load().map_err(|e| e.to_string())?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .map_err(vk_err("vkCreateInstance"))?;
        let physical = instance
            .enumerate_physical_devices()
            .map_err(vk_err("vkEnumeratePhysicalDevices"))?
            .into_iter()
            .find(|&p| {
                let mut id = vk::PhysicalDeviceIDProperties::default();
                let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                instance.get_physical_device_properties2(p, &mut props);
                id.device_uuid
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
                    == uuid
            })
            .ok_or_else(|| format!("no native device with UUID {uuid}"))?;
        let family = instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("no graphics queue")? as u32;
        let exts: [&CStr; 2] = [
            c"VK_KHR_external_memory_fd",
            c"VK_KHR_external_semaphore_fd",
        ];
        let ext_ptrs: Vec<_> = exts.iter().map(|e| e.as_ptr()).collect();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let prio = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&prio)];
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&ext_ptrs)
                    .push_next(&mut v12)
                    .push_next(&mut v13),
                None,
            )
            .map_err(vk_err("vkCreateDevice"))?;
        let queue = device.get_device_queue(family, 0);
        let memory_props = instance.get_physical_device_memory_properties(physical);
        let mem_type = |bits: u32, flags: vk::MemoryPropertyFlags| {
            (0..memory_props.memory_type_count).find(|&i| {
                bits & (1 << i) != 0
                    && memory_props.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
        };

        // The timeline semaphore.
        let mut timeline =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = device
            .create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                None,
            )
            .map_err(vk_err("vkCreateSemaphore"))?;
        let sem_ext = ash::khr::external_semaphore_fd::Device::new(&instance, &device);
        sem_ext
            .import_semaphore_fd(
                &vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(semaphore)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                    .fd(sem_fd),
            )
            .map_err(vk_err("vkImportSemaphoreFdKHR"))?;
        let value = device
            .get_semaphore_counter_value(semaphore)
            .map_err(vk_err("vkGetSemaphoreCounterValue"))?;
        println!("import-test: imported timeline semaphore, value {value}");
        // Host → native: the host's GPU signals 1..=5.
        let t = Instant::now();
        let sems = [semaphore];
        let r = device.wait_semaphores(
            &vk::SemaphoreWaitInfo::default()
                .semaphores(&sems)
                .values(&[5]),
            20_000_000_000,
        );
        println!(
            "import-test: waited for host GPU signal 5: {r:?} after {} ms",
            t.elapsed().as_millis()
        );
        // Native → host: signal 10 from here; the host waits for it.
        device
            .signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(semaphore)
                    .value(10),
            )
            .map_err(vk_err("vkSignalSemaphore"))?;
        println!("import-test: signalled 10 natively");

        // The same image the host created, on the imported memory.
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST,
                    )
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .push_next(&mut external),
                None,
            )
            .map_err(vk_err("vkCreateImage"))?;
        let req = device.get_image_memory_requirements(image);
        if req.size != size {
            println!(
                "import-test: note: native image needs {} bytes, host allocated {size}",
                req.size
            );
        }
        // Which allocation the host's driver made is not visible here: try
        // dedicated and plain imports on every device-local type. A failed
        // import does not consume the fd, a successful one does.
        let mut memory = None;
        for dedicated_alloc in [true, false] {
            for ty in (0..memory_props.memory_type_count).filter(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && memory_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            }) {
                let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
                let mut import = vk::ImportMemoryFdInfoKHR::default()
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
                    .fd(mem_fd);
                let mut info = vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(ty)
                    .push_next(&mut import);
                if dedicated_alloc {
                    info = info.push_next(&mut dedicated);
                }
                let r = device.allocate_memory(&info, None);
                println!(
                    "import-test:   import (dedicated {dedicated_alloc}, type {ty}) -> {:?}",
                    r.as_ref().err()
                );
                if let Ok(m) = r {
                    memory = Some(m);
                    break;
                }
            }
            if memory.is_some() {
                break;
            }
        }
        let memory = memory.ok_or("no import variant accepted the memory fd")?;
        device
            .bind_image_memory(image, memory, 0)
            .map_err(vk_err("vkBindImageMemory"))?;

        // Read the pixels back.
        let bytes = u64::from(w) * u64::from(h) * 8;
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .map_err(vk_err("vkCreateBuffer"))?;
        let breq = device.get_buffer_memory_requirements(buffer);
        let bmem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(breq.size)
                    .memory_type_index(
                        mem_type(
                            breq.memory_type_bits,
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                        .ok_or("no host-visible type")?,
                    ),
                None,
            )
            .map_err(vk_err("vkAllocateMemory(readback)"))?;
        device
            .bind_buffer_memory(buffer, bmem, 0)
            .map_err(vk_err("vkBindBufferMemory"))?;
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
            .map_err(vk_err("vkCreateCommandPool"))?;
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
            .map_err(vk_err("vkAllocateCommandBuffers"))?[0];
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(vk_err("vkBeginCommandBuffer"))?;
        // Acquire from the external owner, keeping the GENERAL layout (and
        // contents) the host left the image in.
        let acquire = [vk::ImageMemoryBarrier2::default()
            .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(family)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&acquire),
        );
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::GENERAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })],
        );
        device
            .end_command_buffer(cmd)
            .map_err(vk_err("vkEndCommandBuffer"))?;
        let cmds = [cmd];
        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                vk::Fence::null(),
            )
            .map_err(vk_err("vkQueueSubmit"))?;
        device
            .queue_wait_idle(queue)
            .map_err(vk_err("vkQueueWaitIdle"))?;
        let ptr = device
            .map_memory(bmem, 0, bytes, vk::MemoryMapFlags::empty())
            .map_err(vk_err("vkMapMemory"))?;
        let values = std::slice::from_raw_parts(ptr.cast::<u16>(), (bytes / 2) as usize);
        let got = checksum(values);
        device.unmap_memory(bmem);

        device.destroy_command_pool(pool, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(bmem, None);
        device.destroy_semaphore(semaphore, None);
        device.destroy_image(image, None);
        device.free_memory(memory, None);
        device.destroy_device(None);
        instance.destroy_instance(None);

        println!("import-test: checksum host {expected:#018x}, native {got:#018x}");
        if got == expected {
            println!("import-test: OK: the host's output image is readable natively");
            Ok(())
        } else {
            Err("pixels differ".into())
        }
    }
}
