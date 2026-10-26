//! PipeWire backend (Linux).
//!
//! The capture card's HDMI audio is an `Audio/Source` node. One thread runs
//! a PipeWire main loop with two streams: capture from that node, playback
//! to the selected (or default) sink, both F32 stereo at 48 kHz with a small
//! requested quantum. Their process callbacks run on PipeWire's realtime
//! data thread and are joined by the same lock-free ring as on Windows; the
//! output side is the shared [`Playout`] (prefill, ±0.5 % drift correction,
//! volume). PipeWire converts formats and rate-matches the devices itself.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::Duration;

use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use rtrb::{Consumer, Producer, RingBuffer};

use crate::config::{AudioConfig, AudioShared};
use crate::matching::Endpoint;
use crate::playout::Playout;

const RATE: u32 = 48_000;
const CHANNELS: u32 = 2;
/// Requested graph quantum in frames (5.3 ms at 48 kHz); the end-to-end
/// latency is set by the ring fill target on top of it.
const QUANTUM: u32 = 256;
/// Bound for the registry roundtrip when listing nodes.
const LIST_TIMEOUT: Duration = Duration::from_secs(2);

fn io(e: pw::Error) -> std::io::Error {
    std::io::Error::other(e.to_string())
}

pub mod endpoints {
    pub use crate::matching::Endpoint;

    /// Lists audio nodes: sources (`capture`) or sinks. `id` is the node
    /// name (stable across restarts), `name` its description.
    pub fn list(capture: bool) -> std::io::Result<Vec<Endpoint>> {
        super::list_nodes(capture).map_err(super::io)
    }
}

pub mod passthrough {
    pub use super::AudioPassthrough;
    pub use crate::config::{AudioConfig, AudioShared};
}

fn list_nodes(capture: bool) -> Result<Vec<Endpoint>, pw::Error> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry()?;
    let class = if capture {
        "Audio/Source"
    } else {
        "Audio/Sink"
    };
    let found = Rc::new(RefCell::new(Vec::new()));

    let pending = core.sync(0)?;
    let ml = mainloop.clone();
    let _core = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pw::core::PW_ID_CORE && seq == pending {
                ml.quit();
            }
        })
        .register();
    let sink = found.clone();
    let _registry = registry
        .add_listener_local()
        .global(move |g| {
            if g.type_ != pw::types::ObjectType::Node {
                return;
            }
            let Some(props) = g.props else { return };
            if props.get(*pw::keys::MEDIA_CLASS) != Some(class) {
                return;
            }
            let Some(name) = props.get(*pw::keys::NODE_NAME) else {
                return;
            };
            let description = props
                .get(*pw::keys::NODE_DESCRIPTION)
                .or_else(|| props.get(*pw::keys::NODE_NICK))
                .unwrap_or(name);
            sink.borrow_mut().push(Endpoint {
                id: name.to_owned(),
                name: description.to_owned(),
            });
        })
        .register();
    let ml = mainloop.clone();
    let timer = mainloop.loop_().add_timer(move |_| ml.quit());
    let _ = timer.update_timer(Some(LIST_TIMEOUT), None);
    mainloop.run();
    let out = found.borrow().clone();
    Ok(out)
}

/// `EnumFormat` pod: F32LE interleaved, 48 kHz, FL/FR.
fn format_pod() -> Vec<u8> {
    let mut info = spa::param::audio::AudioInfoRaw::new();
    info.set_format(spa::param::audio::AudioFormat::F32LE);
    info.set_rate(RATE);
    info.set_channels(CHANNELS);
    let mut position = [0; spa::param::audio::MAX_CHANNELS];
    position[0] = spa::sys::SPA_AUDIO_CHANNEL_FL;
    position[1] = spa::sys::SPA_AUDIO_CHANNEL_FR;
    info.set_position(position);
    spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(spa::pod::Object {
            type_: spa::sys::SPA_TYPE_OBJECT_Format,
            id: spa::sys::SPA_PARAM_EnumFormat,
            properties: info.into(),
        }),
    )
    .map(|(c, _)| c.into_inner())
    .unwrap_or_default()
}

/// Capture-card audio → speakers with bounded, drift-corrected latency.
pub struct AudioPassthrough {
    shared: Arc<AudioShared>,
    quit: pw::channel::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for AudioPassthrough {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioPassthrough")
            .field("shared", &self.shared)
            .finish_non_exhaustive()
    }
}

impl AudioPassthrough {
    /// Connects both streams; fails if PipeWire refuses them.
    pub fn start(cfg: AudioConfig) -> std::io::Result<Self> {
        let shared = Arc::new(AudioShared::default());
        shared.volume_milli.store(
            (cfg.volume.clamp(0.0, 2.0) * 1000.0) as u32,
            Ordering::Relaxed,
        );
        shared.offset_ms.store(cfg.av_offset_ms, Ordering::Relaxed);
        shared.sample_rate.store(RATE, Ordering::Relaxed);
        shared.channels.store(CHANNELS, Ordering::Relaxed);
        let (quit, quit_rx) = pw::channel::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let sh = shared.clone();
        let thread = std::thread::Builder::new()
            .name("audio".into())
            .spawn(move || {
                let r = run(&cfg, sh, quit_rx, &ready_tx);
                if let Err(e) = r {
                    tracing::error!("audio passthrough stopped: {e}");
                    let _ = ready_tx.send(Err(e.to_string()));
                }
            })?;
        let mut p = Self {
            shared,
            quit,
            thread: Some(thread),
        };
        match ready_rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(())) => Ok(p),
            Ok(Err(e)) => {
                p.shutdown();
                Err(std::io::Error::other(e))
            }
            Err(_) => {
                p.shutdown();
                Err(std::io::Error::other("PipeWire did not answer"))
            }
        }
    }

    pub fn shared(&self) -> &AudioShared {
        &self.shared
    }

    pub fn set_volume(&self, v: f32) {
        self.shared
            .volume_milli
            .store((v.clamp(0.0, 2.0) * 1000.0) as u32, Ordering::Relaxed);
    }

    pub fn set_offset_ms(&self, ms: i32) {
        self.shared.offset_ms.store(ms, Ordering::Relaxed);
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        let _ = self.quit.send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for AudioPassthrough {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct CaptureState {
    ring: Producer<f32>,
}

struct PlaybackState {
    ring: Consumer<f32>,
    playout: Playout,
    scratch: Vec<f32>,
    shared: Arc<AudioShared>,
}

fn run(
    cfg: &AudioConfig,
    shared: Arc<AudioShared>,
    quit: pw::channel::Receiver<()>,
    ready: &std::sync::mpsc::Sender<Result<(), String>>,
) -> Result<(), pw::Error> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let ch = CHANNELS as usize;
    let latency = format!("{QUANTUM}/{RATE}");

    // 500 ms of headroom; the controller keeps it near the target.
    let (producer, consumer) = RingBuffer::<f32>::new(RATE as usize * ch / 2);

    let capture = pw::stream::StreamBox::new(
        &core,
        "CapturePlay capture",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Game",
            *pw::keys::NODE_LATENCY => latency.as_str(),
            *pw::keys::TARGET_OBJECT => cfg.input_id.as_str(),
            // Never fall back to another source: that would play a microphone.
            "node.dont-fallback" => "true",
            "node.dont-reconnect" => "true",
        },
    )?;
    let _capture_listener = capture
        .add_local_listener_with_user_data(CaptureState { ring: producer })
        .state_changed(|_, _, old, new| tracing::debug!(?old, ?new, "audio capture stream"))
        .process(|stream, st| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let Some(data) = buffer.datas_mut().first_mut() else {
                return;
            };
            let (offset, size) = {
                let c = data.chunk();
                (c.offset() as usize, c.size() as usize)
            };
            let Some(bytes) = data.data() else { return };
            let end = (offset + size).min(bytes.len());
            let (samples, _) = bytes[offset.min(end)..end].as_chunks::<4>();
            let n = samples.len().min(st.ring.slots());
            if let Ok(mut chunk) = st.ring.write_chunk_uninit(n) {
                let (a, b) = chunk.as_mut_slices();
                for (dst, src) in a.iter_mut().chain(b.iter_mut()).zip(samples) {
                    dst.write(f32::from_le_bytes(*src));
                }
                // SAFETY: the chunk has exactly `n` slots and there are at
                // least `n` samples, so every slot was written above.
                unsafe { chunk.commit(n) };
            }
        })
        .register()?;

    let mut playback_props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::MEDIA_ROLE => "Game",
        *pw::keys::NODE_LATENCY => latency.as_str(),
    };
    if let Some(out) = &cfg.output_id {
        playback_props.insert(*pw::keys::TARGET_OBJECT, out.as_str());
    }
    let playback = pw::stream::StreamBox::new(&core, "CapturePlay playback", playback_props)?;
    let state = PlaybackState {
        ring: consumer,
        playout: Playout::new(RATE, ch, cfg.target_latency_ms, cfg.av_offset_ms),
        scratch: Vec::new(),
        shared: shared.clone(),
    };
    let _playback_listener = playback
        .add_local_listener_with_user_data(state)
        .state_changed(|_, _, old, new| tracing::debug!(?old, ?new, "audio playback stream"))
        .process(move |stream, st| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let requested = buffer.requested() as usize;
            let Some(data) = buffer.datas_mut().first_mut() else {
                return;
            };
            let stride = 4 * ch;
            let frames = {
                let Some(bytes) = data.data() else { return };
                let capacity = bytes.len() / stride;
                let frames = if requested > 0 {
                    requested.min(capacity)
                } else {
                    capacity.min(QUANTUM as usize)
                };
                st.scratch.resize(frames * ch, 0.0);
                st.playout.fill(&mut st.scratch, &mut st.ring, &st.shared);
                for (dst, s) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(&st.scratch) {
                    *dst = s.to_le_bytes();
                }
                frames
            };
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = stride as i32;
            *chunk.size_mut() = (frames * stride) as u32;
        })
        .register()?;

    let pod = format_pod();
    let flags = pw::stream::StreamFlags::AUTOCONNECT
        | pw::stream::StreamFlags::MAP_BUFFERS
        | pw::stream::StreamFlags::RT_PROCESS;
    let mut params = [spa::pod::Pod::from_bytes(&pod).ok_or(pw::Error::CreationFailed)?];
    capture.connect(spa::utils::Direction::Input, None, flags, &mut params)?;
    let mut params = [spa::pod::Pod::from_bytes(&pod).ok_or(pw::Error::CreationFailed)?];
    playback.connect(spa::utils::Direction::Output, None, flags, &mut params)?;

    let ml = mainloop.clone();
    let _quit = quit.attach(mainloop.loop_(), move |()| ml.quit());
    tracing::info!(
        input = %cfg.input_id,
        output = ?cfg.output_id,
        rate = RATE,
        channels = CHANNELS,
        quantum = QUANTUM,
        "audio passthrough started"
    );
    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}
