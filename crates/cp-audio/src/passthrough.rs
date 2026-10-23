//! Capture-card audio → speakers with bounded, drift-corrected latency.
//!
//! Two event-driven WASAPI threads (MMCSS "Pro Audio") joined by a lock-free
//! SPSC ring of interleaved `f32` frames. The render side initializes with
//! the capture format and lets Windows convert (`AUTOCONVERTPCM`), so the
//! only resampling we do is the ±0.5 % drift correction.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::thread::JoinHandle;

use rtrb::{Consumer, Producer, RingBuffer};
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::*;
use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
use windows::Win32::Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, CreateEventW,
    WaitForSingleObject,
};
use windows::core::w;

use crate::drift::{DriftController, RatePuller};
use crate::endpoints::open_device;

#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Capture endpoint id.
    pub input_id: String,
    /// Render endpoint id; `None` = default console device.
    pub output_id: Option<String>,
    pub target_latency_ms: u32,
    /// Extra audio delay to line up with the video pipeline (may be < 0 to
    /// reduce the target, never below a safety floor).
    pub av_offset_ms: i32,
    pub volume: f32,
}

/// Live counters/controls shared with the UI.
#[derive(Debug, Default)]
pub struct AudioShared {
    pub volume_milli: AtomicU32,
    pub offset_ms: AtomicI32,
    pub underruns: AtomicU64,
    pub fill_ms: AtomicU32,
    pub sample_rate: AtomicU32,
    pub channels: AtomicU32,
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("WASAPI: {0}")]
    Os(#[from] windows::core::Error),
    #[error("unsupported capture format (tag {0:#x}, {1} bit)")]
    Format(u16, u16),
}

struct Com;
impl Com {
    fn init() -> windows::core::Result<Self> {
        // SAFETY: balanced in Drop on the same thread.
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED).ok()? };
        Ok(Self)
    }
}
impl Drop for Com {
    fn drop(&mut self) {
        // SAFETY: paired with init.
        unsafe { CoUninitialize() };
    }
}

struct Mmcss(Option<HANDLE>);
impl Mmcss {
    fn pro_audio() -> Self {
        let mut idx = 0u32;
        // SAFETY: plain MMCSS registration for this thread.
        Self(unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut idx) }.ok())
    }
}
impl Drop for Mmcss {
    fn drop(&mut self) {
        if let Some(h) = self.0 {
            // SAFETY: handle from AvSetMmThreadCharacteristicsW.
            unsafe {
                let _ = AvRevertMmThreadCharacteristics(h);
            }
        }
    }
}

struct Event(HANDLE);
impl Event {
    fn new() -> windows::core::Result<Self> {
        // SAFETY: anonymous auto-reset event.
        Ok(Self(unsafe { CreateEventW(None, false, false, None)? }))
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: owned handle.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// Sample encoding of the capture endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleKind {
    F32,
    I16,
    I24In32,
    I32,
}

fn sample_kind(fmt: &WAVEFORMATEX) -> Result<SampleKind, AudioError> {
    let tag = u32::from(fmt.wFormatTag);
    let bits = fmt.wBitsPerSample;
    let is_float = if tag == WAVE_FORMAT_EXTENSIBLE {
        // SAFETY: tag says this is a WAVEFORMATEXTENSIBLE.
        let ext = unsafe { &*std::ptr::from_ref(fmt).cast::<WAVEFORMATEXTENSIBLE>() };
        let sub = ext.SubFormat;
        sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
    } else {
        tag == WAVE_FORMAT_IEEE_FLOAT
    };
    let valid = if tag == WAVE_FORMAT_EXTENSIBLE {
        // SAFETY: as above.
        unsafe {
            (*std::ptr::from_ref(fmt).cast::<WAVEFORMATEXTENSIBLE>())
                .Samples
                .wValidBitsPerSample
        }
    } else {
        bits
    };
    match (is_float, bits, valid) {
        (true, 32, _) => Ok(SampleKind::F32),
        (false, 16, _) => Ok(SampleKind::I16),
        (false, 32, 24) => Ok(SampleKind::I24In32),
        (false, 32, _) => Ok(SampleKind::I32),
        _ => Err(AudioError::Format(fmt.wFormatTag, bits)),
    }
}

fn float_format(rate: u32, channels: u16) -> WAVEFORMATEX {
    WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_IEEE_FLOAT as u16,
        nChannels: channels,
        nSamplesPerSec: rate,
        nAvgBytesPerSec: rate * u32::from(channels) * 4,
        nBlockAlign: channels * 4,
        wBitsPerSample: 32,
        cbSize: 0,
    }
}

#[derive(Debug)]
pub struct AudioPassthrough {
    stop: Arc<AtomicBool>,
    shared: Arc<AudioShared>,
    threads: Vec<JoinHandle<()>>,
}

impl AudioPassthrough {
    pub fn start(cfg: AudioConfig) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(AudioShared::default());
        shared.volume_milli.store(
            (cfg.volume.clamp(0.0, 2.0) * 1000.0) as u32,
            Ordering::Relaxed,
        );
        shared.offset_ms.store(cfg.av_offset_ms, Ordering::Relaxed);
        // The capture side decides the format and hands the ring's read end
        // to the render side.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<(u32, u16, Consumer<f32>)>();

        let (s1, sh1, c1) = (stop.clone(), shared.clone(), cfg.clone());
        let capture = std::thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || {
                if let Err(e) = capture_thread(&c1, &s1, &sh1, ready_tx) {
                    tracing::error!("audio capture stopped: {e}");
                }
            })?;
        let (s2, sh2) = (stop.clone(), shared.clone());
        let render = std::thread::Builder::new()
            .name("audio-render".into())
            .spawn(move || {
                let Ok((rate, ch, consumer)) = ready_rx.recv() else {
                    return;
                };
                if let Err(e) = render_thread(&cfg, &s2, &sh2, rate, ch, consumer) {
                    tracing::error!("audio render stopped: {e}");
                }
            })?;
        Ok(Self {
            stop,
            shared,
            threads: vec![capture, render],
        })
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
        self.stop.store(true, Ordering::Release);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for AudioPassthrough {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn capture_thread(
    cfg: &AudioConfig,
    stop: &AtomicBool,
    shared: &AudioShared,
    ready_tx: std::sync::mpsc::Sender<(u32, u16, Consumer<f32>)>,
) -> Result<(), AudioError> {
    let _com = Com::init()?;
    let _mmcss = Mmcss::pro_audio();
    let device = open_device(&cfg.input_id)?;
    let event = Event::new()?;
    // SAFETY: standard shared-mode event-driven capture setup; the mix
    // format pointer is freed with CoTaskMemFree.
    unsafe {
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let mix = client.GetMixFormat()?;
        let kind = sample_kind(&*mix);
        let (rate, channels) = ((*mix).nSamplesPerSec, (*mix).nChannels);
        let init = client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            100_000,
            0,
            mix,
            None,
        );
        CoTaskMemFree(Some(mix as *const _));
        let kind = kind?;
        init?;
        client.SetEventHandle(event.0)?;
        let cap: IAudioCaptureClient = client.GetService()?;
        shared.sample_rate.store(rate, Ordering::Relaxed);
        shared
            .channels
            .store(u32::from(channels), Ordering::Relaxed);
        tracing::info!(rate, channels, ?kind, "audio capture started");

        // 500 ms of headroom; the controller keeps it near the target.
        let (mut producer, consumer): (Producer<f32>, Consumer<f32>) =
            RingBuffer::new(rate as usize * usize::from(channels) / 2);
        let _ = ready_tx.send((rate, channels, consumer));

        client.Start()?;
        let ch = usize::from(channels);
        while !stop.load(Ordering::Acquire) {
            if WaitForSingleObject(event.0, 100) != WAIT_OBJECT_0 {
                continue;
            }
            loop {
                let n = cap.GetNextPacketSize()?;
                if n == 0 {
                    break;
                }
                let mut data = std::ptr::null_mut();
                let mut frames = 0u32;
                let mut flags = 0u32;
                cap.GetBuffer(&mut data, &mut frames, &mut flags, None, None)?;
                let samples = frames as usize * ch;
                let silent = flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0;
                if let Ok(mut chunk) = producer.write_chunk_uninit(samples.min(producer.slots())) {
                    let n = chunk.len();
                    let (a, b) = chunk.as_mut_slices();
                    let mut i = 0usize;
                    let mut put = |dst: &mut std::mem::MaybeUninit<f32>| {
                        let v = if silent {
                            0.0
                        } else {
                            match kind {
                                SampleKind::F32 => *data.cast::<f32>().add(i),
                                SampleKind::I16 => f32::from(*data.cast::<i16>().add(i)) / 32768.0,
                                SampleKind::I24In32 | SampleKind::I32 => {
                                    *data.cast::<i32>().add(i) as f32 / 2_147_483_648.0
                                }
                            }
                        };
                        dst.write(v);
                        i += 1;
                    };
                    a.iter_mut().for_each(&mut put);
                    b.iter_mut().for_each(&mut put);
                    chunk.commit(n);
                }
                cap.ReleaseBuffer(frames)?;
            }
        }
        client.Stop()?;
    }
    Ok(())
}

fn render_thread(
    cfg: &AudioConfig,
    stop: &AtomicBool,
    shared: &AudioShared,
    rate: u32,
    channels: u16,
    mut consumer: Consumer<f32>,
) -> Result<(), AudioError> {
    let _com = Com::init()?;
    let _mmcss = Mmcss::pro_audio();
    let device = match &cfg.output_id {
        Some(id) => open_device(id)?,
        None => {
            // SAFETY: plain COM activation.
            let e: IMMDeviceEnumerator =
                unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? };
            // SAFETY: valid enumerator.
            unsafe { e.GetDefaultAudioEndpoint(eRender, eConsole)? }
        }
    };
    let event = Event::new()?;
    let ch = usize::from(channels);
    let fmt = float_format(rate, channels);
    // SAFETY: standard shared-mode render setup with our own float format.
    unsafe {
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK
            | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
            | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 200_000, 0, &fmt, None)?;
        client.SetEventHandle(event.0)?;
        let buffer_frames = client.GetBufferSize()?;
        let render: IAudioRenderClient = client.GetService()?;
        tracing::info!(rate, channels, buffer_frames, "audio render started");

        let frames_per_ms = f64::from(rate) / 1000.0;
        let target_for = |offset: i32| {
            f64::from((cfg.target_latency_ms as i32 + offset).max(10)) * frames_per_ms
        };
        let mut ctl = DriftController::new(target_for(shared.offset_ms.load(Ordering::Relaxed)));
        let mut puller = RatePuller::new(ch);
        let mut waiting = true; // prefill to target before playing
        let mut scratch = vec![0f32; buffer_frames as usize * ch];
        client.Start()?;
        while !stop.load(Ordering::Acquire) {
            if WaitForSingleObject(event.0, 100) != WAIT_OBJECT_0 {
                continue;
            }
            ctl.set_target(target_for(shared.offset_ms.load(Ordering::Relaxed)));
            let padding = client.GetCurrentPadding()?;
            let avail = (buffer_frames - padding) as usize;
            if avail == 0 {
                continue;
            }
            let fill = (consumer.slots() / ch) as f64;
            shared
                .fill_ms
                .store((fill / frames_per_ms) as u32, Ordering::Relaxed);
            let excess = ctl.excess(fill);
            if excess > 0 {
                if let Ok(c) = consumer.read_chunk(excess * ch) {
                    c.commit_all();
                }
                ctl.reset();
            }
            if waiting && fill >= ctl.target() {
                waiting = false;
            }
            let out = &mut scratch[..avail * ch];
            if waiting {
                out.fill(0.0);
            } else {
                let ratio = ctl.ratio(fill);
                let real = puller.fill(out, ratio, |frame| {
                    if consumer.slots() < frame.len() {
                        return false;
                    }
                    match consumer.read_chunk(frame.len()) {
                        Ok(c) => {
                            let (a, b) = c.as_slices();
                            frame[..a.len()].copy_from_slice(a);
                            frame[a.len()..].copy_from_slice(b);
                            c.commit_all();
                            true
                        }
                        Err(_) => false,
                    }
                });
                if real < avail {
                    shared.underruns.fetch_add(1, Ordering::Relaxed);
                    waiting = true;
                }
                let vol = shared.volume_milli.load(Ordering::Relaxed) as f32 / 1000.0;
                if (vol - 1.0).abs() > f32::EPSILON {
                    out.iter_mut().for_each(|s| *s *= vol);
                }
            }
            let dst = render.GetBuffer(avail as u32)?;
            std::ptr::copy_nonoverlapping(out.as_ptr(), dst.cast::<f32>(), out.len());
            render.ReleaseBuffer(avail as u32, 0)?;
        }
        client.Stop()?;
    }
    Ok(())
}
