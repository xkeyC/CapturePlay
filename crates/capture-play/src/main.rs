//! CapturePlay: capture card → (DLSS NR) → frame interpolation → native HDR.
//!
//! Threads:
//! * **main** – winit event loop only (window messages, input, fullscreen).
//! * **render** – D3D12, pacing, egui UI, owns the capture session.
//! * **capture** – Media Foundation reader (one copy into GPU upload heap).

#![windows_subsystem = "windows"]

mod app;
mod config;
mod cursor;
mod i18n;
mod render_thread;
mod ui;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn init_logging() -> Option<std::fs::File> {
    let path = config::data_dir().join("capture-play.log");
    let file = std::fs::File::create(&path).ok();
    let filter =
        EnvFilter::try_from_env("CAPTUREPLAY_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    match file.as_ref().and_then(|f| f.try_clone().ok()) {
        Some(f) => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(f))
            .init(),
        None => tracing_subscriber::fmt().with_env_filter(filter).init(),
    }
    file
}

fn main() -> Result<()> {
    let _log = init_logging();
    std::panic::set_hook(Box::new(|info| {
        tracing::error!("panic: {info}");
    }));
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "CapturePlay starting");
    let settings = config::load();
    app::run(settings)
}
