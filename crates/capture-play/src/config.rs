//! Settings file location and (de)serialization.

use std::path::PathBuf;

use cp_core::settings::Settings;

/// `%APPDATA%\CapturePlay` (created on demand).
pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("APPDATA").map_or_else(|| PathBuf::from("."), PathBuf::from);
    let dir = base.join("CapturePlay");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn settings_path() -> PathBuf {
    data_dir().join("settings.toml")
}

pub fn load() -> Settings {
    match std::fs::read_to_string(settings_path()) {
        Ok(text) => toml::from_str(&text).unwrap_or_else(|e| {
            tracing::warn!("settings.toml invalid, using defaults: {e}");
            Settings::new()
        }),
        Err(_) => Settings::new(),
    }
}

pub fn save(s: &Settings) {
    match toml::to_string_pretty(s) {
        Ok(text) => {
            // Write-then-rename so a crash never leaves a truncated file.
            let tmp = settings_path().with_extension("toml.tmp");
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, settings_path());
            }
        }
        Err(e) => tracing::warn!("cannot serialize settings: {e}"),
    }
}
