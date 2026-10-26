//! Platforms without an audio backend yet (Linux until the PipeWire
//! backend lands): the same API as the WASAPI modules; no endpoints are
//! found and passthrough reports `Unsupported`.

pub mod endpoints {
    pub use crate::matching::Endpoint;

    /// No endpoints on this platform yet.
    pub fn list(_capture: bool) -> std::io::Result<Vec<Endpoint>> {
        Ok(Vec::new())
    }
}

pub mod passthrough {
    pub use crate::config::{AudioConfig, AudioShared};

    #[derive(Debug)]
    pub struct AudioPassthrough {
        shared: AudioShared,
    }

    impl AudioPassthrough {
        pub fn start(_cfg: AudioConfig) -> std::io::Result<Self> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "audio passthrough is not implemented on this platform yet",
            ))
        }

        pub fn shared(&self) -> &AudioShared {
            &self.shared
        }

        pub fn set_volume(&self, _v: f32) {}

        pub fn set_offset_ms(&self, _ms: i32) {}

        pub fn stop(self) {}
    }
}
