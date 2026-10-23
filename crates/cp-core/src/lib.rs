//! Platform-independent core of CapturePlay.
//!
//! Everything in this crate is pure Rust with no OS or GPU dependency, so it
//! can be unit-tested anywhere. GPU shaders mirror the math defined here; the
//! tests in this crate are the reference for those shaders.

pub mod cadence;
pub mod color;
pub mod format;
pub mod pacing;
pub mod settings;
pub mod time;

pub use color::{ColorDesc, Primaries, Range, TransferFunction, YuvMatrix};
pub use format::{PixelFormat, Rational, VideoMode};
