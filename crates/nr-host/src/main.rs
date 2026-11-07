//! `captureplay-nr-host`: runs NVIDIA's DLSS Neural Rendering snippet
//! (`nvngx_dlssnr.dll`, NGX feature 18) for the Linux build of CapturePlay.
//!
//! The snippet only exists for Windows, so this program is built for
//! Windows and runs under Wine/Proton, where it drives the host's NVIDIA
//! Vulkan driver through winevulkan. Calls go through `captureplay_nvngx.dll`
//! (nr-bridge), which the snippet accepts as its caller. Images cross to the
//! native process as Linux file descriptors exported by the driver.
//!
//! Experimental and unsupported, like the community NR path on Windows.

#[cfg(target_os = "linux")]
mod client;
#[cfg(windows)]
mod host;

fn main() {
    #[cfg(windows)]
    std::process::exit(host::main());
    #[cfg(target_os = "linux")]
    std::process::exit(client::main());
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        eprintln!("captureplay-nr-host is a Windows program; build it for x86_64-pc-windows-gnu");
        std::process::exit(2);
    }
}
