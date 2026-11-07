#!/usr/bin/env bash
# Builds the DLSS NR host for Linux — captureplay-nr-host.exe and
# captureplay_nvngx.dll, Windows programs run under Proton — and installs
# them into runtime/nvidia-nr, where CapturePlay looks for them (with
# nvngx_dlssnr.dll, which you provide).
#
# Run inside `nix develop` (it provides the mingw cross linker), with the
# Rust target installed: `rustup target add x86_64-pc-windows-gnu`.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --target x86_64-pc-windows-gnu -p nr-host -p nr-bridge
out=target/x86_64-pc-windows-gnu/release
mkdir -p runtime/nvidia-nr
install -m644 "$out/captureplay-nr-host.exe" "$out/captureplay_nvngx.dll" runtime/nvidia-nr/
echo "installed into runtime/nvidia-nr:"
ls -1 runtime/nvidia-nr
