# CapturePlay

Low-latency capture-card player: **PS5 / console → HDMI capture card → PC → real-time frame interpolation → native HDR output**.
Rust + Direct3D 12 (Windows 10/11) / Vulkan (Linux, port in progress), single process, single swapchain.

[简体中文](README.md) | **English**

> Technical preview (0.1.0).

---

## 1. Features

### 1.1 Overview

CapturePlay treats the capture card as a display: the console image is presented on the PC with the lowest practical latency, then interpolated, upscaled and output in HDR at the display's refresh rate. It does not record or transcode; the whole chain is one process and one swapchain.

### 1.2 Features

| Feature | Description |
|---|---|
| Low-latency preview | Arrival→present ≈ 15 ms (30→60 content, RTX 4090) |
| AI interpolation | NVIDIA VFG; 30/40/50/60 fps content to the display rate, 2×/3×/4× |
| Native HDR | Fixed FP16 scRGB swapchain; HDR survives fullscreen/window changes |
| AI upscaling | NVIDIA RTX VSR, 1440p→4K, with a Catmull-Rom fallback |
| NR enhancement | DLSS Neural Rendering with 3 styles and several parameters; experimental |
| Variable refresh | G-SYNC / FreeSync / HDMI VRR; detects fixed refresh vs VRR |
| Audio passthrough | Capture-card HDMI audio through WASAPI, with clock-drift correction |
| UI | Chinese/English, drawn into the HDR back buffer, with GPU time and present jitter |

### 1.3 System and hardware requirements

**System**

| Platform | Requirement |
|---|---|
| Windows | Windows 10/11 x64; HDR needs "Use HDR" on in the display settings |
| Linux | A Wayland compositor; HDR needs the color-management protocol (`wp_color_manager_v1`), tested on KDE Plasma 6.8 (KWin). On X11 or without color management the output is SDR. Audio needs PipeWire; capture goes through V4L2 |
| NR on Linux | Proton 11 from Steam (Wine 10 or later; tested Proton 11.0, GE-Proton11-11, DW-Proton 11.0-16), started through `steam-run` on NixOS |

**Hardware**

| Feature | GPU | Other |
|---|---|---|
| Basic playback | Windows: a D3D12 GPU; Linux: a Vulkan 1.3 GPU | An HDMI capture card (UVC / V4L2) |
| HDR | — | An HDR display; a 10-bit (P010) capture |
| VFG interpolation | NVIDIA RTX 40/50 | NVIDIA Video Effects runtime |
| VSR upscaling | NVIDIA RTX 20 or later | Windows: RTX Video SDK; Linux: the Video Effects runtime (the build links the DLSS SDK's static NGX loader) |
| NR official | NVIDIA RTX 50 with a driver exposing NGX feature 18 | Windows |
| NR community (unofficial) | NVIDIA RTX 40/30/20 | User-supplied `nvngx_dlssnr.dll`; on Windows the `captureplay_nvngx.dll` built by this project, on Linux Proton and `captureplay-nr-host.exe` |
| VRR | — | A G-SYNC / FreeSync / HDMI VRR display; on Linux, KWin adaptive sync set to "Automatic" (fullscreen) or "Always" |

Developed and tested on an RTX 4090 with an AVerMedia Live Gamer Ultra 2.1 (USB 10 Gbps); on Linux with NixOS, NVIDIA driver 615 and KDE Plasma 6.8.

### 1.4 Usage

| Action | Shortcut |
|---|---|
| Show/hide panel | F1 / Tab |
| Fullscreen | F11 / double-click |
| Leave fullscreen | Esc |
| Freeze on an interpolated frame | P |

The mouse hides after 2 seconds of inactivity over the video (not while hovering the panel).

Settings live in `%APPDATA%\CapturePlay\settings.toml`; the log is `capture-play.log` in the same directory. `CAPTUREPLAY_LOG=debug` raises the log level and `CAPTUREPLAY_D3D_DEBUG=1` enables the D3D12 debug layer.

NR panel: backend, model (Auto / Quality / Performance), style (Default / Natural / Cinematic), model strength, composite strength; "More parameters" exposes local structure, local tone, skin structure, auto mask and UI correction.

### 1.5 Performance

RTX 4090, AVerMedia Live Gamer Ultra 2.1, 1440p P010, release build:

| Interpolation model | GPU per frame | Arrival→present |
|---|---|---|
| Fast | 0.95 ms | ≈15.3 ms |
| Balanced (default) | 1.80 ms | ≈15.4 ms |
| Quality | 8.29 ms | ≈16.0 ms |

| Item | GPU time | Notes |
|---|---|---|
| RTX VSR Medium / High / Ultra | 1.35 / 3.61 / 4.94 ms | High looks best; Ultra adds little on clean game content |
| NR (1440p) | ≈5.0 ms per unique frame | Community 310.9.16, intensity 1.0, default style |

These are reference numbers for this machine and content. Actual cost depends on the GPU, capture format and effect combination.

### 1.6 Capture cards

Taking the AVerMedia Live Gamer Ultra 2.1 as an example:

| Mode | Recommendation |
|---|---|
| **1440p60 P010** | Best for HDR; accurate duplicate detection and low USB bandwidth |
| 4K60 NV12 | 8-bit; HDR is converted to SDR |
| 4K30 P010 | Not recommended; the 60 Hz signal is sampled every other frame, causing duplicates and judder |

The USB model needs no hardware tone-mapping setup; PCIe models disable it automatically.

### 1.7 Build and run

Requirements: Rust 1.95+ (MSVC), Windows SDK (`dxc.exe`, or set `DXC_PATH`), a D3D12 GPU.

~~~powershell
cargo build --release
.\target\release\capture-play.exe
~~~

Optional components go into `runtime/` (git-ignored):
- `runtime/nvidia-vfx/`: VFG runtime;
- `runtime/rtx-video-sdk/`: RTX Video SDK (link `nvsdk_ngx_d.lib` at build time, load `nvngx_vsr.dll` at run time);
- `runtime/nvidia-nr/`: community `nvngx_dlssnr.dll` and `captureplay_nvngx.dll`.

`cargo build` also produces `captureplay_nvngx.dll`. The first time VSR/NR is enabled, model creation takes about 2–3 seconds; frames pass through meanwhile.

**Linux (port in progress)**: V4L2 capture + Vulkan rendering + Wayland color management (tested on KDE Plasma 6). Requirements: Rust 1.95+, `dxc` (HLSL → SPIR-V; override with `DXC_PATH`), a Vulkan 1.3 GPU. On NixOS, `flake.nix` provides the development shell (dxc, runtime libraries, validation layers):

~~~sh
nix develop
cargo build --release
./target/release/capture-play
~~~

Status: capture, color conversion, HDR output (scRGB, mapped through the compositor's reference luminance), the UI, present-timing feedback (`VK_EXT_present_timing`) and PipeWire audio passthrough work; interpolation (VFG, CUDA ↔ Vulkan interop) and upscaling (VSR, NGX on Vulkan) are implemented, need the NVIDIA Video Effects runtime (see `runtime/README.md`) and await validation on real hardware; NR runs through Proton (below). Settings live in `$XDG_CONFIG_HOME/CapturePlay/` (default `~/.config/CapturePlay/`); `CAPTUREPLAY_VK_DEBUG=1` enables the Vulkan validation layer, `CAPTUREPLAY_VK_TIMED_PRESENT=1` tries presenting at target times (experimental).

VRR is KWin's call: set Display Settings → Adaptive sync to "Automatic" (applies to the active fullscreen window) or "Always", then go fullscreen with F11. The app reads that policy as the starting point of the refresh detection and tags its window as game content, so TVs switch to their low-latency game mode. With V-Sync off, allow tearing in fullscreen in KDE.

NR (experimental): NVIDIA ships `nvngx_dlssnr.dll` for Windows only, so on Linux `captureplay-nr-host.exe` runs the model under Proton (NGX on Vulkan through winevulkan); frames and synchronisation cross between the processes as GPU-memory fds and timeline semaphores, with no copies and no per-frame CPU waits; inference takes about 5.7 ms at 1440p. Setup: `rustup target add x86_64-pc-windows-gnu`, run `scripts/build-nr-host.sh` inside `nix develop` (builds the host and `captureplay_nvngx.dll` and installs them into `runtime/nvidia-nr/`), then add your own `nvngx_dlssnr.dll`. Proton defaults to the newest one in Steam (through `steam-run` on NixOS); `nr_proton = ["steam-run", "/…/proton"]` and `nr_proton_prefix` under `[enhance]` in `settings.toml` override it. Enabling NR takes a few seconds to start Proton, with frames passing through meanwhile. The host logs to `captureplay-nr-host.log` in the Wine prefix (default `~/.local/share/CapturePlay/nr-prefix/`).

### 1.8 Notes

- HDR requires both the display and Windows HDR to be enabled; the swapchain reapplies its output contract after fullscreen or window changes.
- NR is experimental. Community runtimes are unsigned and may crash; use them for learning and experimentation only. The official route depends on NVIDIA exposing feature 18 in the driver.
- If the UI reports fixed refresh on a VRR-capable display, disable NVIDIA or third-party overlays and test again.

---

## 2. Implementation

### 2.1 Architecture and threading

~~~
 Capture thread (MF async)          Render thread (D3D12 + egui)        Main thread (winit)
 ───────────────────────           ─────────────────────────────      ─────────────
 IMFSourceReader ──memcpy──▶ UploadRing ──▶ Cadence ──▶ ConvertPass (YUV → linear scRGB)
   (one copy, fingerprint)   (persistent mapped heap)             │
                                                             NR (optional)
                                                                  │
                       Pacer (event-driven) ──▶ Interpolator (VFG) ──▶ OutputPass
                                                                (scale + tone map)
                                                                      │
 WASAPI capture ──SPSC ring──▶ WASAPI output                  EguiPass ──▶ FP16 scRGB swapchain
   (drift correction ±0.5%)                                      (UI in the same back buffer)
~~~

| Crate | Responsibility |
|---|---|
| `cp-core` | Color math, format selection, cadence/pacing, settings |
| `cp-capture` | Media Foundation capture: devices, formats, async reads, color detection |
| `cp-audio` | WASAPI passthrough, endpoint matching, clock-drift correction |
| `cp-render` | D3D12: upload, conversion, NR, interpolation, upscaling, output, swapchain/HDR, egui |
| `cp-nvidia` | CUDA driver subset, Video Effects (VFG), NGX (VSR/NR) |
| `nr-bridge` | Forwarding host `captureplay_nvngx.dll` for community `nvngx_dlssnr.dll` |
| `capture-play` | Window, thread orchestration, UI, settings persistence |

The main thread only pumps window messages; the render thread owns the device, swapchain, capture session and UI. The render thread sleeps and wakes on the pacer's deadlines, so moving or resizing the window does not block the picture.

### 2.2 Capture and color

- Media Foundation `IMFSourceReader` in async mode, `MF_LOW_LATENCY`, format converters disabled, driver queue depth 2.
- One copy per frame: driver buffer → persistent mapped D3D12 upload heap → GPU copy into a planar texture; a 64×36 luma fingerprint is sampled during the copy.
- When the renderer falls behind, the oldest unconsumed frame is recycled, so latency does not accumulate.
- Working space: linear scRGB (BT.709 primaries, 1.0 = 80 nits, FP16).
- The YUV→RGB matrix is composed on the CPU from matrix, range and bit depth; PQ/HLG/SDR are decoded to absolute luminance or SDR white.
- Color detection priority: user override > explicit driver HDR flag > 10-bit heuristic > driver flag > resolution default.

### 2.3 Cadence and retiming

- A 30 fps console game appears twice per 60 Hz signal; interpolating between duplicates is meaningless. `CadenceTracker` detects unique frames from fingerprint differences and estimates the content interval.
- Capture cards quantize VRR input to a fixed 60 Hz, so unique-frame intervals alternate between 16.7 and 33.3 ms. `Retimer` fits the latest line through the quantization bounds over the last 24 frames to rebuild even content times. Locked content is unaffected; variable-rate content waits about 5 ms longer on average.

### 2.4 Pacing and presentation

- `Pacer` schedules `M-1` intermediate frames per content interval; the added latency is `(M-1)/M × source interval`: half a frame (16.7 ms) for 30→60, the theoretical minimum. Default is 2×, with 3×/4× available.
- Gaps longer than 1.75× the content interval (loads, stalls) are not interpolated; the real frame is shown.
- The first slot = content time + offset. The offset is derived from the p95 recent processing budget plus composition lead, keeping arrival jitter off the screen.
- The swapchain uses FLIP_DISCARD, 3 buffers, `SetMaximumFrameLatency(1)` and a waitable object; tearing is allowed when vsync is off.
- On VRR the frame is presented exactly at its target time; on fixed refresh targets snap to vblanks and only integer refresh-period multipliers are used. `GetFrameStatistics` separates the two and reports present jitter.
- Upscaling runs after interpolation, once per presented frame, in the same GPU queue.

### 2.5 HDR output

- The swapchain is always `R16G16B16A16_FLOAT` + `G10_NONE_P709`; moving between monitors or toggling Windows HDR does not recreate it.
- HDR displays get a hue-preserving BT.2390 EETF tone map in PQ to the display peak; SDR displays map to SDR white.
- The UI and video share one back buffer, with the UI drawn at the Windows SDR-white reference.
- After every `ResizeBuffers` the color space is reapplied and HDR10 metadata cleared, so fullscreen/window changes do not lose HDR.

### 2.6 Frame interpolation

- Frames are encoded into D3D12/CUDA shared buffers (RGB10A2; PQ/BT.2020 for HDR, sRGB for SDR); `NvVFX_Run` generates any t in `Timestep` mode.
- D3D12 Signal → CUDA Wait → Run → CUDA Signal → D3D12 Wait, with no CPU copies or waits.
- Backends plug in through the `Interpolator` trait.

### 2.7 Upscaling

- `cp_nvidia::ngx` is a hand-written NGX FFI, initialized once per process and serialized.
- `NgxVsrUpscaler` encodes FP16 scRGB to PQ BT.2020 10-bit, invokes VSR and decodes back into the working frame; everything is recorded in the frame's own command list, ordered with interpolation and output.
- The model is created on a background thread; size and quality are per-call parameters. Upscaling is post-interpolation and runs once per presented frame.
- The target size is the smallest VSR factor covering the display; resize changes are debounced. Catmull-Rom in the PQ domain is the fallback.

### 2.8 NR

- Order: conversion → NR (once per unique frame) → interpolation → upscaling → output. NR runs before both, so interpolated and upscaled frames are built on the enhanced image.
- Data flow: the linear frame is encoded into an sRGB proxy (HDR uses an invertible highlight roll-off) → NGX runs at the same resolution (Color / constant Depth / zero MVec / Output) → the result is composited back as a bounded linear residual.
- Styles 0/1/2; model intensity, local structure, local tone, skin structure, auto mask and UI correction are per-evaluation parameters and apply immediately. Changing the model preset recreates the feature.
- Official route (RTX50): create NGX feature 18 directly; if the driver does not expose it, the UI shows unavailable and passes frames through.
- Community route (Unofficial RTX40/30/20): `captureplay_nvngx.dll` loads the user-supplied `nvngx_dlssnr.dll` and forwards the NGX entry points. The community DLL is unsigned, unsupported and intended for learning only.
- Deployment: put `captureplay_nvngx.dll` next to the executable or in `runtime/nvidia-nr/`, put `nvngx_dlssnr.dll` in `runtime/nvidia-nr/`, then select the community backend in the UI. Restart after replacing the DLL.

### 2.9 Audio

Capture-card HDMI audio is an ordinary WASAPI capture endpoint, matched to the video device by product name. Two event-driven threads are connected by a lock-free SPSC ring; the output side is initialized in the capture format and handed to Windows conversion (`AUTOCONVERTPCM`). Clock drift between the two devices is absorbed by ±0.5% linear resampling controlled by buffer fill, with a 30 ms default buffer and an adjustable A/V offset.

---

## 3. Licensing and SDKs

### 3.1 Project license

CapturePlay is released under **GPL-3.0-or-later**; see `LICENSE`. The additional permission in `LICENSE-EXCEPTION.md` allows linking or combining with NVIDIA proprietary components (RTX Video SDK / NGX, Video Effects SDK, CUDA). Those SDKs are not part of this project and are not covered by the GPL; they remain under NVIDIA's terms.

### 3.2 NVIDIA SDKs and runtimes

| Component | Purpose | How to obtain | Distribution |
|---|---|---|---|
| `nvcuda.dll` | CUDA driver (VFG interop) | Ships with the NVIDIA driver | Not distributed here |
| Video Effects runtime (`runtime/nvidia-vfx/`) | VFG interpolation | From NVIDIA | Shipped as object code inside the application, notices retained |
| RTX Video SDK (`runtime/rtx-video-sdk/`) | VSR; links `nvsdk_ngx_d.lib` | From NVIDIA's developer site | Ship `nvngx_vsr.dll` inside the app, retain notices; notify NVIDIA before a commercial release |
| NGX feature 18 (NR) | DLSS Neural Rendering | Official or community runtime, see 3.3 | This project does not distribute `nvngx_dlssnr.dll` |

Without the RTX Video SDK, `cfg(ngx)` is off; the application still builds, without the VSR/NR NGX backends.

### 3.3 Community NR runtime

- `captureplay_nvngx.dll` is built from `crates/nr-bridge` and is GPL code; it contains no NVIDIA code.
- `nvngx_dlssnr.dll` is a community-modified, unsigned binary. It is not in this repository and is not distributed; users place it in `runtime/nvidia-nr/` themselves.
- The community snippet validates the caller module name; the host exists only to satisfy that check. This is not an NVIDIA-supported usage and may violate its license terms. Use it for learning and experimentation only, not as a stable or commercial feature.

### 3.4 runtime directory

`runtime/` is git-ignored (only `runtime/README.md` is tracked):

| Path | Contents | Environment variables |
|---|---|---|
| `runtime/rtx-video-sdk/` | RTX Video SDK | Build `CAPTUREPLAY_RTX_VIDEO_SDK`; run `CAPTUREPLAY_RTX_VIDEO_RUNTIME` |
| `runtime/nvidia-vfx/` | Video Effects runtime | `CAPTUREPLAY_VFX_RUNTIME` |
| `runtime/nvidia-nr/` | Community `nvngx_dlssnr.dll` + `captureplay_nvngx.dll` | `CAPTUREPLAY_NR_RUNTIME`, `CAPTUREPLAY_NR_BRIDGE` |

### 3.5 Third-party attribution

- The NR proxy/residual color math follows an MIT-licensed implementation (video2dlssnr, © 2026 Daniil Sokolyuk); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full license.
- CUDA FFI declarations are hand-written from NVIDIA's public headers and documentation; the repository contains no NVIDIA headers, libraries or runtimes.
