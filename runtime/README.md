# runtime/

Optional third-party SDKs and runtimes, obtained by the user from NVIDIA and
ignored by git. None of them is part of CapturePlay or covered by its GPL
license (see `LICENSE-EXCEPTION.md`); each stays under NVIDIA's own terms.

| Folder | Contents | Used by |
|---|---|---|
| `rtx-video-sdk/` | NVIDIA RTX Video SDK 1.1.0, unpacked as-is (`include/`, `lib/`, `bin/`, `doc/`, `samples/`, license PDF). Download `RTX_Video_SDK_v1.1.0.zip` from <https://developer.nvidia.com/rtx-video-sdk/getting-started> (NVIDIA developer login). | AI upscaling (RTX Video Super Resolution) through NGX on D3D12: `lib/Windows/x64/nvsdk_ngx_d.lib` is linked at build time, `bin/Windows/x64/rel/nvngx_vsr.dll` is loaded at run time |
| `nvidia-nr/` | `nvngx_dlssnr.dll` for DLSS Neural Rendering (NGX feature 18), if the driver/SDK you use provides it. Not shipped or downloadable from this repository; the feature is unavailable while the installed NGX runtime does not expose feature 18. | Experimental NR stage (`cp_render::NgxNrEnhancer`) |
| `nvidia-vfx/` | From NVIDIA's Video Effects SDK (`nvidia-vfx` wheel on pypi.nvidia.com, `nvvfx/libs/`): `cudart64_12.dll`, `NVCVImage.dll`, `nvngxruntime.dll`, `NVVideoEffects.dll`, `nvVFXVideoFrameGeneration.dll` | AI frame interpolation (RTX 40/50) |

All DLLs are NVIDIA-signed (check with `Get-AuthenticodeSignature`).

## RTX Video SDK

Build time: `crates/cp-nvidia/build.rs` looks for the SDK in
`$CAPTUREPLAY_RTX_VIDEO_SDK`, `$NV_RTX_VIDEO_SDK`, then `runtime/rtx-video-sdk`.
Without it the build still succeeds and the upscaler is unavailable (the
output pass scales instead).

Run time: `nvngx_vsr.dll` is searched in `$CAPTUREPLAY_RTX_VIDEO_RUNTIME`, the
executable's folder, then `runtime/rtx-video-sdk/bin/Windows/x64/rel` next to
the executable or in any parent of it. NGX writes its logs/caches to
`%APPDATA%\CapturePlay\ngx`.

License obligations when distributing a build (NVIDIA RTX SDKs License and
its supplement): ship `nvngx_vsr.dll` only inside the application (not as a
stand-alone SDK), keep NVIDIA's notices, attribute the SDK in the app (the
overlay shows it) and notify NVIDIA before any *commercial* release.

## DLSS Neural Rendering (NR)

Lookup order: `enhance.nr_runtime_dir` in `settings.toml` →
`CAPTUREPLAY_NR_RUNTIME` → the executable's folder → `runtime/nvidia-nr` next
to the executable or in any parent directory.

Two routes exist:

- **Official**: the NR stage asks `NVSDK_NGX_D3D12_GetFeatureRequirements` for
  feature 18 and creates it through the installed NGX runtime. The public
  feature table has no entry for it, so this only works when the driver
  exposes it.
- **Community (opt-in)**: `captureplay_nvngx.dll` is CapturePlay's own
  forwarding host (no NVIDIA code). It loads the community snippet here and
  forwards the NGX D3D12 entry points; NVIDIA's caller-module check in the
  snippet then accepts the call. Community builds are unsigned, patched for
  more GPU architectures, unsupported and may crash: use them for learning
  and experimentation only. No NVIDIA DLL is modified.

Deployment: put `nvngx_dlssnr.dll` here (or set `CAPTUREPLAY_NR_RUNTIME`),
put the built `captureplay_nvngx.dll` next to the executable (or here; set
`CAPTUREPLAY_NR_BRIDGE` to override) and select the community backend in the
NR panel. The host caches the first snippet it loads, so restart after
replacing the DLL.

NGX is initialized once per process; the feature search paths of the upscaler
and of NR are merged for that call.

## Video Effects SDK

Lookup order: `interpolation.vfx_runtime_dir` in `settings.toml` →
`CAPTUREPLAY_VFX_RUNTIME` → `runtime/nvidia-vfx` next to the executable or in
any parent directory of it (so `target/release` builds find this folder).

The CUDA driver (`nvcuda.dll`) ships with the display driver and needs
nothing here.
