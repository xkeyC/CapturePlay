# runtime/

Optional third-party runtimes, installed locally and ignored by git.

| Folder | Contents | Used by |
|---|---|---|
| `nvidia-vfx/` | From NVIDIA's Video Effects SDK (`nvidia-vfx` wheel on pypi.nvidia.com, `nvvfx/libs/`): `cudart64_12.dll`, `NVCVImage.dll`, `nvngxruntime.dll`, `NVVideoEffects.dll`, `nvVFXVideoFrameGeneration.dll` (VFG), `nvVFXVideoSuperRes.dll` + `nvngx_vsr.dll` (VSR), `nvVFXTrueHDR.dll` + `nvngx_truehdr.dll` | AI frame interpolation (RTX 40/50), AI upscaling |

All files are NVIDIA-signed (check with `Get-AuthenticodeSignature`).

Lookup order: `interpolation.vfx_runtime_dir` in `settings.toml` →
`CAPTUREPLAY_VFX_RUNTIME` → `runtime/nvidia-vfx` next to the executable or in
any parent directory of it (so `target/release` builds find this folder).

The CUDA driver (`nvcuda.dll`) ships with the display driver and needs
nothing here.
