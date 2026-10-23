# runtime/

Optional third-party runtimes, installed locally and ignored by git.

| Folder | Contents | Used by |
|---|---|---|
| `nvidia-vfg/` | `NVVideoEffects.dll`, `NVCVImage.dll`, `nvVFXVideoFrameGeneration.dll`, `nvngxruntime.dll`, `cudart64_12.dll` from NVIDIA's Video Effects SDK (`nvidia-vfx` package) | AI frame interpolation backend "NVIDIA VFG" (RTX 40/50) |

Lookup order for the VFG runtime: `interpolation.vfg_runtime_dir` in
`settings.toml` → `CAPTUREPLAY_VFG_RUNTIME` → `runtime/nvidia-vfg` next to the
executable or in any parent directory of it (so `target/release` builds find
this folder).

NVIDIA Optical Flow (`nvofapi64.dll`) and CUDA (`nvcuda.dll`) ship with the
display driver and need nothing here.
