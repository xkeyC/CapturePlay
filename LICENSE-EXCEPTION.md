# Additional permission (GNU GPL version 3, section 7)

CapturePlay is licensed under the GNU General Public License, version 3 or
(at your option) any later version — see [LICENSE](LICENSE) — with the
following additional permission.

If you modify this Program, or any covered work, by linking or combining it
with any of the following NVIDIA proprietary components (or a modified
version of them), containing parts covered by the terms of their respective
NVIDIA licenses, the licensors of this Program grant you additional
permission to convey the resulting work:

- the NVIDIA RTX Video SDK, including its NGX SDK libraries (such as
  `nvsdk_ngx_*.lib`) and feature runtimes (such as `nvngx_vsr.dll` and
  `nvngx_truehdr.dll`);
- the NVIDIA Video Effects (VFX) SDK runtimes;
- the NVIDIA CUDA runtime and the CUDA driver API (`nvcuda.dll`);
- other libraries shipped with the NVIDIA display driver.

Corresponding Source for a non-source form of such a combination shall not
include the source code for the parts of those NVIDIA components used as
well as that of the covered work.

These NVIDIA components are **not** part of this Program and are **not**
covered by the GPL; they remain under NVIDIA's own license terms, which you
must accept and comply with separately. This repository does not contain any
of them (no SDK headers, libraries or runtimes): they are obtained by the
user from NVIDIA and placed under the git-ignored `runtime/` folder.
