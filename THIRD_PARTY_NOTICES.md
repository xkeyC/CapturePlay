# Third-party notices

CapturePlay itself is GPL-3.0-or-later (see `LICENSE`). The proprietary NVIDIA
SDKs and runtimes it links against are not part of the project and stay under
NVIDIA's own terms (see `LICENSE-EXCEPTION.md` and `runtime/README.md`).

## video2dlssnr

The display-referred proxy and bounded-residual composite in
`crates/cp-render/shaders/nr.hlsl` follow the color contract described by
video2dlssnr by Daniil Sokolyuk
(<https://github.com/DaniilSokolyuk/video2dlssnr>), which is MIT licensed:

```
MIT License

Copyright (c) 2026 Daniil Sokolyuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The NGX feature-18 parameter names and input contract were recovered by that
project and by the community; they are not part of a public NVIDIA SDK. DLSS
Neural Rendering is therefore a community-modified, unofficial use of an
undocumented NGX feature: it is included for learning and experimentation
only, not as an NVIDIA-supported capability.

`crates/nr-bridge` (shipped as `captureplay_nvngx.dll`) is CapturePlay's own
code and contains no NVIDIA code: it only forwards the NGX entry points to a
user-supplied snippet so community builds accept the caller.
