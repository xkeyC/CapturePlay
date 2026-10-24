# CapturePlay

低延迟采集卡播放器：**PS5 / 游戏机 → HDMI 采集卡 → PC → 实时补帧 → 原生 HDR 输出**。
Rust + Direct3D 12，单进程、单交换链，Windows 10/11。

> 技术预览（0.1.0）。

---

## 一、功能介绍

### 1.1 概述

CapturePlay 把采集卡当作显示器：主机画面在 PC 上以尽可能低的延迟呈现，并按显示器刷新率补帧、上采样和输出 HDR。它不录制、不转码，整条链路只有一个进程和一条交换链。

### 1.2 功能

| 功能 | 说明 |
|---|---|
| 低延迟预览 | 到达→呈现约 15 ms（30→60 内容，RTX 4090） |
| AI 补帧 | NVIDIA VFG，30/40/50/60 fps 内容补到显示器刷新率，2×/3×/4× |
| 原生 HDR | 固定 FP16 scRGB 交换链，全屏/窗口切换保持 HDR |
| AI 超分 | NVIDIA RTX VSR，1440p→4K，可回退 Catmull-Rom |
| NR 画面增强 | DLSS Neural Rendering，3 种风格与多项参数，实验功能 |
| 可变刷新 | G-SYNC / FreeSync / HDMI VRR，自动识别固定刷新与 VRR |
| 音频直通 | 采集卡 HDMI 音频经 WASAPI 输出，带时钟漂移校正 |
| 界面 | 中文/英文，HDR 后缓冲内绘制，可显示 GPU 耗时与呈现抖动 |

### 1.3 使用

| 操作 | 快捷键 |
|---|---|
| 显示/隐藏面板 | F1 / Tab |
| 全屏 | F11 / 双击 |
| 退出全屏 | Esc |
| 定格插值帧 | P |

鼠标在画面上静止 2 秒后自动隐藏（悬停在面板上时除外）。

设置保存在 `%APPDATA%\CapturePlay\settings.toml`，日志为同目录的 `capture-play.log`。`CAPTUREPLAY_LOG=debug` 提高日志级别，`CAPTUREPLAY_D3D_DEBUG=1` 启用 D3D12 调试层。

NR 面板：后端、模型（Auto / Quality / Performance）、风格（默认 / 自然 / 电影）、模型强度、合成强度；“更多参数”中可调局部结构、局部影调、肤色结构、自动蒙版与 UI 校正。

### 1.4 性能参考

RTX 4090，AVerMedia Live Gamer Ultra 2.1，1440p P010，release 构建：

| 补帧模型 | 每帧 GPU | 到达→呈现 |
|---|---|---|
| 快速 | 0.95 ms | ≈15.3 ms |
| 均衡（默认） | 1.80 ms | ≈15.4 ms |
| 高质量 | 8.29 ms | ≈16.0 ms |

| 项目 | GPU 时间 | 说明 |
|---|---|---|
| RTX VSR Medium / High / Ultra | 1.35 / 3.61 / 4.94 ms | High 画质最佳，Ultra 在干净游戏画面上收益很小 |
| NR（1440p） | ≈5.0 ms / 唯一帧 | 社区 310.9.16，强度 1.0，默认风格 |

以上为特定机器与素材的参考值，实际取决于显卡、采集格式和效果组合。

### 1.5 采集卡与硬件

以 AVerMedia Live Gamer Ultra 2.1 为例：

| 模式 | 建议 |
|---|---|
| **1440p60 P010** | HDR 首选，去重准确，USB 带宽压力小 |
| 4K60 NV12 | 8-bit，HDR 会转成 SDR |
| 4K30 P010 | 不推荐，60 Hz 信号被隔帧采样，可能出现重复帧与顿挫 |

该卡（USB 版）不需要设置硬件色调映射；PCIe 型号会自动关闭。

| 功能 | 要求 |
|---|---|
| 基础播放 | Windows 10/11 x64，D3D12 显卡 |
| VFG 补帧 | NVIDIA RTX 40/50 + Video Effects 运行时 |
| VSR 超分 | NVIDIA RTX 20+ + RTX Video SDK |
| NR 官方（RTX50） | 驱动/NGX 暴露 feature 18 |
| NR 社区（Unofficial RTX40/30/20） | 自备社区 `nvngx_dlssnr.dll` + 本项目生成的 `captureplay_nvngx.dll` |

### 1.6 构建与运行

需求：Rust 1.95+（MSVC）、Windows SDK（`dxc.exe`，可用 `DXC_PATH` 指定）、D3D12 显卡。

~~~powershell
cargo build --release
.\target\release\capture-play.exe
~~~

可选组件放在 `runtime/`（git 忽略）：
- `runtime/nvidia-vfx/`：VFG 运行时；
- `runtime/rtx-video-sdk/`：RTX Video SDK（构建时链接 `nvsdk_ngx_d.lib`，运行时加载 `nvngx_vsr.dll`）；
- `runtime/nvidia-nr/`：社区 `nvngx_dlssnr.dll` 与 `captureplay_nvngx.dll`。

`cargo build` 会同时生成 `captureplay_nvngx.dll`。首次启用 VSR/NR 时模型创建约 2–3 秒，期间画面直通。

### 1.7 注意事项

- HDR 需要显示器与 Windows HDR 同时开启；切换全屏或窗口后交换链会重新应用输出合约。
- NR 为实验功能；社区运行库未签名、可能崩溃，仅用于学习交流。官方路径取决于 NVIDIA 是否在驱动中开放 feature 18。
- 若 UI 报告“固定刷新”而显示器支持 VRR，先关闭 NVIDIA 或第三方叠加层再测。

---

## 二、技术实现

### 2.1 架构与线程

~~~
 采集线程 (MF 异步)                   渲染线程 (D3D12 + egui)                 主线程 (winit)
 ───────────────────                 ─────────────────────────────          ─────────────
 IMFSourceReader ──memcpy──▶ UploadRing ──▶ 去重(Cadence) ──▶ ConvertPass(YUV→线性 scRGB)
   (一次拷贝, 指纹)        (持久映射上传堆)                         │
                                                               NR (可选)
                                                                     │
                                 Pacer(事件驱动) ──▶ Interpolator(VFG) ──▶ OutputPass
                                                                    (缩放+色调映射)
                                                                          │
 WASAPI 采集 ──SPSC 环形缓冲──▶ WASAPI 输出                       EguiPass ──▶ FP16 scRGB 交换链
   (漂移校正 ±0.5%)                                                 (UI 画在同一后缓冲)
~~~

| crate | 职责 |
|---|---|
| `cp-core` | 色彩数学、格式选择、去重/节奏、设置 |
| `cp-capture` | Media Foundation 采集：设备、格式、异步读取、色彩识别 |
| `cp-audio` | WASAPI 音频直通、端点匹配、时钟漂移校正 |
| `cp-render` | D3D12：上传、转换、NR、补帧、超分、输出、交换链/HDR、egui |
| `cp-nvidia` | CUDA driver 子集、Video Effects (VFG)、NGX（VSR/NR） |
| `nr-bridge` | 社区 `nvngx_dlssnr.dll` 的转发宿主 `captureplay_nvngx.dll` |
| `capture-play` | 窗口、线程编排、UI、设置持久化 |

主线程只跑窗口消息；渲染线程拥有设备、交换链、采集会话与 UI。渲染线程按节奏器给出的截止时间休眠/唤醒，窗口拖动或缩放不会阻塞画面。

### 2.2 采集与色彩

- Media Foundation `IMFSourceReader` 异步模式，`MF_LOW_LATENCY`，禁用格式转换器，驱动队列深度 2。
- 每帧一次拷贝：驱动缓冲 → 持久映射的 D3D12 上传堆 → GPU 拷贝到平面纹理；拷贝时采样 64×36 亮度指纹。
- 渲染来不及时回收最旧的未消费帧，延迟不累积。
- 色彩空间为线性 scRGB（BT.709 原色，1.0 = 80 nit，FP16）。
- YUV→RGB 矩阵在 CPU 端按矩阵、量化范围、位深合成；PQ/HLG/SDR 分别解码为绝对亮度或参考白。
- 色彩识别优先级：用户覆盖 > 驱动 HDR 标记 > 10-bit 启发式 > 驱动标记 > 分辨率默认。

### 2.3 去重与重定时

- 主机 30 fps 游戏在 60 Hz 输出中每帧出现两次；在重复帧之间插值没有意义。`CadenceTracker` 用指纹差判断唯一帧，并估计内容帧间隔。
- 采集卡会把 VRR 输入量化成固定 60 Hz，唯一帧间隔呈 16.7/33.3 ms 交替。`Retimer` 在最近 24 帧上取满足量化约束的最晚直线，重建均匀的内容时间；锁帧内容不受影响，可变帧率平均多等约 5 ms。

### 2.4 节奏与呈现

- `Pacer` 按下一帧的内容时间排程 `M-1` 个中间帧，附加延迟为 `(M-1)/M × 源帧间隔`：30→60 为半帧 16.7 ms，是插值的理论下限；默认 2×，可选 3×/4×。
- 超过内容间隔 1.75 倍的空档不插值，直接显示真实帧。
- 首个槽位 = 内容时间 + 偏移；偏移由近期处理预算的 p95 与合成提前量决定，避免到达抖动传到屏幕。
- 交换链使用 FLIP_DISCARD、3 缓冲、`SetMaximumFrameLatency(1)` 与可等待对象；关闭垂直同步时允许 tearing。
- VRR 下精确在目标时刻 Present；固定刷新下对齐 vblank 并使用整数刷新周期倍数。`GetFrameStatistics` 用于区分两者并统计呈现抖动。
- 超分在插帧之后、对每个呈现帧执行，与输出同队列。

### 2.5 HDR 输出

- 交换链固定为 `R16G16B16A16_FLOAT` + `G10_NONE_P709`，跨显示器或切换 Windows HDR 不重建。
- HDR 显示器按 BT.2390 EETF 在 PQ 域做保色相色调映射；SDR 显示器映射到 SDR 白。
- UI 与视频画在同一后缓冲，按 Windows「SDR 内容亮度」的参考白绘制。
- 每次 `ResizeBuffers` 后重新应用色彩空间并清除 HDR10 元数据，全屏/窗口切换不丢失 HDR。

### 2.6 补帧

- 帧编码进 D3D12 与 CUDA 共享缓冲区（RGB10A2；HDR 用 PQ/BT.2020，SDR 用 sRGB），`NvVFX_Run` 以 `Timestep` 模式生成任意 t 的中间帧。
- D3D12 Signal → CUDA Wait → Run → CUDA Signal → D3D12 Wait，全程无 CPU 拷贝与等待。
- 补帧后端通过 `Interpolator` trait 接入。

### 2.7 超分

- `cp_nvidia::ngx` 手写 NGX FFI，进程内初始化一次并串行化。
- `NgxVsrUpscaler` 将 FP16 scRGB 编码为 PQ BT.2020 10-bit，调用 VSR，再解码回工作帧；全部录在帧自己的命令列表里，与补帧、输出同队列。
- 模型在后台线程创建，尺寸与档位是每次调用的参数；超分后置，对每个呈现帧执行。
- 目标尺寸按显示区域选 VSR 支持的最小覆盖倍数；窗口变化去抖。VSR 不可用时回退 PQ 感知域 Catmull-Rom。

### 2.8 NR

- 顺序：转换 → NR（每个唯一源帧一次）→ 补帧 → 超分 → 输出。NR 先于补帧与超分，插值帧和超分帧都基于增强后的画面。
- 数据流：线性帧编码为 sRGB 代理（HDR 使用可逆高光滚降）→ NGX 同分辨率推理（Color / 常量 Depth / 零 MVec / Output）→ 把结果作为有界线性残差叠回原帧。
- 风格 0/1/2；模型强度、局部结构、局部影调、肤色结构、自动蒙版、UI 校正为逐次评估参数，调整即时生效。模型预设切换会重建特性。
- 官方路径（RTX50）：直接创建 NGX feature 18，驱动未开放时显示不可用并直通。
- 社区路径（Unofficial RTX40/30/20）：`captureplay_nvngx.dll` 加载自备的 `nvngx_dlssnr.dll` 并转发 NGX 入口。社区 DLL 未签名、不受支持，仅用于学习交流。
- 部署：`captureplay_nvngx.dll` 放 exe 同目录或 `runtime/nvidia-nr/`，`nvngx_dlssnr.dll` 放 `runtime/nvidia-nr/`，在 UI 选择社区后端。替换 DLL 后需重启。

### 2.9 音频

采集卡 HDMI 音频作为 WASAPI 采集端点，按产品名与视频设备自动配对。采集与输出两条事件驱动线程通过无锁 SPSC 环形缓冲连接；输出端交给 Windows 转换（`AUTOCONVERTPCM`）。两个设备时钟的漂移由 ±0.5% 的线性重采样吸收，比例由缓冲水位控制，默认 30 ms 缓冲，可调音画偏移。

---

## 三、授权与 SDK

### 3.1 项目许可

本项目以 **GPL-3.0-or-later** 发布，见 `LICENSE`。附加许可见 `LICENSE-EXCEPTION.md`：允许与 NVIDIA 专有组件（RTX Video SDK / NGX、Video Effects SDK、CUDA）链接或组合后分发。这些 SDK 不属于本项目、不受 GPL 约束，按 NVIDIA 各自的许可使用。

### 3.2 NVIDIA SDK 与运行库

| 组件 | 用途 | 获取 | 分发 |
|---|---|---|---|
| `nvcuda.dll` | CUDA driver（VFG 互操作） | 随 NVIDIA 驱动 | 不随本项目分发 |
| Video Effects 运行时（`runtime/nvidia-vfx/`） | VFG 补帧 | 用户从 NVIDIA 获取 | 随应用以目标代码分发，保留通知 |
| RTX Video SDK（`runtime/rtx-video-sdk/`） | VSR；构建时链接 `nvsdk_ngx_d.lib` | 用户从 NVIDIA 开发者站点获取 | 随应用分发 `nvngx_vsr.dll`，保留通知；商业发布前通知 NVIDIA |
| NGX feature 18（NR） | DLSS Neural Rendering | 官方或社区运行库，见 3.3 | 本项目不分发 `nvngx_dlssnr.dll` |

找不到 RTX Video SDK 时 `cfg(ngx)` 关闭，应用仍可构建，只是没有 VSR/NR 的 NGX 后端。

### 3.3 NR 社区运行库

- `captureplay_nvngx.dll` 是 `crates/nr-bridge` 编译出的转发宿主，GPL 代码，不含 NVIDIA 代码。
- `nvngx_dlssnr.dll` 是社区修改、未签名的二进制，仓库不包含、不分发；用户自行放入 `runtime/nvidia-nr/`。
- 社区 snippet 会校验调用方模块名，宿主仅用于满足该校验。这不是 NVIDIA 支持的使用方式，可能违反其许可条款；仅用于学习交流，不要作为稳定或商业特性依赖。

### 3.4 runtime 目录

`runtime/` 被 git 忽略（只保留 `runtime/README.md`）：

| 路径 | 内容 | 环境变量 |
|---|---|---|
| `runtime/rtx-video-sdk/` | RTX Video SDK | 构建 `CAPTUREPLAY_RTX_VIDEO_SDK`；运行 `CAPTUREPLAY_RTX_VIDEO_RUNTIME` |
| `runtime/nvidia-vfx/` | Video Effects 运行时 | `CAPTUREPLAY_VFX_RUNTIME` |
| `runtime/nvidia-nr/` | 社区 `nvngx_dlssnr.dll` + `captureplay_nvngx.dll` | `CAPTUREPLAY_NR_RUNTIME`、`CAPTUREPLAY_NR_BRIDGE` |

### 3.5 第三方归属

- NR 的代理/残差颜色数学遵循 MIT 许可（原实现：video2dlssnr，© 2026 Daniil Sokolyuk），完整许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- CUDA 的 FFI 声明依据 NVIDIA 公开的头文件与文档手写；仓库不包含任何 NVIDIA 头文件、库或运行时。
