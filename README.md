# CapturePlay

低延迟采集卡播放器：**PS5 / 游戏机 → HDMI 采集卡 → PC → 实时补帧 → 原生 HDR 输出**。
Rust + Direct3D 12，单进程、单交换链，Windows 10/11。

> 状态：技术预览（0.1.0）。核心链路已在真机上验证；本文档同时作为技术报告。

---

## 1. 目标与结论

| 目标 | 做法 | 实测（RTX 4090，AVerMedia LGU 2.1，PS5 30 fps 游戏） |
|---|---|---|
| 低延迟 | 事件驱动节奏、单次拷贝采集、可等待交换链 MaxFrameLatency=1 | 到达→呈现 ≈ **15.3 ms**（30→60 理论下限 16.7 ms 的半帧） |
| 高效补帧 | 先去重找回真实 30 fps，再用 NVIDIA VFG（AI）插值 | 每帧插值 GPU：VFG 均衡 **1.80 ms**（1440p） |
| 原生 HDR | 固定 FP16 scRGB 交换链 + 每次 Resize 后重设输出合约 | 全屏/窗口反复切换 HDR 不丢失 |
| 工程治理 | 6 个职责单一的 crate，纯逻辑全部单元测试，clippy 零警告 | 42 个单元测试 |
| 可选增强 | `Enhancer` 扩展点（色彩转换后、补帧前） | 预留给官方 SDK（如 DLSS 神经渲染公开后） |

## 2. 架构

```
 采集线程 (MF 异步)                   渲染线程 (D3D12 + egui)                 主线程 (winit)
 ───────────────────                 ─────────────────────────────          ─────────────
 IMFSourceReader ──memcpy──▶ UploadRing ──▶ 去重(Cadence) ──▶ ConvertPass(YUV→线性 scRGB)
   (一次拷贝, 指纹)        (持久映射上传堆)                         │
                                                              Enhancer (预留)
                                                                    │
                                 Pacer(事件驱动) ──▶ Interpolator(VFG) ──▶ OutputPass
                                                                    (缩放+色调映射)
                                                                          │
 WASAPI 采集 ──SPSC 环形缓冲──▶ WASAPI 输出                       EguiPass ──▶ FP16 scRGB 交换链
   (漂移校正 ±0.5%)                                                 (UI 画在同一后缓冲)
```

| crate | 职责 |
|---|---|
| `cp-core` | 纯逻辑：色彩数学、格式选择、去重/节奏、设置。无 OS 依赖，着色器数学的参考实现与测试 |
| `cp-capture` | Media Foundation 采集：设备枚举、格式选择、异步读取、色彩识别、厂商控制 |
| `cp-audio` | WASAPI 音频直通、采集卡端点匹配、时钟漂移校正 |
| `cp-render` | D3D12：上传、转换、补帧后端、输出映射、交换链/HDR、egui 渲染、GPU 计时 |
| `cp-nvidia` | NVIDIA 运行时的动态绑定：CUDA driver 子集、Video Effects (VFG) |
| `capture-play` | 应用：窗口、线程编排、UI（中/英）、设置持久化 |

线程模型：主线程只跑窗口消息；渲染线程拥有设备、交换链、采集会话与 UI。窗口拖动/缩放的模态循环不会阻塞画面；渲染线程在一个事件 + 高精度可等待定时器上休眠，按节奏器给出的截止时间唤醒（普通超时有 15.6 ms 系统时钟粒度，实测会丢帧）。

## 3. 低延迟设计

### 3.1 采集
- Media Foundation `IMFSourceReader` 异步模式，`MF_LOW_LATENCY`、禁用格式转换器、驱动队列深度 2。
- 每帧只做**一次拷贝**：驱动缓冲 → 持久映射的 D3D12 上传堆（行距按 D3D12 对齐），随后 GPU `CopyTextureRegion` 到平面纹理。
- 拷贝时顺便采样 64×36 亮度指纹用于去重，几微秒。
- 渲染端来不及时，采集端回收**最旧**的未消费帧，而不是丢弃最新帧，延迟不会累积。
- 停止采集不依赖驱动返回（异步回调 + 50 ms 轮询），无信号时也能干净退出。

### 3.2 去重：找回游戏真实帧率
主机 30 fps 游戏仍以 60 Hz HDMI 输出，每个游戏帧出现两次（或不均匀的 3:1）。在重复帧之间插值没有意义，并会让补帧节奏错乱。`CadenceTracker` 用指纹差（均值 + 变化像素比例，容忍采集噪声）判断唯一帧，并用中位数估计内容帧间隔；长时间静止（菜单）不污染估计。

### 3.3 事件驱动节奏（插值的延迟下限）
插值 A、B 之间的帧必须等 B 到达。`Pacer` 在 B 到达的瞬间提交 `M-1` 个中间帧，间隔 `interval/M`，B 最后显示：

- 附加延迟 = `(M-1)/M × 源帧间隔`。30→60（M=2）为 **半帧 16.7 ms**，这是插值的理论下限。
- 更高倍数延迟递增（4× 为 25 ms），因此默认 2×，UI 可选 3×/4×。
- 超过内容间隔 1.75 倍的空档（加载、卡顿）不插值，直接显示真实帧。
- 引擎外接的 DLSS-FG 一类方案通常在此之上还要多排队一帧以上，这是本项目不采用它的原因之一。

### 3.4 呈现
FLIP_DISCARD、3 缓冲、`SetMaximumFrameLatency(1)`、可等待对象在录制前尽量晚地等待；关闭垂直同步时允许 tearing（配合 VRR 进一步降延迟）。

## 4. HDR 方案

**工作空间**：线性 scRGB（BT.709 原色，1.0 = 80 nit，FP16，可存负值即广色域）。

**输入解码**（`convert.hlsl`，数学在 `cp-core::color` 中有单元测试）：
- YUV→RGB 仿射矩阵在 CPU 端按矩阵/量化范围/位深/存储位宽合成（P010 的 10-bit 码值位于 16-bit 高位）。
- PQ → 绝对亮度；HLG → 逆 OETF + 按显示器峰值的 BT.2100 OOTF；SDR → sRGB/γ2.2/BT.1886 × SDR 参考白。
- BT.2020 → BT.709 矩阵转换（保留负值）。

**色彩识别**：优先级为 用户覆盖 > 驱动显式 HDR 标记 > 10-bit 启发式 > 驱动标记 > 分辨率默认。
采集卡普遍把 HDR 的 P010 错标为 BT.709（LGU 2.1 即如此）；主机只有在 HDR 开启时才输出 10-bit，因此 10-bit 无标记默认按 HDR10 处理（可关闭）。

**输出**：交换链**固定**为 `R16G16B16A16_FLOAT` + `G10_NONE_P709`。DWM 在 SDR 与 HDR 显示器上都能正确合成，跨显示器移动或切换 Windows HDR 时只改输出着色器常量，不重建交换链：
- HDR 显示器：按 BT.2390 EETF 在 PQ 域做保色相的色调映射到显示器峰值（`IDXGIOutput6::GetDesc1`）。
- SDR 显示器：映射到 SDR 白并按 SDR 白归一。
- SDR 内容放在 Windows「SDR 内容亮度」对应的参考白（`DISPLAYCONFIG_SDR_WHITE_LEVEL`），UI 同样按此白点绘制。

**参考项目“全屏后 HDR 丢失”的根因与修复**：参考实现只在创建交换链时调用一次 `SetColorSpace1`，全屏触发的 `ResizeBuffers` 之后从未重设；它的视频窗口还是嵌入 Qt 的子 HWND、跨线程呈现。本项目：
1. `Presenter::apply_output_contract()` 幂等，在创建与**每次** `ResizeBuffers` 后执行（色彩空间 + 清除 HDR10 元数据）。
2. 全屏只用无边框（禁止 DXGI 独占与 Alt+Enter）。
3. 显示器信息按「窗口重叠面积最大的输出」查询；`IDXGIFactory::IsCurrent()` 失效、窗口移动/缩放时重新查询；查询失败保留上次结果，绝不误降级到 SDR。
4. UI 与视频在同一后缓冲，不存在子窗口/空域问题。

已实测：4K 全屏 ↔ 窗口多次切换，每次 Resize 后都重新应用输出合约，显示器保持 HDR，补帧节奏不中断。

## 5. 补帧：NVIDIA Video Frame Generation（`interp_vfg`，RTX 40/50，AI 模型）

- 帧直接编码进 D3D12 与 CUDA 共享的缓冲区（`D3D12_HEAP_FLAG_SHARED` → `cuImportExternalMemory`），RGB10A2：HDR 内容用 PQ/BT.2020，SDR 内容用 sRGB。
- 同步：D3D12 队列 Signal(producer) → CUDA 流 Wait → `NvVFX_Run` → CUDA Signal(completion) → D3D12 队列 Wait。全程无 CPU 拷贝、无 CPU 等待。
- 使用 `Timestep` 模式，任意 t 都可生成，与节奏器的时间模型一致。
- 关闭 SDK 自带镜头切换检测（会让每次 Run 产生 CPU 同步）。
- 运行时 DLL 不在仓库中，见 `runtime/README.md`。
- 插值后端通过 `Interpolator` trait 接入，可再扩展。

### 5.1 性能（30→60，同一游戏画面，release，RTX 4090）

| 模型 | 1440p P010 每帧 GPU | 4K NV12 每帧 GPU | 到达→呈现 |
|---|---|---|---|
| 快速 | 0.95 ms | — | ≈15.3 ms |
| 均衡（默认） | 1.80 ms | 2.12 ms | ≈15.4 ms |
| 高质量 | 8.29 ms | — | ≈16.0 ms |

GPU 时间用时间戳查询测量，包含 CUDA 跨引擎部分。UI 提供 **定格插值帧**（P 键）用于检查伪影。

### 5.2 画质：真值测试与“为什么不用 NVIDIA Optical Flow”

`cargo test -p cp-render -- --ignored --nocapture` 运行 GPU 真值测试：用可在连续坐标上求值的合成纹理生成 A/B 两帧（运动精确已知），把插值帧与 t=0.5 的真值比较（PSNR，越高越好），重点看**遮挡区**（运动物体边缘，重影出现的地方）。

早期版本另有一个 NVIDIA Optical Flow（NVOFA 硬件光流 + 自写 warp）后端。真值测试表明光流本身精确（已知平移 (40,−20) 测得 (40.000, −20.000)），但块匹配光流在物体边界天然模糊：

| 方案（1440p，物体移动 (60,20)） | 物体边缘带 PSNR |
|---|---|
| 两帧直接混合 | 16.4 dB |
| 光流 + 反向 warp | 18.9 dB |
| 光流 + 前向投影（遮挡感知） | 22.3 dB |
| VFG 快速 / 均衡 / 高质量 | 26.1 / 27.8 / 32.6 dB |

即使改为遮挡感知的前向投影，光流方案在边缘仍比 VFG 均衡低约 6 dB，GPU 耗时却相近，因此被移除。测试也定位并修复了它的一个崩溃：注册给 NVOFA 的资源在 `nvOFDestroy` 之后释放会访问冲突（运行中切换后端时闪退）。

## 6. 音频

采集卡 HDMI 音频是普通 WASAPI 采集端点，按产品名与视频设备自动配对；不会回退到无关输入（避免误播麦克风）。两条事件驱动线程（MMCSS Pro Audio）经无锁 SPSC 环形缓冲连接；输出端以采集格式初始化，交给 Windows 转换（`AUTOCONVERTPCM`）。两个设备时钟的漂移由 ±0.5% 的线性重采样吸收，比例由缓冲水位的慢速控制器决定（有仿真测试），延迟目标默认 30 ms，可调音画偏移。

## 7. 采集卡建议（以 AVerMedia Live Gamer Ultra 2.1 为例）

| 模式 | 结论 |
|---|---|
| **1440p60 P010** | HDR 首选。完整看到 60 Hz 信号，去重准确；每帧约 11 MB，USB 传输短 |
| 4K60 NV12 | 8-bit，HDR 会被转成 SDR |
| 4K30 P010 | 不推荐：60 Hz 信号被隔帧采样，可能取到重复帧（实测偶发唯一帧 28.4/s → 顿挫），也可能取到较晚的那份副本；每帧约 25 MB，USB 传输更久 |

该卡（USB 版）不支持 AVerMedia 硬件色调映射属性集，HDR 直通无需设置；PCIe 型号会自动关闭硬件色调映射。

## 8. 构建与运行

需求：Windows 10/11 x64，Rust 1.95+（MSVC 工具链），Windows SDK（`dxc.exe`，构建时编译 HLSL 为签名 DXIL；可用 `DXC_PATH` 指定），D3D12 显卡；补帧需要 NVIDIA RTX 40/50 显卡与 VFG 运行时。

```powershell
cargo build --release
.\target\release\capture-play.exe
```

可选 VFG 运行时放到 `runtime/nvidia-vfg/`（见 `runtime/README.md`）。

| 操作 | 快捷键 |
|---|---|
| 显示/隐藏面板 | F1 / Tab |
| 全屏 | F11 / 双击 |
| 退出全屏 | Esc |
| 定格插值帧 | P |

鼠标在画面上静止 2 秒后自动隐藏（悬停在面板上时不隐藏），移动即恢复。

设置保存在 `%APPDATA%\CapturePlay\settings.toml`，日志在同目录 `capture-play.log`（`CAPTUREPLAY_LOG=debug` 提高级别，`CAPTUREPLAY_D3D_DEBUG=1` 启用 D3D12 调试层）。UI 支持中文/英文，默认跟随系统语言。

## 9. 相对参考项目（Veyra-NRVideo）的取舍

| 方面 | 参考项目 | 本项目 |
|---|---|---|
| 规模/结构 | 约 4 万行 C++，多个 3000–9000 行的单文件 | 约 1.1 万行 Rust + 500 行 HLSL，6 个职责单一的 crate |
| UI | Qt QML + 嵌入的原生子窗口（空域、跨线程呈现） | egui 直接画进 HDR 后缓冲 |
| 采集 | DirectShow + 自写 COM 滤镜 | Media Foundation 异步读取 |
| 节奏 | 固定前瞻的时间线 | 事件驱动，半帧延迟下限，去重后插值 |
| HDR | 色彩空间只设一次 → 全屏后丢失 | 固定 scRGB + 每次 Resize 重设，显示器状态只升不误降 |
| 补帧 | 多后端，含针对签名 DLL 的内存补丁/架构伪装 | 只用官方、可正常加载的运行时（NVIDIA VFG），画质用真值测试量化 |
| DLSS 5 NR | 依赖未公开功能与对 NVIDIA 访问限制的规避 | 不实现；保留 `Enhancer` 接口等待官方 SDK |

## 10. 已知限制与后续

- 画质对比需在 HDR 显示器上主观评估；尚无客观指标（可用 60 fps 内容做“隔帧插值 vs 真实帧”的 PSNR/SSIM 评估）。
- 端到端延迟（含采集卡与显示器）需要光电/高速相机测量；软件内只统计到达→呈现。
- 尚未实现：设备热插拔通知（目前定时重试）、设备丢失后自动重建、HDR 元数据（InfoFrame）读取、音频设备选择 UI、CI。
- 镜头切换检测：目前依赖空档判断，场景硬切会生成一帧混合帧。

## 11. 许可证

本项目以 **GPL-3.0-or-later** 发布，见 [LICENSE](LICENSE)。

仓库**不包含**任何专有 SDK、头文件或运行时。运行时按需动态加载：
- 随 NVIDIA 显卡驱动提供：`nvcuda.dll`（CUDA driver）。
- 用户自行安装：NVIDIA Video Effects SDK 运行时（VFG），见 `runtime/README.md`，受 NVIDIA 许可约束。

CUDA 的 FFI 声明依据 NVIDIA 公开的头文件与文档手写。分发捆绑专有运行时的二进制包前，请自行评估 GPL 兼容性（例如为这些运行时添加 GPLv3 第 7 节的附加许可）。
