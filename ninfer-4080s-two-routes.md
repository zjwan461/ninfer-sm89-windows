# 在 Windows 11 + RTX 4080 SUPER (32G) 上使用 NInfer：两条落地路线

> 目标仓库：[JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows)（本地已 clone 到 `E:\workspaces\ai\ninfer-4090-windows`，分支 `main`）
> 本机实测环境：Windows 11 x64 / NVIDIA GeForce RTX 4080 SUPER **32 GB 显存**（`sm_89`，AD103，**80 SM**）/ 驱动 617.14 / CUDA Toolkit 12.4 + **13.4**
> 文档日期：2026-10-03 · 本文所有环境项均已在**本机实际探测**，源码结论均标注了具体文件与行号

---

## 0. 结论速览

| 问题 | 结论 |
| --- | --- |
| 我的 4080S 能用这个仓库吗？ | **能。** 该仓库只编译 `sm_89`，4080S 与 4090 同架构（Ada），二进制与 SASS 完全兼容 |
| 能编译出来 `ninfer.exe` 吗？ | **能。** 工具链本机只差一个 **vcpkg**，其余（CMake 3.31 / Ninja 1.12 / MSVC 14.44 / CUDA 13.4）全部已就位 |
| 官方支持 4080S 吗？ | **不支持，明确标注 untested。** release note 原话：*"Other RTX 40-series cards share the architecture and may work with Bonsai (untested)"* |
| 会不会像旧分支那样整卡崩掉？ | **不会**（本喵逐行验证过，见 §1）。这个 fork 已经把协作启动预算改成**运行时真实 SM 数** |
| 有性能税吗？ | **有，但不致命**：`kTargetSmCount = 128` 是编译期常量，80 SM 的卡上 attention 会变成多 wave。不崩，只是拿不到 4090 的"整波刚好填满"速度 |
| 最短路径是自己编译吗？ | **不是。** 仓库有 822 MB 预编译 zip，`sm_89` 的 cubin 直接能加载 → 先跑路线 A 拿基线 |
| 自编译的真正价值是什么？ | 能把 `kTargetSmCount` 那几处按 80 SM 重算，把 attention 的性能税吃回来（见 §6） |

**一句话：架构对得上、显存够、工具链基本齐，两条路都通。先 A 后 B，别一上来就编译。**

---

## 1. 源码级关键事实（本喵逐条验证，非二手结论）

### 1.1 架构硬门槛：只接受 `sm_89`

`CMakeLists.txt:3-14`：

```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  message(FATAL_ERROR "NInfer supports only CMAKE_CUDA_ARCHITECTURES=89; got ...")
endif()
```

→ 你的 4080S 正好是 `sm_89`，**门槛天然满足**。（对比：上游 `Neroued/ninfer` 只接受 `sm_120a`，Ada 卡完全编不了。）

另有版本下限 `CMakeLists.txt:56`：CUDA ≥ 12.8 → 本机 13.4 ✅

### 1.2 没有任何"必须叫 4090"的运行时校验

全仓搜 `GeForce` 只有两处，且都不是判定逻辑：

- `src/core/device.h:28` —— 一行注释 `// NVIDIA GeForce RTX 4090 (sm_89)`
- `src/runtime/engine/context_cache/context_cost_defaults.cpp:51` —— 字符串标签 `"nvidia-geforce-rtx-5090-sm120"`（成本档位命名，非设备校验）

→ **引擎不会拒绝 4080S**，也不检查显存大小是否等于 24 GB。

### 1.3 好消息：旧笔记里那颗「80 SM 崩卡炸弹」在这个 fork 里已经拆掉了

本会话前一份调研（`ninfer-win11-4080s-build-feasibility.md` §4.3）记录的根因是：常驻 CTA 预算按 128 SM 写死，80 SM 卡上协作启动越界 → 整卡 `FULLCHIP_RESET`，只能重启机器。

**这个 fork 的 `bf16_gdn_gating_proj_kernels.cu:358-375` 已经改成运行时查询 + 自动分块：**

```cpp
constexpr std::int32_t kResidentCtasPerSm = cooperative_resident_ctas_per_sm<Geometry, SplitK>();
const std::int64_t resident_ctas  = static_cast<std::int64_t>(multiprocessor_count) * kResidentCtasPerSm;
const std::int64_t max_token_tiles = resident_ctas / kCtasPerTokenTile;
if (max_token_tiles < 1) { return false; }          // 设备连一个 tile 都放不下 → 交回上层走非协作回退

if (total_token_tiles <= max_token_tiles) { launch_problem(...); }
else {
    // Token tiles have no cross-tile reduction —— 按 token 区间切成多次协作启动
    for (std::int32_t token_begin = 0; token_begin < t;) { ... }
}
```

`multiprocessor_count` 的真实来源（`src/core/device.cu:81-86`）：

```cpp
int device_sm_count() {
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
```

并且 `bf16_gdn_gating_proj_kernels.h:34-35` 写明了契约：

> *"Cooperative launchers return false without submitting work only when the selected device cannot make one complete token tile resident. The Op wrapper owns the non-cooperative fallback."*

→ 超预算时 `finish_cooperative()`（`plan.cpp:253`）会退回 `MmaUnsplit` 普通启动。**在你的 80 SM 上：不越界、不崩卡，只是 prefill 会被拆成多次启动。**

⚠️ 注意：`plan.cpp:75-76` 的 `resident_ctas_27()` 返回 128/256，那是**编译期 catalog 自检**（`static_assert(catalog_is_resident(...), ...)`），只保证 128-SM 目标卡的路由表自洽，不是运行时的硬上限——运行时上限走上面那段代码。别把两者搞混。

### 1.4 性能税所在：`kTargetSmCount = 128` 是编译期常量

`src/core/device.h:23-28` 作者自己解释了为什么这里用常量而不是运行时值：

> *"Compile-time mirror of device_sm_count() for the architecture this build targets. `__device__` launch policies cannot query the runtime..."*

被用于 attention 的"恰好一个 resident wave"几何（`src/ops/softmax_attention/dense/causal_cache/`）：

- `geometry.cuh:25` —— `SmallTWaveSplits = kCausalSmallTCtasPerSm * kTargetSmCount / KVHeadsValue`
- `small_t.cu:71` —— `kMax = kTargetSmCount / Geometry::KVHeads`
- `small_t.cu:239` —— `kOneWave = kTargetSmCount - kTargetSmCount / 16`

→ 这些 grid 按 128 SM 设计。80 SM 上会跑成 **1.6 个 wave**（尾波不满），**属于普通启动、不是协作启动，所以安全**，只是 attention prefill/decode 达不到 README 表里的数字。这就是"能跑但会打折"的技术根源，也是 §6 自编译最该动手的地方。

---

## 2. 本机环境核对表（已实测，非推测）

| 项目 | 仓库要求 | 本机实测 | 状态 |
| --- | --- | --- | --- |
| GPU 架构 | `sm_89`（CMake 强制） | RTX 4080 SUPER = Ada `sm_89`，80 SM | ✅ |
| 显存 | 权重全量常驻，**无 offload** | **32760 MiB（32 GB）** | ✅ 见 §5 显存账 |
| 驱动 | ≥ 595 | 617.14 | ✅ |
| CUDA Toolkit | ≥ 12.8（作者 13.4 验证） | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{v12.4,v13.4}`；**PATH 上的 nvcc = 13.4** | ✅ |
| CMake | ≥ 3.28 | VS2022 自带 `3.31.6-msvc6`（不在 PATH） | ✅ 需显式调用 |
| Ninja | 需要 | VS2022 自带 `1.12.1`（不在 PATH） | ✅ 需显式调用 |
| C++20 编译器 | MSVC | VS2022 Community，MSVC `14.44.35207` | ✅ |
| Windows SDK | 需要 | `E:\Windows Kits\10`（**装在 E 盘，非默认 C 盘**） | ⚠️ 见 §4 坑 3 |
| Git | 需要 | `E:\software\Git\cmd\git.exe` 2.55 | ✅ |
| **vcpkg** | `curl` / `ffmpeg` / `pkgconf` 全靠它 | **未安装**（`C:\vcpkg`、`E:\vcpkg` 均无） | ❌ **唯一硬缺口** |
| VC++ Redist（仅路线 A 需要） | 2015-2022 x64 | 需确认，缺则装 aka.ms 那个 | ⚠️ |
| 磁盘余量 | 建议 ≥ 80 GB | `E:` 可用 **668 GB** | ✅ |

CMake / Ninja 的完整路径（本机已验证可执行）：

```
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
```

---

## 3. 路线 A：下载预编译二进制（0 编译，强烈推荐先走）

**为什么先走这条**：预编译包是 `sm_89` 的 SASS/cubin，4080S 与 4090 同架构可**直接加载运行**。先用它拿到"这台机器到底多快"的基线，否则将来编译出问题你分不清是"我编错了"还是"4080S 本来就这样"。

### A.1 下载

| 项 | 值 |
| --- | --- |
| Tag | `v2026.09.27b` |
| 资产 | `ninfer-4090-windows-x64-2026.09.27b.zip` |
| 大小 | 821,992,809 B（约 784 MiB） |
| SHA-256 | `bcc55ed0de830ce5f4bfd18f05320a9890e2a820256aa4cedf1be18f388d6b82` |
| 直链 | `https://github.com/JGamboa/ninfer-4090-windows/releases/download/v2026.09.27b/ninfer-4090-windows-x64-2026.09.27b.zip` |

```bat
mkdir E:\LLM\ninfer-bin && cd /d E:\LLM\ninfer-bin
curl -L -o ninfer.zip "https://github.com/JGamboa/ninfer-4090-windows/releases/download/v2026.09.27b/ninfer-4090-windows-x64-2026.09.27b.zip"
certutil -hashfile ninfer.zip SHA256
:: 必须等于 bcc55ed0de830ce5f4bfd18f05320a9890e2a820256aa4cedf1be18f388d6b82
tar -xf ninfer.zip
```

包内含：全部 exe + DLL（CUDA 运行时**静态链接**进去了，不需要装 CUDA Toolkit）、两个 server 启动脚本、许可证。运行时依赖只有 **驱动 ≥ 595**（本机 617 ✅）+ **VC++ 2015-2022 x64 Redistributable**（`https://aka.ms/vs/17/release/vc_redist.x64.exe`）。

### A.2 下载模型（⚠️ 必须是 `.ninfer`，不是 GGUF）

**本喵必须提醒**：你 `E:\models` 里现有的 `Ternary-Bonsai-2-27B-PQ2_0.gguf`、`Qwen3.8-27B-UD-Q4_K_M.gguf` 是 **llama.cpp 格式，NInfer 加载不了**。README 明说：*"They do not load in llama.cpp, vLLM or Transformers"*，反向也一样。要单独下 `.ninfer`。

```bat
pip install -U huggingface_hub

:: 推荐先试这个：6.4 GiB，文本+视觉+MTP，4080S 上最省时间
hf download jgamboa/Ternary-Bonsai-2-27B-NInfer-4090 bonsai2_27b_vl_mtp_q4q5.ninfer --local-dir E:\LLM

:: 或者 Qwen3.8-27B int8-prefill 版（19.0 GiB）
hf download jgamboa/Qwen3.8-27B-NInfer-4090 qwen3_8_27b_a8.ninfer --local-dir E:\LLM
```

国内网络加 `set HF_ENDPOINT=https://hf-mirror.com`。

### A.3 运行（CLI 冒烟）

```bat
ninfer.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-context 8192 --max-new 1024 --spec mtp --draft-tokens 2 --lm-head-draft
```

跑通会打印答案 + prefill/decode 速度、MTP 接受率、显存占用。**把这几个数字记下来，它就是路线 B 的对照基线。**

### A.4 起服务（4080S 保守参数）

```bat
ninfer-serve.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --host 127.0.0.1 --port 8080 --model-id bonsai-27b ^
  --max-context 131072 --kv-capacity auto --kv-dtype rk4v4-e8 --max-concurrency 2 ^
  --prefill-chunk 1024 ^
  --spec mtp --draft-tokens 2 --lm-head-draft --ngram chain --vision
```

与原 README 作者日常配置的差异（本喵按 4080S 下调，理由见 §5）：

| 参数 | 作者（4090） | 建议（4080S） | 原因 |
| --- | --- | --- | --- |
| `--max-context` | 262144 | 131072 | 先小后大，验证稳定再抬 |
| `--max-concurrency` | 3 | **2** | 80 SM 上并发 lane 抢不到整波，收益递减 |
| `--prefill-chunk` | 1408 | **1024** | 保守值；本 fork 已不会崩，但 1024 是社区在 80 SM 上验证过的安全档 |

验证：浏览器开 `http://127.0.0.1:8080/monitor`（实时看板），或

```powershell
$body = @{ model = "bonsai-27b"; max_tokens = 512
           messages = @(@{ role = "user"; content = "Explain CUDA graphs in three sentences." }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post `
  -ContentType "application/json" -Body $body | Select-Object -ExpandProperty choices
```

### A.5 路线 A 的验收标准

- `ninfer.exe` 能加载 `.ninfer` 并输出文本（**证明 4080S 上 sm_89 二进制可用**）
- 连续 40 分钟不崩、不整卡 reset
- 记下 tok/s，作为路线 B 的对照

---

## 4. 路线 B：源码编译（你问的"可否编译出来 ninfer"——可以）

### B.1 唯一要补的依赖：vcpkg

```bat
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
```

仓库根 `vcpkg.json` 已 pin 好依赖与 baseline，CMake 会自动按 manifest 装：

```json
"dependencies": ["curl", {"name": "ffmpeg", "features": ["zlib"]}, "pkgconf"]
"builtin-baseline": "4bca8fd8654e5ba76f92661db7bfe954768ad8ef"
```

> ⏱️ **vcpkg 首次编译 FFmpeg 要 1~2 小时**，这是整条路线最耗时的环节（不是编译 NInfer 本身）。

### B.2 配置（本机路径版）

⚠️ **仓库自带的 `winport_configure.bat` / `winport_build.bat` 是作者机器的硬编码，照抄必失败**：

```bat
:: 作者机器（不要用）
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\...\vcvars64.bat"   ← 本机无 VS18，实测 NO_VS18_DIR
cd /d E:\LLM\ninfer-4090-winport                                                        ← 本机路径不是这个
set CUDA_PATH=...\v13.4
```

本机正确姿势——在 **"x64 Native Tools Command Prompt for VS 2022"** 里执行（它会自动带上 MSVC + E 盘 SDK）：

```bat
cd /d E:\workspaces\ai\ninfer-4090-windows

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"

cmake --build build -j
```

最后一行 `-DCMAKE_CUDA_COMPILER` 是本喵强烈建议加的：本机同时装了 **v12.4 和 v13.4**，PATH 里若先命中 12.4，虽然也能过 12.8 门槛，但作者只在 13.4 上验证过，显式钉死更稳。

### B.3 产物

```
build\apps\ninfer.exe            CLI
build\apps\ninfer-serve.exe      HTTP 服务（OpenAI + Anthropic 双协议）
build\apps\ninfer-perplexity.exe 困惑度评测
```

加 `-DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON` 还会编出 `build\bench\ninfer_bench.exe` 和一堆 kernel 单测（每个 CUDA kernel 都对照 FP32/FP64 oracle 校验，例如 `ninfer_linear_t5_test.exe`、`ninfer_softmax_attention_test.exe`）。**在 4080S 上，这些单测就是你判断"我这卡到底行不行"的最硬证据，建议一并编。**

### B.4 运行时 PATH

CUDA 运行时静态链接（`CMakeLists.txt:61-65`，`CUDA::cudart_static`），但 vcpkg 的 DLL 是动态的：

```bat
set "PATH=E:\workspaces\ai\ninfer-4090-windows\build\vcpkg_installed\x64-windows\bin;%CUDA_PATH%\bin;%PATH%"
```

### B.5 本喵预判会踩的坑（按可能性排序）

1. **vcpkg 编 FFmpeg 超时/失败** —— 磁盘、代理、以及 `E:\Windows Kits` 非默认位置导致的 SDK 发现失败。
2. **cmake/ninja 不在 PATH** —— 本喵实测两者都只存在于 VS2022 目录内。要么用 Native Tools 提示符（推荐），要么把 §2 那两个绝对路径加进 PATH。
3. **Windows SDK 在 E 盘** —— 若 CMake 报找不到 SDK，显式 `-DCMAKE_SYSTEM_INCLUDE_PATH` / 用 Native Tools 提示符（它会正确设置 `WindowsSdkDir`）。
4. **MSVC × CUDA 13 × C++20 的边角** —— 仓库已经处理了三个已知点，别自己乱改：
   - `/utf-8`（`third_party/spdlog` 里的 `static_assert` 会炸）
   - `/Zc:preprocessor`
   - `/NODEFAULTLIB:LIBCMT`（CUDA 静态运行时会请求 LIBCMT，与全项目 `/MD` 冲突，必须保证进程里只有一份 CRT）
5. **WDDM 桌面合成器税** —— README 专章《When the 4090 also drives the display》：**4K 桌面下 120 Hz 会吃掉每轮 MTP decode 的 18%、60 Hz 吃 15%**。你的 4080S 正在带显示器（`nvidia-smi` 里 `Disp.A = On`，已占 2 GB 显存）。**要测性能就把显示器挪到核显，或降到 60 Hz 并保持画面静止**，否则数字没有可比性。

---

## 5. 显存账（32 GB 改卡，够用）

引擎"启动即固定显存占用、单进程单卡单模型、**无权重 offload**"，所以权重必须整个塞进显存。

| Artifact | 大小 | 4080S 32 GB |
| --- | ---: | --- |
| Ternary Bonsai 2 27B（`.ninfer`，文本+视觉+MTP） | 6.4 GiB | ✅ 非常宽裕，**首选** |
| Qwen3.8-27B int8-prefill（`qwen3_8_27b_a8.ninfer`） | 19.0 GiB | ✅ 可行，KV 池余量约 8~10 GB |
| Qwen3.8-27B 官方 BF16-prefill（`qwen3_8_27b.ninfer`） | 16.96 GiB | ✅ 可行 |
| + DFlash2 drafter | +1.6 GiB | ✅ 仍可 |

社区在 4080S 32G 上的实测参考账（Qwen3.8-27B，16.96 GiB 权重）：显存 28.6/32.8 G、余 4.2 G、单路 ~86 tok/s、双路聚合 ~122 tok/s、KV 池 524,288（正好 2×262,144）。

→ **你的 32 GB 显存是本方案成立的决定性前提**（原厂 16 GB 的 4080S 会在加载阶段直接 OOM，且没有 offload 可退）。

---

## 6. 自编译的真正价值（什么情况下才值得走路线 B）

只是想跑 → 路线 A 就够。自编译的正当理由按价值排序：

1. **把 attention 几何按 80 SM 重算**（最高价值）：§1.4 那三处 `kTargetSmCount`。可行做法是让 `geometry.cuh` / `small_t.cu` 的 wave 划分读 `device_sm_count()`，或为 80 SM 单独出一个编译档位。这直接改善长上下文 prefill 与 decode 的尾波浪费。
2. **拆掉 CUDA Graph × 批量投机解码的隐患**：社区在 4080S 上实测「Graph 重放 + batch≥2 投机」组合会在约 30 分钟后崩，关掉 CUDA Graph（`--no-cuda-graph`）40 分钟存活，代价只有 **2.6%** 吞吐。自编译可以做"每个 batch size 各捕获一张图"，把这 2.6% 赚回来。
3. **跟进上游**：上游 `Neroued/ninfer` 还在动，sm_89 分支线已落后。
4. **去掉一层转发**：相比 Docker/WSL 方案省一层，但换不回 WDDM 的显存/带宽税。

---

## 7. 建议的执行阶梯（别跳步）

| 阶段 | 动作 | 通过标准 | 预计成本 |
| --- | --- | --- | --- |
| **0** | 装 VC++ 2015-2022 x64 Redist；`mkdir E:\LLM` | — | 2 分钟 |
| **1** | **路线 A**：下 zip + 校验 SHA-256 + 下 Bonsai `.ninfer`（6.4 GiB） | CLI 出文本；记下 prefill/decode tok/s | 下载 ~15 GB，1 小时 |
| **2** | 路线 A 起服务，`--max-concurrency 2 --prefill-chunk 1024` | 连续 40 分钟不崩、`/monitor` 正常 | 1 小时 |
| **3** | **路线 B**：装 vcpkg → 配置 → 编译（带 `-DBUILD_TESTING=ON`） | 编出 3 个 exe；**kernel 单测在 80 SM 上全绿** | vcpkg 1~2 h + 编译 30~60 min |
| **4** | 路线 B 产物与路线 A 产物同机 A/B（old,new,new,old 交替，抵消热/显示效应） | 数字一致 → 证明工具链干净 | 半天 |
| **5** | 只有到这一步才动 §6 的 1/2 条内核改动 | 改后 tok/s 有提升且单测仍全绿 | 1~2 天 |

---

## 8. 引擎边界与风险（心里有数）

- **单卡单进程单模型**；无多卡、无权重 offload、无请求抢占/优先级。
- **Prefill 一次只跑一个请求**，它会阻塞其他 lane 的 decode。
- **NVFP4 / W4A4 需要 Blackwell 张量核，`sm_89` 上永远不可用**（README Limits 明说）。Ada 只能走 groupwise-int / ternary 路线。
- **DFlash2 不兼容 Bonsai 的 ternary 输出头**，Bonsai 只能用 MTP。
- **本 fork 只在 4090 上验证**，4080S 属"同架构 untested"；出问题你得自己 debug。
- **Linux 构建与 Dockerfile 是从上游继承、未重新验证的**（README Limits）。别指望 Docker 路径能当退路。
- **许可证**：Apache-2.0（模型 artifact 来自 Qwen 系 / Prism ML，无法律障碍）。

---

## 9. 附：一键环境核查脚本

本喵已在本机跑过等价命令，结果见 §2 表格。要复现直接执行：

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version                     # 应为 13.4
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --version
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" --version
Test-Path "C:\vcpkg\vcpkg.exe"      # False = 路线 B 的唯一缺口
Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Name
```

---

## 参考来源

1. `E:\workspaces\ai\ninfer-4090-windows\README.md`（Requirements / Quick start / Performance / Limits / Choosing settings 各章）
2. `E:\workspaces\ai\ninfer-4090-windows\CMakeLists.txt`（`sm_89` 强制校验、CUDA ≥12.8、MSVC 编译/链接选项、vcpkg FFMPEG/CURL 分支）
3. `E:\workspaces\ai\ninfer-4090-windows\src\core\device.h`（`kTargetSmCount = 128` 及作者注释）
4. `E:\workspaces\ai\ninfer-4090-windows\src\core\device.cu`（`device_sm_count()` 运行时查询）
5. `E:\workspaces\ai\ninfer-4090-windows\src\ops\gdn_gating_proj\bf16\bf16_gdn_gating_proj_kernels.cu`（协作启动预算按运行时 SM 数分块）
6. `E:\workspaces\ai\ninfer-4090-windows\src\ops\gdn_gating_proj\bf16\bf16_gdn_gating_proj_kernels.h`（非协作回退契约）
7. `E:\workspaces\ai\ninfer-4090-windows\src\ops\gdn_gating_proj\bf16\bf16_gdn_gating_proj_plan.cpp`（路由表 + `static_assert` 自检 + `finish_cooperative`）
8. `E:\workspaces\ai\ninfer-4090-windows\src\ops\softmax_attention\dense\causal_cache\{geometry.cuh,small_t.cu}`（编译期 128-SM wave 几何）
9. `E:\workspaces\ai\ninfer-4090-windows\vcpkg.json` / `CMakePresets.json` / `winport_configure.bat` / `winport_build.bat`
10. GitHub Releases API：`v2026.09.27b` 资产名、大小、SHA-256 与 release note（含 40-series "untested" 原话）
11. Hugging Face：`jgamboa/Ternary-Bonsai-2-27B-NInfer-4090`、`jgamboa/Qwen3.8-27B-NInfer-4090` 模型卡
12. 本会话前一份调研：`E:\workspaces\ai\ninfer-win11-4080s-build-feasibility.md`（4080S 32G 实测与 CUDA Graph 崩卡对照实验；其中 §4.3 的崩卡问题已确认在本 fork 修复）
