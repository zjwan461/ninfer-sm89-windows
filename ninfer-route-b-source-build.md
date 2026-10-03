# NInfer 路线二：源码自编译（Windows 11 + RTX 4080 SUPER 32GB）

> 目标机器：Windows 11 / RTX 4080 SUPER（AD103，**80 SM**，`sm_89`，32760MiB）／驱动 617.14／CUDA Toolkit v13.4
> 上游仓库：`https://github.com/JGamboa/ninfer-4090-windows`
> 本文只讲**方案二（自己编译）**。方案一（预编译包）见 `ninfer-4080s-two-routes.md`。

---

## 0. 一句话总结

本机**只缺 vcpkg**，其余工具链全部就位。补齐 vcpkg → 用 VS2022 的 x64 原生工具提示符 → 自己写 CMake 命令（**别用仓库自带的 winport bat**）→ 首编约 2 小时。

---

## 1. 本机环境盘点（已实测）

| 项目 | 状态 | 位置 / 版本 |
|---|---|---|
| Visual Studio 2022 Community | ✅ | MSVC 14.44.35207 |
| Windows SDK | ✅ | `E:\Windows Kits\10\10.0.26100.0`（在 E 盘，vcvars64 能自动定位，已验证） |
| CUDA Toolkit | ✅ | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`（nvcc V13.4.92，满足 CMakeLists 要求的 ≥12.8） |
| CMake | ✅ | 3.31.6-msvc6，在 VS 自带目录，**不在 PATH** |
| Ninja | ✅ | 1.12.1，在 VS 自带目录，**不在 PATH** |
| Git | ✅ | 2.55.0.windows.2（`E:\software\Git`） |
| **vcpkg** | ❌ **唯一缺口** | `C:\vcpkg`、`E:\vcpkg` 均不存在，计划装到 `E:\workspaces\c\vcpkg` |
| 磁盘 | ✅ | E: 615 GB 可用 / C: 232 GB 可用 |

> CUDA 目录下同时存在 v12.4 与 v13.4，PATH 里生效的是 **v13.4**，用它。

---

## 2. 步骤清单

### 第 0 步 · 装 vcpkg（唯一要补的软件）

```bat
git clone https://github.com/microsoft/vcpkg E:\workspaces\c\vcpkg
E:\workspaces\c\vcpkg\bootstrap-vcpkg.bat
setx VCPKG_ROOT E:\workspaces\c\vcpkg
```

- 用时约 1~3 分钟（clone 本身只拉 vcpkg 主仓，端口源码是后面 configure 时按需下载的）。
- 装完**新开终端**让 `VCPKG_ROOT` 生效。
- 装到 E 盘的理由：vcpkg 编 FFmpeg/curl 时 `buildtrees`、`packages`、`downloads` 三个目录会吃掉 **10~20 GB**，E 盘有 615 GB 余量，C 盘只有 232 GB，别去挤系统盘。
- 如果 `E:\workspaces\c` 不存在，先 `mkdir E:\workspaces\c`。
- **一个都不要少**：仓库在 Windows 上强制 `find_package(FFMPEG REQUIRED)` + `find_package(CURL 7.85 REQUIRED)`，这两个都由 vcpkg 提供。

### 第 1 步 · 切换到正确的工具链环境

用开始菜单里的 **「x64 Native Tools Command Prompt for VS 2022」**，或在普通 cmd 里手动加载：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

- 理由：CMake / Ninja 都不在默认 PATH，只有这个提示符会把它们和 SDK 一起带进来。
- SDK 装在 E 盘不用特殊处理，vcvars64 会自己找到。

### 第 2 步 · 源码放到短英文路径

```bat
cd /d E:\LLM
git clone https://github.com/JGamboa/ninfer-4090-windows ninfer-4090-winport
cd ninfer-4090-winport
```

- 用短路径（如 `E:\LLM\ninfer-4090-winport`），避免长路径 + 中文路径。
- 仓库的 `third_party/`（cpp-httplib、llama-jinja、nlohmann、spdlog、utf8proc）已随源码就位，**没有 git submodule，不用额外拉**。

### 第 3 步 · Configure（**取代 winport_configure.bat**）

> ⚠️ 不要直接跑仓库自带的 `winport_configure.bat` / `winport_build.bat`：里面硬编码了 **VS18 BuildTools** 和作者的 `E:\LLM\ninfer-4090-winport`，本机照抄必失败。照下面的参数自己敲。

```bat
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe" ^
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"
```

要点：
- `CMAKE_CUDA_ARCHITECTURES` **必须是 89**，CMakeLists 里写死了非 89 直接 `FATAL_ERROR`。
- `CUDAToolkit_ROOT` 必须显式指 v13.4，否则可能抓到 v12.4 那套。
- 这一步 vcpkg 会自动开始拉取并编译它需要的端口（**curl、ffmpeg、zlib、pkgconf**）→ **首次 1~2 小时**，这是全程最慢的环节。

### 第 4 步 · Build

```bat
cmake --build build -j
```

- 编译器自动带 `/Zc:preprocessor`、`/utf-8`、`/NODEFAULTLIB:LIBCMT`（CMakeLists 里已配好，不用手动加）。
- 本体编译约 20~40 分钟（含大量 .cu）。

### 第 5 步 · 验证

- 产物默认在 `build/` 下，跑起来后跟预编译包对个 tok/s，看有没有差异。
- 若 CMakeLists 有安装规则，也可 `cmake --install build --prefix E:\LLM\ninfer-runtime`。

---

## 3. 预估耗时与磁盘

| 阶段 | 耗时 |
|---|---|
| git clone vcpkg + bootstrap | 1~3 分钟 |
| git clone 仓库 | < 1 分钟 |
| vcpkg 首编依赖（FFmpeg 最久） | **1~2 小时** |
| Configure 本体 | 1~2 分钟 |
| Build 本体 | 20~40 分钟 |
| **合计（从零）** | **约 1.5~2.5 小时** |

磁盘：源码 + build + vcpkg 依赖合计建议留 **40~60 GB**（E 盘 615 GB 富余，没问题）。想省空间的话 build 目录别关掉增量。

---

## 4. 已知坑（必读）

### 4.1 SM 数不匹配：`kTargetSmCount = 128` vs 本机 80

- `src/core/device.h:28` 里写死 `inline constexpr int kTargetSmCount = 128;`（针对 RTX 4090 的 128 SM，编译期常量）。
- 它被 attention 的 `causal_cache/geometry.cuh`、`small_t.cu` / `small_t.cuh` 用来算 resident wave split 数 → 在 80 SM 的 4080S 上**会多跑波次，损失一部分理想性能，但不会崩**。
- 运行时真实 SM 数由 `src/core/device.cu:81-86` 的 `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` 查询，所以只影响调度效率。
- **自编译最大的价值就在这**：想榨干 4080S 就把这个常量按 80 重算。不改的话功能正常，只是不是最优。
- 补一句：早前"80 SM 协作启动越界崩卡"的担心在本 fork 里**已修复**——`src/ops/gdn_gating_proj/bf16/*_kernels.cu:358-375` 会用 `resident_ctas = multiprocessor_count * kResidentCtasPerSm` 算预算，超了自动按 token 分块启动并回退 `MmaUnsplit`。

### 4.2 WDDM 显示税

- 本机 4080S 正带显示器（`Disp.A = On`，已占约 2 GB 显存），**4K@120Hz 会吃掉每轮 MTP decode 约 18%**。
- 测性能前把显示输出挪到核显（或拔屏），否则数据没法看。

### 4.3 硬门槛（CMakeLists 强制）

- CUDA **≥ 12.8**、CMake **≥ 3.28**、**C++20**。
- MSVC 需要 `/Zc:preprocessor`（否则预处理器行为不对，编译报错）。
- Windows 平台强制 vcpkg 提供 FFMPEG + CURL 7.85 —— 这也是为什么 vcpkg 是硬依赖。

### 4.4 模型格式

- NInfer 加载的是 **`.ninfer`** 格式，**不是 `.gguf`**。`E:\models` 里现有的 Bonsai / Qwen3.8 是 GGUF（llama.cpp 用），NInfer 加载不了，需要另下 `.ninfer` 权重。

### 4.5 顺带记录的接口坑（来自方案一实测，与编译无关）

- LangChain 在 `create_deep_agent(name="SassyCat", ...)` 下会把 `name` 传给 AI 消息，落到请求体就成了 `{"role":"assistant","name":"SassyCat"}`；NInfer 的 `src/serve/openai_chat_request.cpp:437-450` 校验 `name` 非空且 role ≠ Tool 时直接返回 **400 `message_name_not_supported`**。
- 即：NInfer 只实现 Chat Completions，不吃带 `name` 字段的历史消息。**（本次不修代码，仅记录）**

---

## 5. 与方案一的关系

- 先跑通**方案一（预编译包）拿性能基线**，再考虑自编译，别一上来就啃编译。
- 自编译的收益主要是：(1) 能把 `kTargetSmCount` 按 80 SM 调优；(2) 能改内核；(3) 不依赖预编译发布（发布方明确说 *"targets the RTX 4090 only … other RTX 40-series may work with Bonsai (untested)"*）。
- 自编译的成本：vcpkg 首编 1~2 小时 + 一次性的工具链折腾。