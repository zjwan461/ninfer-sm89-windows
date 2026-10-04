# NInfer Windows 源码编译手册（RTX 4080 SUPER / 80 SM）

> 目标：在 **Windows 11 x64 + RTX 4080 SUPER（AD103，80 SM，`sm_89`）** 上，从源码编译 NInfer（`zjwan461/ninfer-sm89-windows`，fork 自 `JGamboa/ninfer-4090-windows`），并可选地按 80 SM 重定向 attention 几何。
> 本机实测环境：VS2022 Community（MSVC 14.44）、Windows SDK `E:\Windows Kits\10\10.0.26100.0`、CUDA Toolkit `v13.4`（同时装了 v12.4，勿用）、驱动 617.14。
> 唯一缺口：**vcpkg**（本手册第 0 步安装）。
> 配套文档：`README.md`、`WINDOWS_PORT.md`、`docs/maintainer/build-system.md`、`ninfer-4080s-80sm-retarget-plan.md`。

---

## 0. 三十秒速览

```bat
:: 0) 装 vcpkg（唯一缺失项）
git clone https://github.com/microsoft/vcpkg E:\workspaces\c\vcpkg
E:\workspaces\c\vcpkg\bootstrap-vcpkg.bat
setx VCPKG_ROOT E:\workspaces\c\vcpkg      :: 装完新开终端

:: 1) 用「x64 Native Tools Command Prompt for VS 2022」打开 cmd，然后：
cd /d E:\workspaces\ai\ninfer-sm89-windows
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"

:: 2) configure + build（4080S = 80 SM 档，输出到 build-4080s）
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=E:/workspaces/c/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=80 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
cmake --build build-4080s -j
```

首次 configure 会由 vcpkg 拉取并编译 `curl / ffmpeg / pkgconf`（**1~2 小时**，全程最慢环节）；之后增量编译 NInfer 本体约 20~40 分钟。

---

## 1. 三个构建档位（先搞清要建哪个）

| 档位 | `NINFER_TARGET_SM_COUNT` | 输出目录 | 用途 |
|---|---:|---|---|
| `release`（默认） | 128 | `build/` | 4090 基线；无测试/基准 |
| `dev` | 128 | `build/` | 4090 基线 + 测试 + 基准 |
| **`release-4080s`** | **80** | `build-4080s/` | **本机 4080S 优化档** |

> `NINFER_TARGET_SM_COUNT` 是本次新增的编译开关：它只决定 attention wave 几何按多少 SM 编译（**不是架构开关**，架构恒为 `sm_89`）。合法值必须是**偶数且 ≥ 66**；否则 configure 期与编译期都会报错。128 与 80 两档在任意卡上都能跑，只是档位与卡不匹配时会损失一点波次利用率，不影响正确性。
> 编译产物**互不影响**：两档用不同输出目录，可同时保留做 A/B。

**重要**：仓库自带的 `CMakePresets.json` **只含构建选项，不含本机工具链路径**（vcpkg / CUDA / 编译器）。因此有两种用法：
- **用法 A（本手册主推，最直观）**：不用 preset，直接敲完整 `-D` 命令（见 §3.2）。
- **用法 B**：用具名 preset + 本机覆盖文件 `CMakeUserPresets.json`（见 §3.3）。

> ⚠️ **不要用仓库自带的 `winport_configure.bat` / `winport_build.bat`**：里面硬编码了作者机器的 `VS18 BuildTools` 与 `E:\LLM\ninfer-4090-winport`，在本机必然失败。

---

## 2. 环境要求（硬门槛，CMake 会强制）

| 项 | 要求 | 本机 |
|---|---|---|
| CUDA Toolkit | **≥ 12.8** | 13.4 ✅（**显式钉死 v13.4，避免命中 v12.4**） |
| CMake | ≥ 3.28 | VS2022 自带 3.31.6 ✅ |
| Ninja | 需要 | VS2022 自带 1.12.1 ✅ |
| 编译器 | MSVC + C++20 | VS2022 Community ✅ |
| GPU 架构 | `sm_89`（CMakeLists 只接受 89） | 4080S = `sm_89` ✅ |
| 依赖 | vcpkg 提供 `curl` / `ffmpeg` / `pkgconf` | ❌ 待装 |
| 磁盘 | 建议预留 40~60 GB | E: 富余 ✅ |

CMake / Ninja 不在默认 PATH，只在 VS2022 目录内：
```
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
```
→ 用「x64 Native Tools Command Prompt for VS 2022」即可自动带上它们与 SDK。

---

## 3. 分步操作

### 第 0 步 · 安装 vcpkg（唯一要补的软件）

```bat
if not exist E:\workspaces\c mkdir E:\workspaces\c
git clone https://github.com/microsoft/vcpkg E:\workspaces\c\vcpkg
E:\workspaces\c\vcpkg\bootstrap-vcpkg.bat
setx VCPKG_ROOT E:\workspaces\c\vcpkg
```
- 装到 **E 盘**：vcpkg 编 FFmpeg 时 `buildtrees/packages/downloads` 会吃 10~20 GB。
- 装完 **新开终端** 让 `VCPKG_ROOT` 生效。
- 依赖清单已由仓库 `vcpkg.json` 的 manifest 固定（`curl` + `ffmpeg[zlib]` + `pkgconf`），CMake 配置时自动安装，**无需手动 `vcpkg install`**。

### 第 1 步 · 进入正确的工具链环境

用开始菜单的 **「x64 Native Tools Command Prompt for VS 2022」**，或在普通 cmd 里手动加载：
```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```
再把 CUDA v13.4 放到 PATH：
```bat
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
```

### 第 2 步 · 源码位置

```bat
cd /d E:\workspaces\ai\ninfer-4090-windows
```
- 用**短英文路径**，避免中文/超长路径。
- `third_party/`（cpp-httplib、llama-jinja、nlohmann、spdlog、utf8proc）已随源码就位，**无 git submodule**。

### 第 3 步 · Configure

#### 3.2 用法 A：完整命令（推荐）

**A-1 · 4080S 档（80 SM）：**
```bat
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=E:/workspaces/c/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=80 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
```

**A-2 · 4090 基线 + 测试 + 基准（128 SM）：**
```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=E:/workspaces/c/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON ^
  -DPython3_EXECUTABLE="<你的 python.exe 全路径>"
```

要点：
- `CMAKE_CUDA_ARCHITECTURES` **必须 89**，否则 `FATAL_ERROR`。
- 显式 `-DCUDAToolkit_ROOT` + `-DCMAKE_CUDA_COMPILER`：本机有 v12.4/v13.4 两套，钉死 v13.4。
- `-DNINFER_TARGET_SM_COUNT=80`：4080S 优化档；不传则默认 128。
- `-DPython3_EXECUTABLE`：仅测试需要，指向装了依赖的 Python 3。

#### 3.3 用法 B：具名 preset + 本机覆盖

新建 **`CMakeUserPresets.json`**（该文件已被 `.gitignore` 忽略，适合放本机路径）：
```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local-4080s",
      "inherits": "release-4080s",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "E:/workspaces/c/vcpkg/scripts/buildsystems/vcpkg.cmake",
        "VCPKG_TARGET_TRIPLET": "x64-windows",
        "CUDAToolkit_ROOT": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4",
        "CMAKE_CUDA_COMPILER": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe"
      }
    },
    {
      "name": "local-dev",
      "inherits": "dev",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "E:/workspaces/c/vcpkg/scripts/buildsystems/vcpkg.cmake",
        "VCPKG_TARGET_TRIPLET": "x64-windows",
        "CUDAToolkit_ROOT": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4",
        "CMAKE_CUDA_COMPILER": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe",
        "Python3_EXECUTABLE": "<你的 python.exe 全路径>"
      }
    }
  ],
  "buildPresets": [
    { "name": "local-4080s", "configurePreset": "local-4080s" },
    { "name": "local-dev", "configurePreset": "local-dev" }
  ]
}
```
然后：
```bat
cmake --preset local-4080s
cmake --build --preset local-4080s
```
> `release-4080s` 档自带 `binaryDir: build-4080s`；`dev` 档用 `build/`。

### 第 4 步 · Build

```bat
cmake --build build-4080s -j
:: 或（dev 档）cmake --build build -j
```
- 编译选项（`/Zc:preprocessor`、`/utf-8`、`/NODEFAULTLIB:LIBCMT`）已在 CMakeLists 配好，**不要手动增删**。
- Ninja 下链接使用单槽 `ninfer_link` 池，属正常现象。

### 第 5 步 · 运行时 PATH

CUDA 运行时是**静态链接**的，但 vcpkg 的 DLL 是动态的，运行前需：
```bat
set "PATH=E:\workspaces\ai\ninfer-4090-windows\build-4080s\vcpkg_installed\x64-windows\bin;%CUDA_PATH%\bin;%PATH%"
```
（若用 `build/` 则把目录名相应替换。）

### 第 6 步 · 冒烟验证

产物：
```
build-4080s\apps\ninfer.exe            CLI
build-4080s\apps\ninfer-serve.exe      HTTP 服务（OpenAI + Anthropic 双协议）
build-4080s\apps\ninfer-perplexity.exe 困惑度评测
```
先看帮助与设备识别：
```bat
build-4080s\apps\ninfer.exe --help
```
> ⚠️ **模型格式**：NInfer 只加载 **`.ninfer`**，不能加载 `.gguf`（llama.cpp 用）。

### 第 7 步 · 下载模型（`.ninfer`）

```bat
pip install -U huggingface_hub
:: 国内网络可加镜像
set HF_ENDPOINT=https://hf-mirror.com

:: 推荐先试：6.4 GiB，文本 + 视觉 + MTP（最省时间）
hf download jgamboa/Ternary-Bonsai-2-27B-NInfer-4090 bonsai2_27b_vl_mtp_q4q5.ninfer --local-dir E:\LLM

:: Qwen3.8-27B int8-prefill 版（19.0 GiB，1.7-1.9x 更快 prefill）
hf download jgamboa/Qwen3.8-27B-NInfer-4090 qwen3_8_27b_a8.ninfer --local-dir E:\LLM
```
> 每个 artifact 请用其 HuggingFace 模型卡上的校验值核对。

---

## 4. 运行（4080S 保守参数）

**CLI 冒烟：**
```bat
build-4080s\apps\ninfer.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-context 8192 --max-new 1024 --spec mtp --draft-tokens 2 --lm-head-draft
```
跑通会打印答案 + prefill/decode 速度、MTP 接受率、显存占用。

**起服务（按 80 SM 下调的保守档）：**
```bat
build-4080s\apps\ninfer-serve.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --host 127.0.0.1 --port 8080 --model-id bonsai-27b ^
  --max-context 131072 --kv-capacity auto --kv-dtype rk4v4-e8 --max-concurrency 2 ^
  --prefill-chunk 1024 ^
  --spec mtp --draft-tokens 2 --lm-head-draft --ngram chain --vision
```
- 实时看板：浏览器打开 `http://127.0.0.1:8080/monitor`；指标 `/metrics`、槽位 `/slots`。
- 若遇到「CUDA Graph × batch≥2 投机解码」长时间运行崩溃，加 `--no-cuda-graph`（代价约 2.6% 吞吐）。

---

## 5. 测试与基准（dev 档）

```bat
:: 全量
ctest --test-dir build --output-on-failure

:: 只跑关键 kernel oracle 单测
ctest --test-dir build -R "^ninfer_softmax_attention_test$" --output-on-failure
build\tests\ninfer_softmax_attention_test --rk4v4-e8-only

:: 打印误差统计（诊断用，不改判定）
set NINFER_OP_REPORT_STATS=1
ctest --test-dir build -V -R "^ninfer_(rmsnorm|softmax_attention)_test$"

:: 长上下文 attention 基准（128 vs 80 对照用）
build\bench\ninfer_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4 ^
  --kv-dtype int8 --batch 1 --tokens 1024 --context 8192,32768,65536,131072 ^
  --mapping fragmented --execution eager --cache cold --warmup 5 --repeat 21
```
这些是**独立数学 oracle 对比**，改 wave 几何不应影响通过与否。

---

## 6. 确认 `NINFER_TARGET_SM_COUNT` 真的生效

仓库开启了 `CMAKE_EXPORT_COMPILE_COMMANDS`，可直接查编译数据库：
```bat
findstr /C:"NINFER_TARGET_SM_COUNT=80" build-4080s\compile_commands.json
```
能命中即说明 80 SM 档已注入所有相关 TU。

---

## 7. 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `cmake`/`ninja` 找不到 | 不在默认 PATH | 用「x64 Native Tools」提示符，或加 §2 两个绝对路径到 PATH |
| `find_package(FFMPEG REQUIRED)` 失败 | 未接 vcpkg 工具链 | 确认 `-DCMAKE_TOOLCHAIN_FILE=...vcpkg.cmake` 且路径存在 |
| `find_package(CURL 7.85 REQUIRED)` 失败 | vcpkg 未装/未装 curl | 确认 `VCPKG_ROOT`，让 manifest 自动装依赖 |
| 抓到 CUDA **v12.4** | PATH/root 未钉死 | 显式 `-DCUDAToolkit_ROOT` + `-DCMAKE_CUDA_COMPILER` 指 v13.4 |
| `requires CUDA 12.8 or newer` | nvcc 版本过低 | 同上，改用 v13.4 的 nvcc |
| `NInfer supports only CMAKE_CUDA_ARCHITECTURES=89` | 架构值不对 | 传 `-DCMAKE_CUDA_ARCHITECTURES=89` |
| `NINFER_TARGET_SM_COUNT must be an even integer >= 66` | SM 值非法 | 用 128 或 80（偶数且 ≥66） |
| 首次 configure 卡很久 | vcpkg 在编 FFmpeg | 正常，**1~2 小时**；别中断 |
| 找不到 Windows SDK | SDK 在 `E:\Windows Kits` | 用「x64 Native Tools」提示符（会正确设置 `WindowsSdkDir`） |
| 改过 CUDA 编译器后报错 | 旧 build 目录绑定旧编译器 | 删掉 build 目录重新 configure |
| 运行时报缺 DLL | vcpkg bin 不在 PATH | 见 §5 第 5 步设置运行时 PATH |
| 链接报 LIBCMT 冲突 | 手动改了链接选项 | 恢复仓库默认（`/NODEFAULTLIB:LIBCMT` 已在 CMakeLists 中） |

---

## 8. 一键脚本（可选）

把下面存成 `build-4080s.bat` 放到仓库根，双击/命令行运行即可（仍建议在 Native Tools 提示符下用）：

```bat
@echo off
setlocal
if "%VCPKG_ROOT%"=="" set "VCPKG_ROOT=E:\workspaces\c\vcpkg"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
pushd "%~dp0"
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=80 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" 1>configure.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if not "%ERRORLEVEL%"=="0" (type configure.log & popd & exit /b 1)
cmake --build build-4080s -j 1>build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
if not "%ERRORLEVEL%"=="0" type build.log
popd
endlocal
```

---

## 9. 验收清单

- [ ] `build-4080s\apps\ninfer.exe`、`ninfer-serve.exe`、`ninfer-perplexity.exe` 三个 exe 均生成。
- [ ] `ninfer.exe --help` 正常；能加载 `.ninfer` 并输出文本。
- [ ] `findstr /C:"NINFER_TARGET_SM_COUNT=80" build-4080s\compile_commands.json` 命中。
- [ ] dev 档 `ctest` 全绿（尤其 `ninfer_softmax_attention_test`，含 `--rk4v4-e8-only`）。
- [ ] 回归锚点：Bonsai 困惑度 ≈ 5.8549；长上下文检索输出 `ORCHID=493817; COLOR=COBALT`。
- [ ] 起服务连续 40 分钟不崩、`/monitor` 正常。

> 测性能前务必先**控显示税**：本机 4080S 若在带显示器，4K/120Hz 会吃掉每轮 MTP decode 约 18%——把显示输出挪到核显或降到 60Hz 静止后再测，否则数据不可比。