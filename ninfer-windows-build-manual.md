# NInfer Windows 源码编译手册（`sm_89`：RTX 4090 / 4080 SUPER）

> 目标：在 **Windows 11 x64 + `sm_89` 显卡**上，从源码编译 NInfer（本 fork，`sm_89` 原生 Windows 移植），并按需要选择 128 SM（RTX 4090）或 80 SM（RTX 4080 SUPER）的 attention 几何档位。
> 前置：只需要 **Visual Studio 2022 + CUDA Toolkit**；唯一的额外软件 **vcpkg** 由 §3 第 0 步安装。
> 约定：本手册用占位符与环境变量，不与任何机器绑定——
> `<repo>` 源码目录、`<vcpkg-root>` vcpkg 安装目录、`<cuda-path>` CUDA 安装目录、`<model-dir>` 模型目录、`<vcvars64.bat>` VS 的编译环境脚本。
> 配套文档：`README.md`、`WINDOWS_PORT.md`、`docs/maintainer/build-system.md`、`ninfer-4080s-80sm-retarget-plan.md`。
> 想看占位符在一台真实机器上填什么、以及**填好值的完整命令**，见 §11（**示例，非要求**）。

---

## 0. 三十秒速览

```bat
:: 0) 装 vcpkg（唯一的额外依赖；目录随意，建议放在空间充足的盘）
set "VCPKG_ROOT=<vcpkg-root>"
git clone https://github.com/microsoft/vcpkg "%VCPKG_ROOT%"
"%VCPKG_ROOT%\bootstrap-vcpkg.bat"
setx VCPKG_ROOT "%VCPKG_ROOT%"      :: 装完新开终端才有这个变量

:: 1) 打开「x64 Native Tools Command Prompt for VS 2022」（或自行 call vcvars64.bat），然后：
cd /d <repo>
set "CUDA_PATH=<cuda-path>"
set "PATH=%CUDA_PATH%\bin;%PATH%"

:: 2) configure + build（示例：80 SM 档 = RTX 4080 SUPER，输出到 build-4080s；128 SM 档 = RTX 4090 见 §3 第 3 步）
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=80 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
cmake --build build-4080s -j
```

首次 configure 会由 vcpkg 按仓库 manifest 拉取并编译 `curl / ffmpeg / pkgconf`（**1~2 小时**，全程最慢环节，请勿中断）；之后增量编译 NInfer 本体约 20~40 分钟。

---

## 1. 两个构建档位（先搞清要建哪个）

| 档位 | `NINFER_TARGET_SM_COUNT` | 建议输出目录 | 用途 |
|---|---:|---|---|
| `release`（默认） | 128 | `build-4090`（或 `build`） | RTX 4090 基线；无测试/基准 |
| `dev` | 128 | `build` | RTX 4090 基线 + 测试 + 基准 |
| `release-4080s` | 80 | `build-4080s` | RTX 4080 SUPER 优化档 |

> `NINFER_TARGET_SM_COUNT` 是本 fork 新增的编译开关：它只决定 attention wave 几何按多少 SM 编译（**不是架构开关**，架构恒为 `sm_89`）。合法值必须是**偶数且 ≥ 66**，否则 configure 期与编译期都会报错。128 与 80 两档在任意 `sm_89` 卡上都能跑，只是档位与卡不匹配时会损失一点波次利用率，不影响正确性。
> 编译产物**互不影响**：两档用不同输出目录，可同时保留做 A/B。

**重要**：仓库自带的 `CMakePresets.json` **只含构建选项，不含工具链路径**（vcpkg / CUDA / 编译器），因此有两种用法：
- **用法 A（主推，最直观）**：不用 preset，直接敲完整 `-D` 命令（见 §3 第 3 步）。
- **用法 B**：用具名 preset + 本机覆盖文件 `CMakeUserPresets.json`。

> ⚠️ 仓库自带的 `winport_configure.bat` / `winport_build.bat` 是移植作者机器上的原始脚本，里面写死了其本机的 VS 安装目录与源码目录，**在别的机器上会失败**：请用本手册的命令，或自行改这两个脚本。

---

## 2. 环境要求（硬门槛，CMake 会强制）

| 项 | 要求 | 检查方法 |
|---|---|---|
| CUDA Toolkit | **≥ 12.8** | `nvcc --version` |
| CMake | ≥ 3.28 | `cmake --version` |
| Ninja | 需要 | `ninja --version` |
| 编译器 | MSVC + C++20 | VS 2022（Community / Professional / Build Tools 任一） |
| GPU 架构 | `sm_89`（`CMakeLists.txt` 只接受 89） | `nvidia-smi --query-gpu=name --format=csv` |
| 依赖 | vcpkg 提供 `curl` / `ffmpeg` / `pkgconf` | 由 manifest 自动安装 |
| 磁盘 | 建议预留 40~60 GB | vcpkg 编 FFmpeg 时 `buildtrees/packages/downloads` 会吃 10~20 GB |

CMake 与 Ninja 随 VS 2022 一起安装，但**不在默认 PATH**，只在 VS 安装目录内。用「x64 Native Tools Command Prompt for VS 2022」即可自动带上它们与 Windows SDK，无需手工配 PATH。

---

## 3. 分步操作

### 第 0 步 · 安装 vcpkg（唯一要补的软件）

```bat
set "VCPKG_ROOT=<vcpkg-root>"
git clone https://github.com/microsoft/vcpkg "%VCPKG_ROOT%"
"%VCPKG_ROOT%\bootstrap-vcpkg.bat"
setx VCPKG_ROOT "%VCPKG_ROOT%"
```
- 目录随意，但建议放在**空间充足的盘**：vcpkg 编 FFmpeg 会吃 10~20 GB。
- 装完 **新开终端** 让 `VCPKG_ROOT` 生效。
- 依赖清单已由仓库 `vcpkg.json` 的 manifest 固定（`curl` + `ffmpeg[zlib]` + `pkgconf`），CMake 配置时自动安装，**无需手动 `vcpkg install`**。

### 第 1 步 · 进入正确的工具链环境

优先用开始菜单里的 **「x64 Native Tools Command Prompt for VS 2022」**。若要在普通 cmd 里手动加载，先定位 VS 安装目录（`vswhere` 随 VS 安装器提供）：

```bat
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%i"
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
```
> 上面是**命令行**写法；写进 `.bat` 文件时循环变量要写成 `%%i`。

再把 CUDA 放进 PATH：

```bat
set "CUDA_PATH=<cuda-path>"
set "PATH=%CUDA_PATH%\bin;%PATH%"
```
> 若机器上装了多个 CUDA 版本，务必显式指定要用的那个（见第 3 步的两个 `-D`），否则可能命中旧版本而报「requires CUDA 12.8 or newer」。

### 第 2 步 · 源码位置

```bat
cd /d <repo>
```
- 用**短英文路径**，避免中文/超长路径。
- `third_party/`（cpp-httplib、llama-jinja、nlohmann、spdlog、utf8proc）已随源码就位，**无 git submodule**。

### 第 3 步 · Configure

#### 用法 A：完整命令（推荐）

**A-1 · 80 SM 档（RTX 4080 SUPER）：**
```bat
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=80 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
```

**A-2 · 128 SM 档（RTX 4090）：**
```bat
cmake -S . -B build-4090 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=128 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
```

**A-3 · dev 档（128 SM + 测试 + 基准）：**
```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=128 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON ^
  -DPython3_EXECUTABLE="<python.exe 全路径>"
```

要点：
- `CMAKE_CUDA_ARCHITECTURES` **必须 89**，否则 `FATAL_ERROR`。
- 装了多个 CUDA 时，用 `-DCUDAToolkit_ROOT` + `-DCMAKE_CUDA_COMPILER` 钉死要用的版本。
- `-DNINFER_TARGET_SM_COUNT`：**128 = RTX 4090**（不传即默认 128），**80 = RTX 4080 SUPER**。两张卡用同一份源码、同一 `sm_89` 架构，只是该常量不同 → 建议分别建到 `build-4090` / `build-4080s`。
- `-DPython3_EXECUTABLE`：仅测试需要，指向装了依赖的 Python 3。

#### 用法 B：具名 preset + 本机覆盖

新建 **`CMakeUserPresets.json`**（该文件已被 `.gitignore` 忽略，适合放本机路径）：

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local-4080s",
      "inherits": "release-4080s",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "<vcpkg-root>/scripts/buildsystems/vcpkg.cmake",
        "VCPKG_TARGET_TRIPLET": "x64-windows",
        "CUDAToolkit_ROOT": "<cuda-path>",
        "CMAKE_CUDA_COMPILER": "<cuda-path>/bin/nvcc.exe"
      }
    },
    {
      "name": "local-dev",
      "inherits": "dev",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "<vcpkg-root>/scripts/buildsystems/vcpkg.cmake",
        "VCPKG_TARGET_TRIPLET": "x64-windows",
        "CUDAToolkit_ROOT": "<cuda-path>",
        "CMAKE_CUDA_COMPILER": "<cuda-path>/bin/nvcc.exe",
        "Python3_EXECUTABLE": "<python.exe 全路径>"
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
> 路径分隔符统一用 `/`（JSON 里反斜杠要转义）。`release-4080s` 档自带 `binaryDir: build-4080s`；`dev` 档用 `build/`。

### 第 4 步 · Build

```bat
cmake --build build-4080s -j
:: 或（dev 档）cmake --build build -j
```
- 编译选项（`/Zc:preprocessor`、`/utf-8`、`/NODEFAULTLIB:LIBCMT`）已在 CMakeLists 配好，**不要手动增删**。
- Ninja 下链接使用单槽 `ninfer_link` 池，属正常现象。

### 第 5 步 · 运行时 PATH（通常**不需要**做）

**先别做这一步。** `VCPKG_APPLOCAL_DEPS=ON`（vcpkg 在 Windows 上的默认值）会在构建时对每个 exe 运行 `vcpkg z-applocal`，把该 exe 依赖的 DLL 复制到 **exe 同目录**；而 Windows 的 DLL 搜索顺序默认就包含该目录。所以编译完直接跑即可：

```
build-4080s\apps\ninfer-serve.exe          ← 直接运行
build-4080s\apps\avcodec-62.dll            ← app-local 复制进来的依赖
build-4080s\apps\libcurl.dll  swscale-9.dll  z.dll  ...
```

> 想自证：`dir build-4080s\apps\*.dll` 会看到上面这一小撮（`vcpkg_installed\x64-windows\bin` 里的 `avdevice / avfilter / pkgconf` 运行时并不需要）；`build.ninja` 里每个 exe 的 `POST_BUILD` 就是那条 `vcpkg.exe z-applocal` 命令。

CUDA 侧同样不需要：Windows 走的是**静态** CUDA 运行时（`CMakeLists.txt:78-82` → `CUDA::cudart_static`），且工程没有引用 cublas / cufft / cusparse —— `%CUDA_PATH%\bin` 只对**编译期**有意义。

只有下列情况才需要手动补 PATH（或把 DLL 一起拷过去）：

| 情况 | 处理 |
|---|---|
| 把 exe 单独拷到别处（没带同目录的 DLL） | 连 DLL 一起拷，或加 PATH |
| 构建时关了 app-local（`-DVCPKG_APPLOCAL_DEPS=OFF`），或换了不支持该 POST_BUILD 机制的生成器/工具链 | 加 PATH |
| 缺的是没被 app-local 复制进来的 DLL（如 `pkgconf-8.dll`，构建工具，一般运行不需要） | 加 PATH |

兜底命令：

```bat
set "PATH=<repo>\build-4080s\vcpkg_installed\x64-windows\bin;%CUDA_PATH%\bin;%PATH%"
```
（若用别的构建目录，把 `build-4080s` 相应替换。）

### 第 6 步 · 冒烟验证

产物：
```
build-4080s\apps\ninfer.exe                       CLI
build-4080s\apps\ninfer-serve.exe                 HTTP 服务（OpenAI + Anthropic 双协议）
build-4080s\apps\ninfer-perplexity.exe            困惑度评测
build-4080s\apps\start-bonsai-rtx4080s.bat       启动脚本（80 SM 档；构建后自动随 exe 落位）
build-4080s\apps\start-qwen38-rtx4080s.bat       启动脚本（80 SM 档；构建后自动随 exe 落位）
```
> 启动脚本**按编译期档位各一份**（`start-bonsai-rtx4090.bat` / `start-qwen38-rtx4090.bat` 对应 128 SM，`-rtx4080s` 对应 80 SM），由 `apps/CMakeLists.txt` 的 `ninfer-serve` **POST_BUILD** 步骤按 `NINFER_TARGET_SM_COUNT` 自动把**匹配那一对**拷到 exe 同目录（脚本内 `%~dp0ninfer-serve.exe` 由此解析）。两档脚本只差 `--prefill-chunk`（1408 / 1024）。它们属于本 fork 的本地文件，缺失时不报错；本 fork 的打包脚本 `scripts/package-release-v061-sm89.ps1` 也会按档带上对应两份（上游的 v040/v050/v060 脚本保持原样，不涉及这些 bat）。

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

:: 先试：约 6.4 GiB，文本 + 视觉 + MTP（最省时间）
hf download jgamboa/Ternary-Bonsai-2-27B-NInfer-4090 bonsai2_27b_vl_mtp_q4q5.ninfer --local-dir <model-dir>

:: Qwen3.8-27B int8-prefill 版（约 19 GiB，1.7-1.9x 更快 prefill）
hf download jgamboa/Qwen3.8-27B-NInfer-4090 qwen3_8_27b_a8.ninfer --local-dir <model-dir>
```
> 每个 artifact 请用其 HuggingFace 模型卡上的校验值核对。

---

## 4. 运行（80 SM 保守参数）

**CLI 冒烟：**
```bat
build-4080s\apps\ninfer.exe <model-dir>\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-context 8192 --max-new 1024 --spec mtp --draft-tokens 2 --lm-head-draft
```
跑通会打印答案 + prefill/decode 速度、MTP 接受率、显存占用。

**起服务（按 80 SM 下调的保守档）：**
```bat
build-4080s\apps\ninfer-serve.exe <model-dir>\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --host 127.0.0.1 --port 8080 --model-id bonsai-27b ^
  --max-context 131072 --kv-capacity auto --kv-dtype rk4v4-e8 --max-concurrency 2 ^
  --prefill-chunk 1024 ^
  --spec mtp --draft-tokens 2 --lm-head-draft --ngram chain --vision
```
- 实时看板：浏览器打开 `http://127.0.0.1:8080/monitor`；指标 `/metrics`、槽位 `/slots`。
- 若遇到「CUDA Graph × batch≥2 投机解码」长时间运行崩溃，加 `--no-cuda-graph`（代价约 2.6% 吞吐）。
- 也可以直接用启动脚本（`build-4080s\apps\` 里在编译后也会有，按档自动落位对应那一对），模型路径作为**参数**：
  `start-bonsai-rtx4080s.bat <model-dir>\bonsai2_27b_vl_mtp_q4q5.ninfer`（目录也行；省略参数则依次取 `NINFER_MODEL`、`NINFER_MODEL_DIR`、`models\` 下的默认名）；模型之后的参数原样透传给 `ninfer-serve`。
- 以上是 80 SM 档（RTX 4080 SUPER）的保守参数；128 SM 档（RTX 4090）用 `build-4090\apps\` 下的二进制与 `start-*-rtx4090.bat`，参数可同样按显存余量自行调高 `--max-concurrency` 等上限。两档脚本的具体差异写在各自文件头注释里（`--prefill-chunk` 随 SM 档；4080S 的 Qwen3.8 档还把 `--max-context` 提到了 131072）。

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
能命中即说明 80 SM 档已注入所有相关 TU（128 档同理，把值改成 128）。

---

## 7. 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `cmake`/`ninja` 找不到 | 不在默认 PATH | 用「x64 Native Tools」提示符，或按 §3 第 1 步加载 vcvars64.bat |
| `find_package(FFMPEG REQUIRED)` 失败 | 未接 vcpkg 工具链 | 确认 `-DCMAKE_TOOLCHAIN_FILE=...vcpkg.cmake` 且路径存在 |
| `find_package(CURL 7.85 REQUIRED)` 失败 | vcpkg 未装/未装 curl | 确认 `VCPKG_ROOT`，让 manifest 自动装依赖 |
| 抓到了旧版 CUDA | PATH/root 未钉死 | 显式 `-DCUDAToolkit_ROOT` + `-DCMAKE_CUDA_COMPILER` 指向目标版本 |
| `requires CUDA 12.8 or newer` | nvcc 版本过低 | 同上，换成 ≥ 12.8 的 nvcc |
| `NInfer supports only CMAKE_CUDA_ARCHITECTURES=89` | 架构值不对 | 传 `-DCMAKE_CUDA_ARCHITECTURES=89` |
| `NINFER_TARGET_SM_COUNT must be an even integer >= 66` | SM 值非法 | 用 128 或 80（偶数且 ≥66） |
| 首次 configure 卡很久 | vcpkg 在编 FFmpeg | 正常，**1~2 小时**；别中断 |
| 找不到 Windows SDK | 未进入 VS 编译环境 | 用「x64 Native Tools」提示符（会正确设置 `WindowsSdkDir`） |
| 改过 CUDA 编译器后报错 | 旧 build 目录绑定了旧编译器 | 删掉 build 目录重新 configure |
| `max_context exceeds the configured position capacity` | `--max-context` 超过了 artifact 声明的位置容量（Qwen3.8 / Bonsai 都是 **262144**，因果 attention 的分块几何也以 262144 keys 为上限） | 把 `--max-context` 设在 262144 以内；要冲顶就把 `--max-concurrency` 降到 1 并去掉 host spill 档 |
| 请求期报 `context_length_exceeded`（而启动正常） | `--kv-capacity auto` 按实测显存解出的 KV 容量小于 `--max-context` | 看启动日志 `KV <tokens>, ..., auto \| pages a/b`；调低 `--max-context` / 并发，或减小 `--host-kv-mib` 对显存的占用 |
| 运行时报缺 DLL | app-local 部署没生效，或 exe 被拷离了同目录 DLL | 先 `dir <build-dir>\apps\*.dll` 看 exe 同目录是否已有依赖；确实缺了才按 §3 第 5 步的兜底命令加 PATH |
| 链接报 LIBCMT 冲突 | 手动改了链接选项 | 恢复仓库默认（`/NODEFAULTLIB:LIBCMT` 已在 CMakeLists 中） |

---

## 8. 一键脚本（可选）

把下面存成 `build-sm89.bat` 放到仓库根，双击或命令行运行：

```bat
@echo off
setlocal
rem 用法: build-sm89.bat [SM_COUNT] [BUILD_DIR]
rem   SM_COUNT  128 = RTX 4090（默认）, 80 = RTX 4080 SUPER
rem   BUILD_DIR 默认 build-sm<SM_COUNT>；想让打包脚本直接找到，可传 build-4090 / build-4080s
set "SM_COUNT=%~1"
if not defined SM_COUNT set "SM_COUNT=128"
set "BUILD_DIR=%~2"
if not defined BUILD_DIR set "BUILD_DIR=build-sm%SM_COUNT%"

if not defined VCPKG_ROOT (
  echo VCPKG_ROOT is not set. See step 0 of the manual.
  exit /b 1
)
rem CUDA_PATH 必须指向你要用的那套 CUDA（>= 12.8）；下面是官方默认安装位置，按需改。
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo nvcc not found under %CUDA_PATH%; set CUDA_PATH to your CUDA install.
  exit /b 1
)
set "PATH=%CUDA_PATH%\bin;%PATH%"
pushd "%~dp0"
cmake -S . -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=%SM_COUNT% ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" 1>configure.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if not "%ERRORLEVEL%"=="0" (type configure.log & popd & exit /b 1)
cmake --build "%BUILD_DIR%" -j 1>build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
if not "%ERRORLEVEL%"=="0" type build.log
popd
endlocal
```
> 它假定你已经在 VS 的编译环境里（「x64 Native Tools」提示符），并且 `VCPKG_ROOT` 已设置；`CUDA_PATH` 未设置时会回退到 CUDA 的默认安装位置。

两档的调用示例：

```bat
build-sm89.bat 128 build-4090     :: RTX 4090
build-sm89.bat 80  build-4080s    :: RTX 4080 SUPER
build-sm89.bat                    :: 默认 128 档，输出到 build-sm128
```

---

## 9. 验收清单

- [ ] `<build-dir>\apps\ninfer.exe`、`ninfer-serve.exe`、`ninfer-perplexity.exe` 三个 exe 均生成。
- [ ] `ninfer.exe --help` 正常；能加载 `.ninfer` 并输出文本。
- [ ] `findstr /C:"NINFER_TARGET_SM_COUNT=<档位>" <build-dir>\compile_commands.json` 命中（80 档查 80，128 档查 128）。
- [ ] dev 档 `ctest` 全绿（尤其 `ninfer_softmax_attention_test`，含 `--rk4v4-e8-only`）。
- [ ] 回归锚点（模型相关，仅当用同款 artifact 时适用）：Bonsai 2 27B 困惑度 ≈ 5.8549；长上下文检索输出 `ORCHID=493817; COLOR=COBALT`。
- [ ] 起服务连续 40 分钟不崩、`/monitor` 正常。

> 测性能前务必先**控显示税**：若测试用的显卡正在驱动显示器，高分辨率/高刷新率会明显吃掉每轮 MTP decode 吞吐（实测可达 ~18%）——把显示输出挪到核显或降低刷新率并保持画面静止后再测，否则数据不可比。

---

## 10. 发布打包（可选，两档一起出）

本 fork 是 `sm_89` 移植，`NINFER_TARGET_SM_COUNT` 是**编译期常量**，所以 4090（128 SM）与 4080S（80 SM）各需一份二进制。打包脚本一次把两档收进同一个压缩包的两个子目录。

前置：两档都已编译完成。128 档放 `build-4090`（脚本也接受默认目录 `build`），80 档放 `build-4080s`（即 §1 的 `-DNINFER_TARGET_SM_COUNT=80`）。然后：

    powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1

产物（`dist/` 已被 `.gitignore` 忽略）：

    dist\ninfer-sm89-windows-x64-0.6.1\
      sm128-rtx4090\      exe + vcpkg DLL + VERSION + start-*.bat   （RTX 4090）
      sm80-rtx4080s\      exe + vcpkg DLL + VERSION + start-*-rtx4080s.bat  （RTX 4080 SUPER）
      README.md           哪张卡用哪个子目录
      WINDOWS_PORT.md     本 fork 的移植说明
      ninfer-windows-build-manual.md
      LICENSE / VERSION
      SHA256SUMS.txt      包内所有文件的哈希
    dist\ninfer-sm89-windows-x64-0.6.1.zip
    dist\SHA256SUMS-v0.6.1-sm89.txt   zip 自身的哈希

要点：

- **只打包，不编译**：缺构建目录或 exe 会直接报错，与上游 `scripts/package-release-v0x0.*` 定位一致。
- **档位自检**：打包前用 `compile_commands.json` 断言该目录确实是 `NINFER_TARGET_SM_COUNT=128/80`，防止把 80 档标成 4090。
- 启动脚本**按档分两份**并放进**各自的子目录**（不是根目录）：`sm128-rtx4090\` 放 `start-*-rtx4090.bat`，`sm80-rtx4080s\` 放 `start-*-rtx4080s.bat`（脚本靠 `%~dp0ninfer-serve.exe` 找同目录的 exe，所以不能只放根目录一份）。
- 启动脚本的模型路径是**参数**而非硬编码：`start-bonsai-rtx4080s.bat <model.ninfer>`；不给参数时依次取 `NINFER_MODEL` → `NINFER_MODEL_DIR\<默认文件名>` → `models\<默认文件名>`，模型之后的参数原样透传（另可用 `NINFER_SERVE_ARGS`、`NINFER_SERVER`）。
- `VERSION` 由脚本写入：仓库根的 `VERSION` 是上游 3090 的标签，不适用于本包。
- 孪生的 Bash 脚本 `scripts/package-release-v061-sm89.sh` 打 Linux 版（仓库要求每个 `.ps1` 都有 `.sh` 配对）。

---

## 11. 示例：占位符在一台真实机器上的取值（非要求）

以下是作者撰写本手册时所用机器（RTX 4080 SUPER）的真实取值与**填好值的命令**。它们**只是示例**，不构成最低要求：满足 §2 的门槛即可编译。

### 11.1 占位符对应关系

| 正文占位符 | 作者机器上的值 |
|---|---|
| `<repo>` | `E:\workspaces\ai\ninfer-sm89-windows` |
| `<vcpkg-root>` | `E:\workspaces\c\vcpkg` |
| `<cuda-path>` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` |
| `<model-dir>` | `E:\LLM` |
| `<vcvars64.bat>` | `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat` |
| Windows SDK | `E:\Windows Kits\10\10.0.26100.0`（非默认位置） |

其余环境：

| 项 | 值 |
|---|---|
| 显卡 | RTX 4080 SUPER（AD103，80 SM，`sm_89`） |
| 系统 | Windows 11 x64 |
| VS | Visual Studio 2022 Community（MSVC 14.44） |
| CUDA Toolkit | 13.4（机器上另装了 12.4，构建时必须显式钉死 13.4） |
| 驱动 | 617.14 |
| CMake / Ninja | 随 VS 提供（CMake 3.31.6 / Ninja 1.12.1） |
| 构建档 | `build-4080s`（`NINFER_TARGET_SM_COUNT=80`） |

### 11.2 同一套命令，把值填进去

```bat
:: 0) vcpkg
if not exist E:\workspaces\c mkdir E:\workspaces\c
git clone https://github.com/microsoft/vcpkg E:\workspaces\c\vcpkg
E:\workspaces\c\vcpkg\bootstrap-vcpkg.bat
setx VCPKG_ROOT E:\workspaces\c\vcpkg

:: 1) 编译环境（或用开始菜单的「x64 Native Tools Command Prompt for VS 2022」）
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d E:\workspaces\ai\ninfer-sm89-windows
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"

:: 2) configure + build（80 SM 档 / RTX 4080 SUPER）
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=E:/workspaces/c/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=80 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
cmake --build build-4080s -j

:: 3) 直接运行即可：DLL 已由 app-local 复制到 build-4080s\apps\（见 §3 第 5 步）
::    仅当 exe 被拷到别处 / app-local 被关掉时，才需要这行兜底：
:: set "PATH=E:\workspaces\ai\ninfer-sm89-windows\build-4080s\vcpkg_installed\x64-windows\bin;%CUDA_PATH%\bin;%PATH%"
```

一键脚本的等价调用（§8，可选）：

```bat
build-sm89.bat 128 build-4090     :: RTX 4090
build-sm89.bat 80  build-4080s    :: RTX 4080 SUPER
```

> 该机器上 CUDA 双版本共存、Windows SDK 非默认位置，这两点最容易踩坑：前者必须用 `-DCUDAToolkit_ROOT` + `-DCMAKE_CUDA_COMPILER` 钉死，后者用 VS 的编译环境提示符即可自动带上。通用处理方式见 §2 / §3 第 1 步 / §7。