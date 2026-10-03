# NInfer 4080S（80 SM）重定向 — 设计与执行计划（Review 稿）

> 状态：**待 review**（review 通过后进入 Code 模式实施）
> 范围：把本 fork 从「编译期按 128 SM（RTX 4090）标定 attention wave 几何」改为「128 / 80 双档可选」，在 80 SM 的 **RTX 4080 SUPER** 上吃掉 attention 尾波税，同时**不牺牲 4090 的正确性与可复现性**。
> 依据：本仓源码逐行核实，所有结论标注 `文件:行号`。仓库自带权威文档：`WINDOWS_PORT.md`、`HANDOFF.md`、`AGENTS.md`、`README.md`。

---

## 0. 摘要（一句话）

运行时调度早已自适应（不崩卡）；**唯一需要动的编译期 SM 假设只有 `kTargetSmCount` 一个常量、4 个消费点**。本计划新增 CMake 开关 `NINFER_TARGET_SM_COUNT`（默认 128），使 80 SM 成为可构建的第二档，配套两条编译期 `static_assert` 兜底，并按「先 A 后 B」阶梯做正确性 + 性能验证。

---

## 1. 背景与动机

- 本 fork 仅编译 `sm_89`（`CMakeLists.txt:6-13` 强制），4080 SUPER 与 4090 同架构，二进制可直接运行。
- 但 attention 的小 T（causal_cache）wave 几何是**编译期常量** `kTargetSmCount = 128`（`device.h:28`），在 80 SM 上会跑成 ~1.6 个 wave，尾波不满 → prefill/decode 达不到 4090 的"整波填满"数字。
- 这正是"自编译的真正价值"里价值最高的一条（见 `ninfer-4080s-two-routes.md` §6）。本次把它落成**可维护、可 A/B、零 4090 回归**的工程改动。

---

## 2. 已核实源码事实（地基）

### 2.1 编译期 SM 字面量只有一个 —— `kTargetSmCount`

| 位置 | 内容 | 性质 |
|---|---|---|
| `src/core/device.h:28` | `inline constexpr int kTargetSmCount = 128;` | 定义 |
| `src/ops/softmax_attention/dense/causal_cache/geometry.cuh:25` | `SmallTWaveSplits = 2 * kTargetSmCount / KVHeads` | constexpr 模板成员 |
| `src/ops/softmax_attention/dense/causal_cache/small_t.cu:71` | 主机侧 `kMax = kTargetSmCount / KVHeads` | 主机 wave 策略 |
| `src/ops/softmax_attention/dense/causal_cache/small_t.cuh:114` | **设备镜像** `kMax = kTargetSmCount / KVHeads` | `__device__` |
| `src/ops/softmax_attention/dense/causal_cache/small_t.cu:239` | `kOneWave = kTargetSmCount - kTargetSmCount/16` | 主机 batch grid 上限 |

> **关键不变量**：`small_t.cu:71`（主机）与 `small_t.cuh:114`（设备）**必须逐位一致**（源码注释："Must agree bit-for-bit"）。二者同读一个常量 → **只改该常量即可保持同步**，排除失配风险。
> 除此之外，全仓**没有**任何 live 的 `170`/`128` SM 字面量（`170` 只出现在注释：`rmsnorm.cu:18`、`small_t.cu:236`）。

### 2.2 运行时路径早已自适应（无需改动）

`device_sm_count()`（`device.cu:81-90`，缓存式查询 `cudaDevAttrMultiProcessorCount`）已有 8 个消费点：
`rmsnorm.cu:20`、`rope.cu:21`、`sparse_moe_prefill_kernels.cu:297`、`gated_delta_net/chunked/output.cu:13`、`q5_linear_add_gemm_mma.cu:127`、`t5_project.cu:47,142`、`rope_bench.cu:218`。

### 2.3 协作启动预算完全走运行时（`resident_ctas_27/35` 不参与运行时）

- `resident_ctas_27`（`plan.cpp:75-76`）、`resident_ctas_35`（`plan.cpp:79-83`）**仅出现在 `static_assert`**（`plan.cpp:121-123`，`catalog_is_resident`）。
- 运行时协作预算走 `multiprocessor_count * kResidentCtasPerSm`（`kernels.cu:360-364`），`multiprocessor_count` 来自 `DeviceExecutionView`；超预算时**按 token 区间自动分块**（`kernels.cu:375-395`）；连一个 tile 都放不下时返回 `false` → 上层回退 `MmaUnsplit`（`plan.cpp:253-254`）。
- **80 SM 实例**：27B `MmaCooperativeSplit8`，`resident_ctas = 80×2 = 160`，`kCtasPerTokenTile = (48/16)×8 = 24`，`max_token_tiles = 160/24 = 6`；cols=1280 时 `total_token_tiles = 10 > 6` → 拆成 768+512 两次，各自 grid 144 / 96 ≤ 160 ✅。
  → **在 80 SM 上：路由表选中协作 schedule，运行时自动分块，绝不越界、不崩卡。**

> 结论：**安全层面 fork 自身已解决；本次只把"编译期几何"从固定 128 改为可配。**

---

## 3. 设计

### 3.1 目标 / 非目标

- **目标**：80 SM 可构建、可 A/B；默认 128 档逐位不变；无线程/数值行为回归。
- **非目标**：CUDA Graph × 批量投机专项；`resident_ctas` 深度调优（见 §3.5）；多卡/多设备；改动模型加载或数值路径。

### 3.2 方案选型

| 方案 | 做法 | 优点 | 缺点 | 建议 |
|---|---|---|---|---|
| A 硬改字面量 | `device.h:28` → `80` | 最小 | 只为 4080S 优化、4090 变差、无法 A/B | 备选 |
| **B CMake 变量** | 新增 `NINFER_TARGET_SM_COUNT`（默认 128）生成宏 | 双档可选、可 A/B、仍 sm_89、不动几何逻辑 | 需一处 CMake 接线 | ✅ **首选** |
| C 运行时注入 | geometry/small_t 读 `device_sm_count()` | 单二进制自适应 | `__device__` 模板无法查运行时；需全链路传参；破坏 "bit-for-bit" 简洁性，风险最高 | 不采纳 |

**选 B**：沿用作者"编译期常量、单架构"的设计口径（`device.h:22-27` 注释），只把常量换成可配项。

### 3.3 精确改动（3 处代码 + 1 处可选 preset）

**改动 1 — `src/core/device.h`（替换第 28 行）**
```cpp
#ifndef NINFER_TARGET_SM_COUNT
#define NINFER_TARGET_SM_COUNT 128
#endif
// 编译期镜像：默认 RTX 4090 的 128；RTX 4080 SUPER 用 -DNINFER_TARGET_SM_COUNT=80 生成。
// 合法域见 causal_cache/geometry.cuh 的 static_assert（必须为偶数且 >= 66）。
inline constexpr int kTargetSmCount = NINFER_TARGET_SM_COUNT;
```

**改动 2 — 根 `CMakeLists.txt`（`project()` 之后，与 MSVC 选项同区）**
```cmake
set(NINFER_TARGET_SM_COUNT 128 CACHE STRING
    "Compile-time SM count for attention wave geometry (128 = RTX 4090, 80 = RTX 4080 SUPER)")
math(EXPR NINFER_TARGET_SM_PARITY "${NINFER_TARGET_SM_COUNT} % 2")
if(NINFER_TARGET_SM_COUNT LESS 66 OR NOT NINFER_TARGET_SM_PARITY EQUAL 0)
  message(FATAL_ERROR
    "NINFER_TARGET_SM_COUNT must be an even integer >= 66; got ${NINFER_TARGET_SM_COUNT}")
endif()
add_compile_definitions(NINFER_TARGET_SM_COUNT=${NINFER_TARGET_SM_COUNT})
```
> 用全局 `add_compile_definitions`（与现有 `/utf-8` 等同级），确保所有包含 `device.h` 的 TU（`ninfer_core` / `ninfer_ops`）可见。

**改动 3 — `causal_cache/geometry.cuh`（`CausalAttentionGeometry` 之后补自检）**
```cpp
static_assert(kTargetSmCount % 2 == 0 && kTargetSmCount >= 66,
              "kTargetSmCount must be even and >= 66 so both head geometries can stage 262144 keys");
```
（与 CMake 校验互为双保险：CMake 报错友好，编译期 assert 保证语义不漂移。）

**改动 4（可选）— `CMakePresets.json` 新增档位**
```json
{ "name": "release-4080s", "inherits": "release",
  "cacheVariables": { "NINFER_TARGET_SM_COUNT": "80" } }
```

### 3.4 不变量与合法域（数学验证）

`geometry.cuh` 既有两条 `static_assert` 决定合法区间：

1. `SmallTWaveSplits % SmallTSplitScale == 0`
   - `CausalD256H24Kv4`（KV=4, scale=1）：`SM/2` → 恒成立
   - `CausalD256H16Kv2`（KV=2, scale=2）：`SM` → 要求 **SM 为偶数**
2. `SmallTMaximumSplits * 3968 >= 262144` → `2*SmallTWaveSplits >= 66` → `SmallTWaveSplits >= 33`
   - H24Kv4：`SM/2 >= 33` → **SM >= 66**；H16Kv2：`SM >= 33`

→ **合法域 = 偶数且 ≥ 66**。代入 **SM=80**：

| 几何 | SmallTWaveSplits | SmallTMaximumSplits | 覆盖键数 | 判定 |
|---|---:|---:|---:|---|
| H24Kv4 | 40 | 80 | 80×3968 = 317,440 ≥ 262,144 | ✅ |
| H16Kv2 | 80 | 160 | 160×3968 = 634,880 ≥ 262,144 | ✅ |

**两条 assert 均成立 → 编译安全，256K 上下文不缩水。**

### 3.5 `resident_ctas_27/35` 处理决策

- **v1 不动**：仅服务于 `static_assert` 自检；运行时预算由 `multiprocessor_count` 自适应（§2.3）；改了反而可能让 `static_assert` 与 128-SM 路由表语义脱钩。
- **v2（仅在性能剖析证实"协作分块次数"成为瓶颈时）**：优先下调 `k27Routes`/`k35Routes` 上界（`plan.cpp:40-52`）以减少 80 SM 上的分块次数；`resident_ctas_27/35` 与 `static_assert` 必须**成对修改**。本次不做。

### 3.6 兼容性与回退

- 默认档维持 128 → **现 4090 构建行为逐位不变（零回归）**。
- 编译档与运行卡不匹配（128档跑80卡 / 80档跑128卡）**两个方向都安全**：前者几何偏大→尾波，后者几何偏小→欠填充，均不越界。
- 回退 = 重新 configure 时去掉 `-DNINFER_TARGET_SM_COUNT`（默认 128）。

---

## 4. 构建计划（先 A 后 B 阶梯）

| 阶段 | 动作 | 通过标准 | 成本 |
|---|---|---|---|
| **0 基线** | 路线 A：下预编译 zip（文档记 `v2026.09.27b`，**SHA/大小本机离线未核验**）→ CLI 冒烟 + 记录 prefill/decode tok/s | `ninfer.exe` 出文本；记下数字作对照 | ~1 h |
| **1 工具链** | 装 vcpkg 到 `E:\workspaces\c\vcpkg`，`setx VCPKG_ROOT`。**修正既有文档矛盾：configure 的 `-DCMAKE_TOOLCHAIN_FILE` 必须与实际安装路径一致（文档里一处写 E 盘装、一处写 `C:/vcpkg`）** | `vcpkg version` 可用 | 1-3 min |
| **2 原生编译（默认 128）** | 用 `README.md`/`WINDOWS_PORT.md` 的干净命令（**不用 `winport_*.bat`**，其硬编码 VS18 + `E:\LLM\ninfer-4090-winport`）：`-DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON` | 编出 3 个 exe + 测试/bench；`ctest` 全绿 | vcpkg 1-2 h + 编译 20-40 min |
| **3 应用重定向（80）** | `-DNINFER_TARGET_SM_COUNT=80` 重新 configure + build（独立 `-B build-4080s`） | 编译通过（§3.4 assert 全过） | 20-40 min |
| **4 正确性验证** | 见 §5.1 | 全绿 | ~40 min |
| **5 性能 A/B** | 见 §5.2 | 有可归因提升 | 半天 |
| **6 可选深调** | §3.5 v2 / CUDA Graph 专项 | 见 §6 | 1-2 天 |

**参考 configure 命令（本机路径，SM=80 档）：**
```bat
:: 在「x64 Native Tools Command Prompt for VS 2022」中
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=E:/workspaces/c/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DNINFER_TARGET_SM_COUNT=80 ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
cmake --build build-4080s -j
```

---

## 5. 验证计划

### 5.1 正确性（数值与协议，80 档必须全过）

- **C++ kernel 单测**（每颗 kernel 对照 FP32/FP64 oracle，位于 `tests/ops/`）：
  `ninfer_softmax_attention_test`（**完整 + `--rk4v4-e8-only`**）、`ninfer_kv_cache_append_test`、`ninfer_linear_t5_test`、`ninfer_attn_input_proj_test`、`ninfer_linear_q4_a16_test`、`ninfer_linear_q5_a16_test`、`ninfer_device_test`。
  > 这些是 **oracle 对比**（非 grid 快照），改 wave 几何**不应**影响 PASS；若失败即为真 bug，**禁止放宽阈值**（遵循 `HANDOFF.md` §1 的"correctitud antes que velocidad"）。
- **回归锚点**（来自 `HANDOFF.md`）：
  - Bonsai quick 困惑度 **≈ 5.8549**；
  - Qwen3.8 int8 短上下文文本 **md5 不变**；
  - 长上下文检索 **`ORCHID=493817; COLOR=COBALT`**（64K/128K，int8 与 `rk4v4-e8` 各一遍）。
- **档位一致性**：80 档与 128 档在**相同采样种子**下输出 md5 应一致（几何只影响调度，不影响数值）。

### 5.2 性能 A/B（唯一有意义的对照，必须控显示税）

- **内核级**：`ninfer_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4 --kv-dtype int8 --batch 1 --tokens 1024 --context 8192,32768,65536,131072 --mapping fragmented --execution eager --cache cold --warmup 5 --repeat 21`（`int8` 与 `rk4v4-e8` 各一遍）。
- **服务端**：`tools/bench/run_serve_concurrency.py`（1/2/3 lane，`--kv-capacity auto`），Bonsai + Qwen3.8-27B。
- **控变量**：显示输出挪到核显或 60 Hz 静止（`HANDOFF.md`：桌面合成器会偷时间）；**old,new,new,old 交替**抵消热/时钟漂移；差异 < 噪声（±1 ms/round）就写"噪声内"；配合 nsys/ncu 看**单 kernel 时间**而非仅 ms/round。
- **预期**：长上下文 attention 尾波减少 → prefill/decode 改善；**短上下文（≤8K）应无变化，也不应回退**。

---

## 6. 验收标准

1. 默认（128）构建与改前**逐位一致**（无回归）。
2. `-DNINFER_TARGET_SM_COUNT=80` 可配置、可构建、§3.4 断言全过。
3. 80 档下 §5.1 全部正确性测试**全绿**，困惑度/检索锚点命中。
4. 80 档 vs 预编译基线（128 档）：长上下文 prefill/decode 有**可归因**提升；短上下文无回退。
5. 连续 40 min 服务不崩、无整卡 reset。

---

## 7. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| vcpkg 编 FFmpeg 失败/超时 | 中 | 高 | E 盘装 vcpkg（`buildtrees/packages/downloads` 吃 10-20 GB）；`E:\Windows Kits` 非默认位置需用 vcpkg 的 x64-windows 工具链；预留 1-2 h |
| 文档 vcpkg 路径自相矛盾 | 高 | 中 | 安装路径与 `-DCMAKE_TOOLCHAIN_FILE` 必须一致（§4 阶段 1 已修正） |
| 改常量后主机/设备分块失配 | 低 | 高 | 两处同读一个宏 + 新增 `static_assert` 兜底（§3.3/3.4） |
| 性能"提升"落在噪声内 | 中 | 中 | old/new 交替；nsys/ncu 单 kernel 时间 |
| CUDA Graph × batch≥2 投机长跑崩 | 中 | 中 | 先 `--no-cuda-graph`（代价 ~2.6%）；Graph 修复留专项（不在本次范围） |
| 编译档与运行卡不匹配 | 低 | 低 | 两个方向均安全（§3.6） |

---

## 8. 成本预算

vcpkg 首编 1-2 h + 原生编译 20-40 min + 80 档编译 20-40 min + 正确性 ~40 min + 性能 A/B 半天 ≈ **1.5 天**（不含 §7 可选深调）。

---

## 9. 交付物

- **代码**：`src/core/device.h`、根 `CMakeLists.txt`、`causal_cache/geometry.cuh` 三处改动（+ 可选 `CMakePresets.json`）。
- **文档**：
  - 更新 `WINDOWS_PORT.md` 的 Building 段，新增 `-DNINFER_TARGET_SM_COUNT=80` 说明；
  - 把 §3.4 合法域写进 `docs/maintainer/build-system.md` 的配置表。
- **数据**：`profiles/bench/` 下 128 vs 80 的 A/B 记录（对照 `HANDOFF.md` 的登记规范写入 `docs/maintainer/bonsai-ternary-design.md` §9.1 或 `WINDOWS_PORT.md`）。

---

## 10. 顺带更正（对既有两份调研文档）

复核发现（不影响本计划，建议一并修订）：

1. `ninfer-4080s-two-routes.md` "全仓搜 `GeForce` 只有两处" **不成立**（实际 29 处）；且仓库**确有**依赖设备名的映射函数 `context_cost_hardware_class()`（`context_cost.cpp:522`，由 `model_instance.cpp:175-176` 运行时调用）。但其结论（4080S 不被拒绝）成立：该函数对任意设备名都产出 slug，仅影响成本档位，回落 `generic_context_prefill_cost()`。
2. "§1.4 那三处 `kTargetSmCount`" **实为四处**（漏掉设备镜像 `small_t.cuh:114`）。
3. 行号小错：CUDA ≥ 12.8 在 `CMakeLists.txt:49-53`（文档写 `:56`）；`sm_89` 校验在 6-13 行（文档写 3-14）。
4. `winport_configure.bat` 的 `-DCMAKE_TOOLCHAIN_FILE=E:/LLM/vcpkg/...` 与其他文档的 `C:/vcpkg` 冲突；`route-b` 文档自身"装到 E 盘 / 配 C 盘"矛盾。

---

## 11. 待 review 的决策点（请确认）

| 编号 | 决策 | 建议默认 |
|---|---|---|
| **D1** | 采用方案 B（CMake 变量）还是 A（硬改 80）？ | **B** |
| **D2** | 默认档保持 128，还是直接改默认 80？ | **保持 128**（零回归） |
| **D3** | 是否新增 `release-4080s` preset？ | **是**（方便 A/B） |
| **D4** | 本次是否顺带做 `resident_ctas` v2 调优？ | **否**（等实测数据） |
| **D5** | 文档落地位置 / 命名是否按 §9？ | **按 §9** |

---

## 12. 实施清单（Code 模式用，review 通过后执行）

- [ ] `src/core/device.h`：`kTargetSmCount` 改为读 `NINFER_TARGET_SM_COUNT` 宏（默认 128）。
- [ ] 根 `CMakeLists.txt`：新增 `NINFER_TARGET_SM_COUNT` cache 变量 + 合法性校验 + `add_compile_definitions`。
- [ ] `causal_cache/geometry.cuh`：新增偶数 & ≥66 的 `static_assert`。
- [ ] `CMakePresets.json`：新增 `release-4080s`（D3=是时）。
- [ ] 默认档（128）构建 + `ctest` 全绿，确认无回归。
- [ ] 80 档构建 + §5.1 正确性全绿 + §5.2 A/B 记录。
- [ ] 更新 `WINDOWS_PORT.md` / `docs/maintainer/build-system.md`。