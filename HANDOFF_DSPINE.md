# 交接：把 DSpine 草稿模型接进 NInfer（`feat/dspine-draft`）

> 临时工作计划，给用户 Windows/4090 机器上的下一个会话。全部任务关闭后删除本文件
> （AGENTS.md：临时计划只服务在办工作）。用户用中文交流，回答用中文；数字凡是没在 GPU 上测过的，
> 必须说成估算。

---

## 0. 一句话状态

这件事有**两层**，别混：

1. **权重侧（不需要显卡）已经做完**：转换器能把 `Qwen3.8-27B-DSpine-b16` 完整映射成 `dspine`
   组件，用真实 checkpoint 逐张量对拍通过；顺带在同一台无卡机器上产出了三份可发布的 `.ninfer`
   产物（§2.5），组成与已发布的 jgamboa 产物一致。
2. **运行时侧（缺的是代码，不是显卡）一行没动**：`SpeculativeBackend::DSpine`、配置解析、因果
   块 mask、bridge 与精修两个 op、18 处引擎接线。没有 GPU 的真实限制不是「写不了」，而是
   **能编译不能运行**——bridge 那 8 处 BF16 cast 只能靠逐层对拍确认，所以这些代码必须在有
   nvcc + 4090 的会话里写。

下一个会话从「T2 C++ 配置与绑定」开始，数学契约见第 5 节，不要自己重新推导。

---

## 1. 分支与提交

```
feat/dspine-draft
├─ 0916420a  feat(launcher): 新增模型下载页与自定义参数        <- origin/main 顶端
├─ 477415fe  feat(rtx5060ti): sm_120a / 36-SM 支持
├─ 0c3f7ccf  fix(launcher): 5060 Ti 思考预算
├─ 83da6b55  fix(launcher): 统一 Bonsai 预设预算               <- 已合并 feat/rtx5060ti-bonsai
├─ 458a7849  feat(convert): convert the Qwen3.8-27B DSpine draft as a dspine component
├─ c56db4c9  docs: hand off the DSpine draft integration state
├─ ddb04401  docs: record the pushed branch in the DSpine handoff
├─ 4f4f83ef  test(convert): write and read back a dspine component end to end
└─ (本文件更新)
```

分支已 push 到 `origin/feat/dspine-draft`。本地 `main` 仍停在 `0916420a`，没有动。

---

## 2. 已完成（可复现）

改动文件：

| 文件 | 内容 |
|---|---|
| `tools/convert/qwen3_5.py` | `draft_config()` 支持 `backend="dspine"`；`_dspine_bridge()` 校验并归一化 bridge 配置；`_Builder.dspine_bridge()` / `dspine_shared()` 映射张量；`build_model()` 接受 `dspine` |
| `tools/convert/official_recipes.py` | `_optional()`：backbone 取 Q8、`*/bridge/*` 强制 BF16；`dspine` 也共享 `context_key/value`；最后一层 `bridge/read` 与 `bridge/final_read` 共享 |
| `tools/convert/proposal.py`、`tools/convert/__main__.py` | 组件列表/CLI 帮助加入 `dspine`（`--source dspine=PATH --components text,dspine`） |
| `tests/convert/test_qwen3_5.py` | 新增 3 个测试：配置归一化 + 逐层 bridge 切片、8 个非法配置用例、recipe 量化策略 |
| `tests/convert/test_cli.py` | 端到端：真的写出一个含 `dspine` 组件的 `.ninfer` 再读回（bridge BF16、backbone Q8、read 别名共用同一存储对象） |
| `docs/weight-conversion.md` | 说明 `dspine` 组件与其 BF16 约束，并注明运行时尚未选择它 |

验证证据：

```bash
python3 -m pytest tests/convert -q          # 77 passed, 4 skipped
```

真实 checkpoint 对拍（下载了 `model.safetensors` 4.05 GB + 目标 config）：

```
parameters: 84                      # 逻辑参数（含 10 个 context K/V 共享 + 1 个 read 共享）
logical elements: 2083059456
slice checks OK                     # early_read / feature_write 逐层切片、fc、codebook、final_read 全等
```

生成的组件配置（写入 artifact 的 `components.dspine.config`）：

```json
{
  "architectures": ["DFlashDraftModel"],
  "model_type": "qwen3",
  "intermediate_size": 17408, "num_attention_heads": 32, "num_key_value_heads": 8,
  "head_dim": 128, "num_hidden_layers": 5, "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000000.0},
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                  "sliding_attention", "sliding_attention"],
  "sliding_window": 8192, "vocab_size": 248320,
  "dflash_config": {
    "target_layer_ids": [1, 16, 31, 46, 61],
    "mask_token_id": 248070,
    "draft_block_causal": true,
    "block_size": 16,
    "code_bridge": {
      "code_size": 1024, "rank": 512,
      "eps": 9.999999974752427e-07, "kappa": 10.0,
      "feature_layers": [0, 1, 2, 3, 4], "head_topk": 16
    }
  }
}
```

物理体积（实测由张量头算得）：

| 段 | 元素 | BF16 |
|---|---:|---:|
| HF 原始文件 | 2,025,912,068 | 4.05 GB |
| 本组件（去重后，不含 4 个 version 标量、不含恒零 terminal `out.weight`） | **2,025,387,780** | **3.77 GiB** |
| 其中 bridge（codebook 254,279,680 + read/write/pred/gate…，必须 BF16） | 295,174,656 | 563 MiB |
| 其中 backbone（`fc` + 5 层，可 Q8 → 约 1.7–1.85 GB） | 1,730,213,124 | 3.22 GiB |

---

## 2.5 三份产物：本会话已在本机产出（可作为发布基线）

三份都在无 GPU 的机器上用 `--device cpu` 转换完成。**组成与已发布的 jgamboa 产物一致，文件尺寸
逐字节相同**（19.03 / 19.03 / 6.40 GiB）；内容没有跟已发布文件做逐字节 diff——转换是确定性的，
但没有直接比对过。源权重全部按 HF 的 `lfs.oid` 校验过 sha256。

| 仓库（用户已建） | 文件 | 字节 | 尺寸 | 转换耗时 |
|---|---|---|---:|---:|---:|
| `zjwan461/qwen3-8-27b-int8-ninfer` | `qwen3_8_27b_a8.ninfer` | 20,437,512,960 | 19.03 GiB | 351 s |
| `zjwan461/swift-1-5-27b-ninfer` | `swift15_27b_a8.ninfer` | 20,437,520,896 | 19.03 GiB | 326 s |
| `zjwan461/bonsai2-27b-ninfer` | `bonsai2_27b_vl_mtp_q4q5.ninfer` | 6,869,412,096 | 6.40 GiB | 348 s |

sha256（本机算出，上传后用来核对）：

```
62c98d47704200c83ca5aa04e42358f1daefcb3840cef15c147bc57e7a44e8ff  qwen3_8_27b_a8.ninfer
28c099be84606c7f328faf8e4e6666c8d488333c7ce65222a69b7c20d3ad38b5  swift15_27b_a8.ninfer
e9e6f73db9d59fc509b49cd04fd409d7d757c1e61ab4cd8bc313fd014c2cb9f4  bonsai2_27b_vl_mtp_q4q5.ninfer
```

组成（`tools.artifact.inspect` 实测）：

| 产物 | components | 张量 | 格式 |
|---|---|---:|---|
| qwen3.8 a8 | text, vision, mtp, dflash2 + proposal(131072) | 1184 | q4 183 / q5 246 / q6 1 / q8 30 / bf16 627 / fp32 96 |
| swift a8 | text, vision, mtp, dflash2 + proposal(131072) | 1184 | 同上 |
| bonsai | text, vision, mtp + proposal(131072) | 1058 | t5_g128 258（三元父对象）/ q4 57 / q5 57 / q8 3 / bf16 585 |

**三份都不含 `dspine`**（`dspine/*` 绑定数为 0），理由与后续加法见 §7 决策 6。

构建命令（云端复现用；本机路径换成你的）：

```bash
# 1) Qwen3.8-27B int8
python3 -m tools.convert --model /tmp/src-qwen --recipe qwen3_8_27b_a8 \
  --source dflash2=/tmp/src-dflash2 --components text,vision,mtp,dflash2 \
  --proposal --name qwen3.8-27b --out /tmp/out/qwen3_8_27b_a8.ninfer --device cpu

# 2) Swift 1.5 int8（同一 recipe，换源）
python3 -m tools.convert --model /tmp/src-swift --recipe qwen3_8_27b_a8 \
  --source dflash2=/tmp/src-dflash2 --components text,vision,mtp,dflash2 \
  --proposal --name swift-1.5 --out /tmp/out/swift15_27b_a8.ninfer --device cpu

# 3) Bonsai 2 27B（两步：先建 base，再烤）
python3 -m tools.convert.bonsai_base \
  --gguf .../Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --reference /tmp/out/qwen3_8_27b_a8.ninfer \
  --mmproj .../Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --out /tmp/bonsai2-27b
python3 -m tools.convert --model /tmp/bonsai2-27b --recipe bonsai2_27b_mtp_q4q5 \
  --components text,vision,mtp \
  --source gguf=.../PTQ1_0.gguf --source mmproj=.../mmproj-Q8_0.gguf \
  --source mtp=/tmp/out/qwen3_8_27b_a8.ninfer \
  --proposal --name bonsai2-27b --out /tmp/out/bonsai2_27b_vl_mtp_q4q5.ninfer --device cpu
```

两个用实测才定下来的细节：

- **Bonsai 不能带 `dflash2`**。照 `docs/maintainer/bonsai-ternary-conversion.md` 的命令加
  `--source dflash2=` 会得到 8.47 GiB；去掉（只留 text+vision+mtp+proposal）正好 **6.40 GiB**，
  与已发布产物一致，也解释了 `tools/launcher/config.py` 里"Bonsai 只支持 `--spec mtp`"。
- **`--proposal` 要带**。不带是 6.07 GiB，比已发布的小 0.33 GiB——正好是 131072 行索引头的体积。

校验链（无一靠猜）：

| 对象 | 校验 | 结果 |
|---|---|---|
| `Qwen/Qwen3.8-27B` | 32 个文件 sha256 vs `lfs.oid` | 32/32 通过 |
| `ukisai/Swift-1.5-Qwen3.8-27b` | 同上 | 35/36 通过（缺的是一份 benchmarks json，不参与转换） |
| `z-lab/Qwen3.8-27B-DFlash2` | 同上 | 权重与 config 2/2 通过 |
| `prism-ml` PTQ1_0 / mmproj | sha256 | 均与 Hub 一致 |
| 三份产物 | `tools.artifact.inspect` | 组件/绑定/用途/格式/布局全部可解析 |

**没有验证的**：引擎实际加载。本机没有 CUDA，`inspect` 只证明文件框架与绑定表自洽，不证明
4090 上能跑。这一点必须在 T1/T7 里补，不要当成已验证。

---

## 3. 没做的，以及为什么（别把"没卡所以做不了"当成理由）

本机**没有 GPU 只挡住了运行时实现，没挡住权重与产物**——三份 `.ninfer` 就是无卡产出的（§2.5）。

| 项 | 能不能在无卡机器上做 | 为什么没做 |
|---|---|---|
| 权重映射、组件转换、产物打包 | **能，已完成** | — |
| 把 dspine 组件烤进产物 | **能，随时**（§7 决策 6 有命令） | 引擎选不中它，白加 3.77 GiB × 3；等 T2–T6 落地后一次重烤更划算 |
| `SpeculativeBackend::DSpine` 与引擎接线 | 写得出，**编不过** | 18 个文件 + switch 分支，本机没有 nvcc；不可编译的代码只会拖慢下一个会话 |
| `config.cpp` 的 dspine 解析 | 同上 | 没有枚举/组件入口就是死代码（AGENTS.md 不允许占位实现） |
| 因果块 mask、bridge op、refinement op | **能编译、不能运行** | 这类 op 的正确性只能靠逐层 dump 对拍；没有卡写出来就是 2000 行猜测 |

补一句准确的边界：装 nvcc 本身不需要 root 也做得到（runfile 解压即可），但**没有 GPU 就只能
编译、不能跑对拍**。bridge 的 8 处 BF16 cast、f32 tiny floor、错位门控、`√C` 缩放，恰恰只有
对拍能发现错。所以这不是"环境缺卡所以整件事做不了"，而是"缺卡所以最后那一段必须换机器"。

---

## 4. 下一会话的任务清单（按依赖顺序）

### T1 构建基线（先做，半天）

```bat
git fetch <bundle> feat/dspine-draft:feat/dspine-draft
git checkout feat/dspine-draft
CMakePresets.json 里选 4090 预设（NINFER_TARGET_SM_COUNT=128）
cmake --build build-4090 -j
python -m pytest tests/convert -q     :: 必须 77 passed, 4 skipped
```

### T2 C++ 配置与权重绑定（1–2 天）

1. `include/ninfer/types.h:75` 加 `DSpine`（**放最后**，避免改变已有枚举值）。
2. `src/models/qwen3_5/config.h`：

```cpp
struct DSpineCodeBridge {
    std::uint32_t code_size = 0, rank = 0, head_topk = 0;
    float eps = 0.F;
};
struct DSpineConfig {
    std::uint32_t block_size = 0;
    DSpineCodeBridge bridge;
};
// DraftConfig 增加：
std::optional<DSpineConfig> dspine;
```

3. `src/models/qwen3_5/config.cpp:262 draft()`：把 `bool dflash2` 换成 `DraftBackend`
   枚举；`dspine` 分支要求 `dflash_config` 成员为
   `{target_layer_ids, mask_token_id, draft_block_causal, block_size, code_bridge}`，
   可选 `kappa`；校验 `head_topk` 是 2 的幂且 ≤64、`feature_layers == 0..L-1`、
   `block_size ∈ [2,64]`（Python 侧 `_dspine_bridge()` 是同一份校验，照抄语义即可）。
4. `src/models/qwen3_5/weights.h`：`DraftBlockWeights` 加
   `std::optional<DraftBridgeWeights> bridge;`（`WeightId read, write;`）；
   `DraftWeights` 加 `std::optional<DSpineBridgeWeights> dspine;`
   （`WeightId codebook, gate, gate_bias, predecessor, value, final_read;`）。
5. 新文件 `src/models/qwen3_5/load/dspine.cpp`，签名照 `bind_dflash2`：

```cpp
void bind_dspine(Bindings&, DraftWeights&, const DraftConfig&, const TextConfig&);
```

   在 `load/dflash.cpp:bind_draft()` 末尾按 `if (config.dspine)` 调用；
   `src/models/qwen3_5/loading_sources.cmake` 加一行。
   bridge 张量一律 `b.direct(...)`（BF16），backbone 用 `b.parameter(...)`。
   逻辑名见第 6 节。

### T3 因果块注意力（1–2 天）

`include/ninfer/ops/sliding_window_attention.h` 的掩码加「块内因果」模式：

```
允许(p) = (|p - (F+i)| < S)  &&  (p < F  或  p <= F+i)
```

同时检查 `softmax_attention`（full 层）与 `graph_profiles` 的宽度桶。
验收：同一 prompt 下 K=7 与 K=15 的前 7 个 proposal 必须完全一致（因果 ⇒ 前缀稳定）。

### T4 bridge 两个 op（4–6 天，最容易错）

`dspine_bridge_read`：

```
输入 residual[H,15,B]、read[C,H]、codebook[V,C]、packed[3R,C]、anchor ids
输出 q,s,v [3R,15,B] + stash{raw, rms, packed}
```

`dspine_bridge_write`：

```
输入 raw, rms, q,s,v, write[H,R], next_norm[H]
输出 normalized_next_input[H]
```

数学契约见第 5 节，**逐条照抄，不要「顺手写得更准」**。

### T5 serial last-write 精修（4–6 天）

一个新 op `dspine_serial_refine`：top-16 候选（复用 `ops::linear_topk`）→
按候选前驱精确重做最后一层 write → `[15,16,16]` 转移表 → 贪心沿表走 / 采样沿表走并保留 q。
第一版**不要做全词表 `TG/TSV` 预计算表**（见第 7 节）。
需要补一个「按 token id 取 target head 行」的小 op（现在只有 `linear_topk` 与索引 head）。

### T6 引擎接线（3–5 天）

`grep -rl 'SpeculativeBackend::' src include tools` 就是清单（18 个文件）。
重点：`startup.cpp`（K≤15、块宽 16、draft ring 8192）、`graphs.cpp`、
`round_buffers`、`commit/capture/materialization`、`target_verification.cpp`（新分支）、
`gdn_replay` 的 record 容量（16 列）。

### T7 验证与基准（3–5 天）

1. CPU/PyTorch 参考 → CUDA 逐层对拍 `codes` / `gate` / `delta` 的 max-abs-diff。
2. 与官方 MLX 端口对拍 draft tokens（贪心，温度 0）。
3. `docs/performance/qwen3.8-27b.md` 加 run 行，报 prefill 与 decode 两侧。

---

## 5. 数学契约（以官方 MLX/SGLang 为准，勿自行简化）

### 5.1 bridge 写回（每层一次）

```text
raw  = 该层 MLP 输出（BF16）
f    = raw.float()
rms  = sqrt(max(mean(f*f), 1.1754943508222875e-38))      // f32 tiny，不是 rms_norm_eps
n    = (f / (rms + eps)).bf16                            // eps 加在分母
proj = (n @ W_read[layer].T).float()                     // layer<L-1 用 early_read_weights[l]
code = (sqrt(C) * proj / (||proj||_2 + eps)).bf16        // C = code_size = 1024
codes[:, 0] = (sqrt(C) * codebook[anchor]).bf16          // 第 0 列是 anchor 的 codebook 行
packed = (codes @ [G_q; G_p; A].T)                       // BF16，[3R,C]=[1536,1024]
q, s, v = packed.float().split([R, 2R])                  // R = rank = 512
gate[:, i>0] = sigmoid(q[:, i] + s[:, i-1] + gate_bias)  // 错位一格；第 0 列恒 0
msg  = gate * v[:, i-1]（第 0 行 0） → .bf16()
delta = (msg @ W_write[layer].T).float()                 // [H,R] → [H]
out  = (f + (rms * delta) * (row != 0)).bf16()           // anchor 行不写
next = plain_rmsnorm_weight(out, next_norm)              // y*weight，走 rmsnorm_pack_tail 语义
```

### 5.2 last-write 精修（只重做最后一层）

```text
cand[i]   = top16( target_lm_head @ h_norm_i )            // 全词表 248320，稳定 top-k
sender[i] = anchor (i=0) 或 cand[i-1]
refined[p][c] = raw_i + rms_i * ( gate(q_i + code(sender_p)) * value(code(sender_p)) @ W_write[L-1]^T )
score[p][c]   = ( rmsnorm_weight(refined[p][c]) · head_row(cand[c]) )
j = argmax_c score[0][c]；之后 j = argmax_c score[j][c]   // 沿表走，不是逐位置独立 argmax
```

采样路径：`softmax(score/temp)` → 联合 top-k/top-p/min-p → 采样并保留 q 行给验证器。
过滤规则见 `dspine_mlx/propose.py:filter_probs`（`rank<K && cumsum_before<=top_p`）。

### 5.3 输入块

```
位置        F      F+1 ... F+K        K=15（块宽 16）
token       a      MASK ... MASK      mask_token_id=248070
角色        anchor proposal...
```

---

## 6. 张量逻辑名（转换器已按此写入，C++ 绑定照此读）

| HF 张量 | 逻辑名 | 形状（27B） |
|---|---|---|
| `fc.weight` | `dspine/feature_projection` | (5120, 25600) |
| `hidden_norm.weight` | `dspine/context_norm` | (5120,) |
| `norm.weight` | `dspine/final_norm` | (5120,) |
| `layers.i.*`（同 DFlash） | `dspine/layers/i/...` | 见 `dflash.md` |
| `selected_prefix_bridge.early_read_weights[i]`（i<4） | `dspine/layers/i/bridge/read` | (1024, 5120) |
| `selected_prefix_bridge.read_weight`（i=4） | `dspine/layers/4/bridge/read` **≡** `dspine/bridge/final_read`（共享） | (1024, 5120) |
| `selected_prefix_bridge.feature_write_weights[i]` | `dspine/layers/i/bridge/write` | (5120, 512) |
| `selected_prefix_bridge.codebook` | `dspine/bridge/codebook` | (248320, 1024) |
| `selected_prefix_bridge.gate.weight` / `.bias` | `dspine/bridge/gate_projection` / `gate_bias` | (512, 1024) / (512,) |
| `selected_prefix_bridge.predecessor_gate.weight` | `dspine/bridge/predecessor_projection` | (512, 1024) |
| `selected_prefix_bridge.prev.weight` | `dspine/bridge/value_projection` | (512, 1024) |
| `selected_prefix_bridge.out.weight` | **不入产物**（转换时断言全零） | (1024, 512) |
| `draft_block_causal_version` 等 4 个标量 | **不入产物**（转换时断言 ==1） | () |

`context_key`/`context_value` 与 `attention/key`/`value` 共享，与 DFlash2 完全一致。

---

## 7. 已做的设计决策（别推翻，除非有数据）

1. **bridge 全程 BF16**：code 由 bridge 的 rounding 定义，精修要复现同一 rounding。
   实测 bridge 只有 563 MiB，量化它省不了多少却会掉接受长度。
2. **不做全词表 `TG/TSV` 预计算表**：官方贪心路径用它省算力，但 16×15 行现场重算只有微秒级，
   而词表表要多吃 762 MB 显存并固化量化后 head 的 rounding。等有 benchmark 再议。
3. **不加枚举值以外的「临时」C++ 代码**：本会话所有 C++ 都留给有编译器的会话。
4. **DSpine 必须用全词表 head**：`--lm-head-draft` 的 131072 行 shortlist 会改变 top-16 候选集合，
   接受长度会掉；第一版走 `ProposalHead::Full`（Q8 全词表 head ≈1.27 GB）。
5. **目标权重不动**：现有 `qwen3_8_27b` / `_nvfp4` 产物继续用，只是重新打包一份带 `dspine` 组件的产物。
   注意 `--source reference=` 的直通**不能用来加组件**：`tools/convert/official_recipes.py:141`
   要求参考产物含全部逻辑参数，旧产物没有 `dspine/*` 会直接报错。要么用原始 BF16 源重烤，要么先
   把 `_import_stored` 改成"按参考产物实际拥有的组件导入"。
6. **三份已发布产物里没有 dspine，且暂时不要加**。dspine 组件 2,025,387,780 个 BF16 元素
   = 3.77 GiB（bridge 563 MiB 必须 BF16 + backbone 3.22 GiB）。加进去功能上无害（引擎只加载
   启动时选中的组件），但现在没有 `--spec dspine`，白白多传 3.77 GiB × 3。等 T2–T6 落地后加：

   ```bash
   python3 -m tools.convert --model /tmp/src-qwen --recipe qwen3_8_27b_a8 \
     --source dflash2=... --source dspine=.../Qwen3.8-27B-DSpine-b16 \
     --components text,vision,mtp,dflash2,dspine --proposal --name qwen3.8-27b \
     --out .../qwen3_8_27b_a8_dspine.ninfer --device cpu     # 约 22.8 GiB
   ```

   到那时再按实测决定 backbone 要不要 Q8，可把 3.77 GiB 压到约 2.3 GiB。

---

## 8. 验收标准（这个分支算完成的样子）

1. `--spec dspine --draft-tokens 15` 在 4090 上跑通，greedy 下与官方 MLX 参考的 draft tokens 完全一致。
2. 单请求 decode 超过 `--spec dflash2 --draft-tokens 7` 的基线（4090 上约 134 tok/s 的推算值）。
3. 24 GB 卡上 32k 上下文稳定 200 轮；`--draft-tokens 7` 时上下文上限写进 README。
4. bridge 的 8 个 BF16 cast boundary 在 `docs/maintainer/dspine.md` 里有契约记录。
5. 引擎能把 §2.5 的三份产物正常加载（`--spec mtp` / `--spec dflash2` 各跑一次），证明无卡产出的
   文件在真实引擎里可用——这一条现在**没有**证据，只有 `inspect` 的结构自洽。

---

## 9. 参考实现与资料

- 权重：<https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16>
- 论文：arXiv:2609.36173
- 官方实现（**数值权威**）：<https://github.com/Tencent/DSpine>
  - `dspine_mlx/dspine_mlx/bridge.py`：bridge 的逐行参考（8 处 cast）
  - `dspine_mlx/dspine_mlx/propose.py`：top-k、过滤、沿表走、采样
  - `dspine_mlx/dspine_mlx/model.py`：块前向与 context K/V
  - `sglang_dspine/models/dspine.py`：config 校验与 native SWA 语义（窗口 8192，含 context）
- llama.cpp 侧参考（ggml 图，不是 op 契约）：<https://github.com/Ajay9o9/llama.cpp/tree/dspine>（`src/models/dflash.cpp`）
- 社区 GGUF（只用于行为/默认参数参考）：<https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF>

---

## 10. 取回本分支

分支已经在 origin 上：

```bash
git fetch origin
git checkout -b feat/dspine-draft origin/feat/dspine-draft
```

云端同时留了一份 `ninfer-dspine-draft.bundle` 作为离线备份（需要本地已有 `0916420a`）：

```bash
git bundle verify /path/to/ninfer-dspine-draft.bundle
git fetch /path/to/ninfer-dspine-draft.bundle feat/dspine-draft:feat/dspine-draft
```

云端会话的复现命令（无 GPU 也能跑）：

```bash
python3 -m pytest tests/convert -q
python3 -m tools.convert --help     # 现在接受 --source dspine=PATH
```

---

## 11. 本会话的操作记录（下一个会话别重复踩）

### 11.1 下载限速：只靠 `curl --limit-rate`，不要碰 eth0

用户要求下载总量不超过 20 MB/s。可用的做法是**每连接限速**，不是全局整形（不要动 `eth0` 的
配置，那是服务器的网络设置）：

```bash
# 8 个连接 × 2.4 MiB/s ≈ 19.2 MiB/s
cat files.txt | xargs -P 8 -I{} curl -sL -C - --limit-rate 2400k --retry 3 \
    --max-time 5400 -o "/tmp/src/{}" "https://huggingface.co/<repo>/resolve/main/{}"
```

**踩过的坑**：`job_kill` 只杀外层 shell，`hf download` 的 python 子进程会活下来，继续无限速往
`<local-dir>/.cache/huggingface/download/` 写。表现是接口速率翻倍（测到 32–39 MiB/s）、`du` 比
实际模型文件大一倍。处理：`pkill -f 'hf download'` + 删掉 `.cache`。别再混用 `hf download` 和
`curl` 写同一个目录。

### 11.2 上传限速：`/tmp/hf_upload_limited.py`（未实测，等 token）

`hf upload` 没有限速参数，新版默认走 Xet（Rust 内读文件，Python 拦不住）。做法：

- `HF_HUB_DISABLE_XET=1` 强制走经典 LFS 单次 PUT；
- 包装 `huggingface_hub._commit_api.CommitOperationAdd.as_file`，给返回的文件对象套一个读代理，
  每次 `read()` 后按已发送字节数补 `sleep`；
- 入口：`UPLOAD_MIB=<n> HF_TOKEN=... python3 /tmp/hf_upload_limited.py <repo> <file> <path_in_repo>`。

**还没验证**：token 未到，脚本一次都没跑过。第一次务必先传几 MB 的文件，用 `/proc/net/dev`
核对实际速率，再传 19 GiB。

### 11.3 CPU 转换很便宜，别把它当成瓶颈

- 单张 89.1M 元素的 MLP 矩阵，`q4_g64_fp16` 量化 **0.24 s ≈ 378 M 元素/s**（16 线程）；
- 整份 27B（1184 个参数、读 51.8 GiB、写 19 GiB）**351 s**；Bonsai（1058 个参数）**348 s**；
- 真正的瓶颈是磁盘 I/O 与下载，不是算力。所以"没有 GPU 也能出产物"是这个项目的常态。

### 11.4 `bonsai_mapping_check` 在 PTQ1_0 上会误报

该工具用 `ternary = source.read_encoded is not None` 判断行是不是三元，再决定门槛
（三元 0.80 / 直连 0.98）。PTQ1_0 这个 GGUF 张量类型它不识别，于是把三元行标成 `direct`，
拿 0.98 的门槛判 **NO**。实测那些行的行余弦是 **0.87–0.88**，正好是文档写的三元平台值
（"a wrong mapping collapses to ~0"），所以是标记问题、不是映射问题。

**待 GPU 侧确认**：T7 里用 `--spec mtp` 跑一次 Bonsai，接受长度正常即证明映射可用。若要把工具
修好，判据应改成"按 GGUF 张量类型判断"，而不是 `read_encoded`。

### 11.5 Bonsai 的 chat template 与参考不同

`bonsai_base` 打印 `reference DIFFERS FROM the GGUF's`，并把 GGUF 自带模板另存为
`gguf_chat_template.jinja`。当前三份产物用的是参考（Qwen3.8）模板。若要改成 GGUF 自带模板：

```bash
--resource chat_template.jinja=/tmp/bonsai2-27b/gguf_chat_template.jinja
```

重烤一次即可（348 s）。哪个更对，建议在 4090 上对比同一组 prompt 的回答，别凭直觉定。

### 11.6 磁盘与清理

源权重（Qwen BF16 51.8 + Swift BF16 51.8 + DFlash2 3.6 + Bonsai GGUF 5.5 + mmproj 0.6 ≈ 113 GiB）
+ 产物 45 GiB。产物上传完、源权重确认可再下之后，`/tmp` 下的 `src-*` 可以删。
