# 交接：把 DSpine 草稿模型接进 NInfer（`feat/dspine-draft`）

> 临时工作计划，给用户 Windows/4090 机器上的下一个会话。全部任务关闭后删除本文件
> （AGENTS.md：临时计划只服务在办工作）。用户用中文交流，回答用中文；数字凡是没在 GPU 上测过的，
> 必须说成估算。

---

## 0. 一句话状态

转换器（Python）已经把 `Qwen3.8-27B-DSpine-b16` 完整映射成一个 `dspine` 组件并逐张量对拍通过；
**C++/CUDA 一行没动**（本会话没有 GPU、没有 nvcc，改了也无法编译验证）。
下一个会话从「T2 C++ 配置与绑定」开始，数学契约见第 5 节，不要自己重新推导。

---

## 1. 分支与提交

```
feat/dspine-draft
├─ 0916420a  feat(launcher): 新增模型下载页与自定义参数        <- origin/main 顶端
├─ 477415fe  feat(rtx5060ti): sm_120a / 36-SM 支持
├─ 0c3f7ccf  fix(launcher): 5060 Ti 思考预算
├─ 83da6b55  fix(launcher): 统一 Bonsai 预设预算               <- 已合并 feat/rtx5060ti-bonsai
└─ 458a7849  feat(convert): convert the Qwen3.8-27B DSpine draft as a dspine component
```

云端会话没有 GitHub 凭据，**分支没有 push**。取回方式见第 10 节。

---

## 2. 已完成（可复现）

改动文件：

| 文件 | 内容 |
|---|---|
| `tools/convert/qwen3_5.py` | `draft_config()` 支持 `backend="dspine"`；`_dspine_bridge()` 校验并归一化 bridge 配置；`_Builder.dspine_bridge()` / `dspine_shared()` 映射张量；`build_model()` 接受 `dspine` |
| `tools/convert/official_recipes.py` | `_optional()`：backbone 取 Q8、`*/bridge/*` 强制 BF16；`dspine` 也共享 `context_key/value`；最后一层 `bridge/read` 与 `bridge/final_read` 共享 |
| `tools/convert/proposal.py`、`tools/convert/__main__.py` | 组件列表/CLI 帮助加入 `dspine`（`--source dspine=PATH --components text,dspine`） |
| `tests/convert/test_qwen3_5.py` | 新增 3 个测试：配置归一化 + 逐层 bridge 切片、8 个非法配置用例、recipe 量化策略 |
| `docs/weight-conversion.md` | 说明 `dspine` 组件与其 BF16 约束，并注明运行时尚未选择它 |

验证证据：

```bash
python3 -m pytest tests/convert -q          # 75 passed, 4 skipped
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

## 3. 没做的，以及为什么

| 项 | 原因 |
|---|---|
| `SpeculativeBackend::DSpine` 与引擎接线 | 涉及 18 个文件与 switch 分支；本机没有 nvcc，改完无法编译，留下不能编译的代码只会拖慢下一个会话 |
| `src/models/qwen3_5/config.cpp` 的 dspine 解析 | 需要先有枚举/组件名入口，否则是死代码（AGENTS.md 不允许占位实现） |
| 因果块 mask、bridge op、refinement op | CUDA 数学，必须在有 GPU 的会话里对着 FP32 oracle 写 |

---

## 4. 下一会话的任务清单（按依赖顺序）

### T1 构建基线（先做，半天）

```bat
git fetch <bundle> feat/dspine-draft:feat/dspine-draft
git checkout feat/dspine-draft
CMakePresets.json 里选 4090 预设（NINFER_TARGET_SM_COUNT=128）
cmake --build build-4090 -j
python -m pytest tests/convert -q     :: 必须 75 passed, 4 skipped
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
5. **目标权重不动**：现有 `qwen3_8_27b` / `_nvfp4` 产物继续用，只是重新打包一份带 `dspine` 组件的产物
   （`--source reference=<旧产物>` 的 `import_encoded` 直通，不重新量化）。

---

## 8. 验收标准（这个分支算完成的样子）

1. `--spec dspine --draft-tokens 15` 在 4090 上跑通，greedy 下与官方 MLX 参考的 draft tokens 完全一致。
2. 单请求 decode 超过 `--spec dflash2 --draft-tokens 7` 的基线（4090 上约 134 tok/s 的推算值）。
3. 24 GB 卡上 32k 上下文稳定 200 轮；`--draft-tokens 7` 时上下文上限写进 README。
4. bridge 的 8 个 BF16 cast boundary 在 `docs/maintainer/dspine.md` 里有契约记录。

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

## 10. 取回本分支（云端没有 GitHub 凭据）

云端已生成 `ninfer-dspine-draft.bundle`。在本地仓库里：

```bash
git bundle verify /path/to/ninfer-dspine-draft.bundle
git fetch /path/to/ninfer-dspine-draft.bundle feat/dspine-draft:feat/dspine-draft
git checkout feat/dspine-draft
```

或者在有凭据的机器上直接从本仓库推送：

```bash
git push -u origin feat/dspine-draft
```

云端会话的复现命令（无 GPU 也能跑）：

```bash
python3 -m pytest tests/convert -q
python3 -m tools.convert --help     # 现在接受 --source dspine=PATH
```
