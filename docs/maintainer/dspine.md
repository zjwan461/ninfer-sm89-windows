# DSpine draft

DSpine proposes a whole causal block with a small DFlash-shaped draft model and then folds a learned
codebook message back into the draft's residual stream after **every** draft layer. The target
verifies the block causally and stays the output authority, exactly as for
[DFlash and DFlash2](dflash.md). NInfer implements DSpine as `--spec dspine` over an artifact that
ships the `dspine` component; startup selects one speculative backend, the draft width and the
proposal head.

This reference owns the DSpine mathematics, the bridge's rounding points, the refinement and the
backend's runtime contract. Scheduling, publication and common transactions are defined by
[Engine architecture](engine-architecture.md).

## Configuration and geometry

The component config is the checkpoint's `config.json`, normalized by
[`draft_config`](../../tools/convert/qwen3_5.py). `dflash_config` carries `target_layer_ids`,
`mask_token_id`, `draft_block_causal`, `block_size` and `code_bridge` (`code_size`, `rank`, `eps`,
optional `kappa`, `feature_layers`, `head_topk`). Parsing and validation live in
[`config.cpp`](../../src/models/qwen3_5/config.cpp).

| Quantity | Qwen3.8-27B DSpine-b16 |
|---|---:|
| Hidden width H | 5120 |
| Draft layers | 5 (all sliding attention) |
| Intermediate width | 17408 |
| Q heads / KV heads / head dimension | 32 / 8 / 128 |
| Target block IDs | `[1,16,31,46,61]` |
| Sliding-window scalar S | 8192 |
| Norm epsilon / RoPE theta | `1e-6` / `1e7` |
| Draft block width W (trained) | 16 |
| Code size C / rank R | 1024 / 512 |
| Code-bridge epsilon | 1e-6 |
| Refinement width K | 16 |

Startup accepts `--draft-tokens K` in `1..15` and concurrency in `1..8`; the served block width
`W=K+1` may be narrower than the trained 16 but never wider, because the bridge's anchor and
predecessor rows are defined per block. `head_topk` must be 16: the sparse acceptance Op and the
proposal ring are sixteen wide.

The DSpine draft keeps only fixed cyclic draft state (five sliding layers of capacity 8192) and no
paged Full backend KV, so `backend_frontier_at` reports zero like DFlash2.

## Block forward

```
x       = embedding(block_ids)                       // anchor + K mask tokens
residual= x
x_in    = layer[0].input_norm(x)
for i in 0..L-1:
    attn = attention(x_in, ctx_kv[i], block-causal, |p - p_q| < 8192)
    h    = residual + attn
    mlp  = mlp(post_attention_norm(h))
    raw  = h + mlp                                   // pre-bridge residual
    x_in, residual = bridge(raw, layer i)            // next input already normalized
```

Attention is causal *inside* the block: a proposal row sees every context row inside the window and
only the block rows at or before itself. That is `SlidingWindowMask::BlockCausal` in
[`sliding_window_attention`](../../include/ninfer/ops/sliding_window_attention.h); DFlash and
DFlash2 keep `ContextWindow`, where every live block row is visible.

## The bridge's eight BF16 rounding points

The bridge math is taken from the reference implementation
([`bridge.py`](https://github.com/Tencent/DSpine/blob/main/dspine_mlx/dspine_mlx/bridge.py)) and
must not be "improved". Each numbered point is a place where a value is rounded to BF16 and later
read back; the refinement reproduces the same rounding, so changing any of them changes the draft
tokens.

Read, after the layer's MLP (`raw` is the BF16 residual buffer, `f = FP32(raw)`,
`n = b*W + t` column of the block):

1. `rms = sqrt(max(mean(f*f), 1.1754943508222875e-38))` — the FP32 *tiny* floor, not `rms_norm_eps`.
2. `xnorm = BF16(f / (rms + eps))` — epsilon is added to the denominator.
3. `projected = FP32(xnorm) @ read[layer]^T` (FP32 accumulation), then
   `codes = BF16(sqrt(C) * projected / (||projected||_2 + eps))`.
4. The block's anchor column is replaced by `BF16(sqrt(C) * codebook[anchor])`, which is what makes
   the anchor row's message well defined.
5. `packed = BF16(FP32(codes) @ [G_q; G_p; A]^T)` — the whole `[3R, C]` projection is rounded once.

Write, `msg` and `delta`:

6. `msg[r] = BF16(sigmoid(q[r] + s[r - 1] + gate_bias[r]) * v[r - 1])` for `r = t > 0`, zero on the
   anchor row. `q`, `s` and `v` are the FP32 halves of `packed`; `s` and `v` come from the same
   request's previous block row, one column lower.
7. `out = BF16(f + (t != 0 ? rms * (msg @ write[layer]^T) : 0))` — the anchor row never receives a
   delta, and the delta carries the *stored* `rms` factor.
8. `next = BF16(BF16(out * rsqrt(mean(out^2) + eps)) * next_norm)` — the normalized value is rounded
   to BF16 *before* the gain is applied, i.e. two roundings, and `next_norm` is the next layer's
   `input_layernorm.weight` (or the final `norm.weight` for the last layer).

The Op is [`dspine_bridge_read`](../../include/ninfer/ops/dspine.h) /
`dspine_bridge_write`, implemented in
[`dspine.cu`](../../src/ops/dspine/dspine.cu). Bridge tensors stay BF16 in the artifact: 563 MiB for
the whole bridge, and quantizing it would change the codes and shorten the accepted prefix.

## Serial last-write refinement

`linear_topk` produces the stable top-16 candidate ids of every proposal row
(`F(1..K)` positions) from the final normalized block rows. The last layer's *write* is then redone
per candidate sender, in FP32 except at the stored rounding points:

```
sender(row 0)  = anchor          sender(row i>0) = candidates[i-1]
e              = BF16(sqrt(C) * codebook[sender])
tsv            = BF16(FP32(e) @ [G_p; A]^T)                    // per half
m              = sigmoid(q + tsv_gate + gate_bias) * tsv_value  // FP32
head_row       = BF16 target head row of the candidate
a              = BF16(FP32(head_row) * norm_weight)
base[c]        = sum_h FP32(head_row) * norm_weight * FP32(raw)
tg[c]          = BF16(FP32(a) @ write[L-1]^T)
table[i,s,c]   = base[c] + rms * sum_r m[s][r] * FP32(tg[c][r])
```

`table` is FP32. The greedy chain walks it as `j = argmax_c table[0,0,c]`, then
`j = argmax_c table[i,j,c]`, with exact ties resolved by the lower candidate token id. The sampled
path exponentiates `table[i,sender,:] / temperature`, applies the joint top-k/top-p/min-p filter,
renormalizes, draws with the counter-based uniform at
`(seed, position, kSamplePurposeDSpineProposal)` and stores exactly that filtered distribution as
`proposal_q`, so the target's standard speculative rejection applies unchanged.

Implementation: `dspine_refine_score` (head-row gather, `tg` GEMM, table) and `dspine_refine_chain`
(chain walk, sampling) in [`dspine.cu`](../../src/ops/dspine/dspine.cu).

### Deliberate differences from the reference

- **No vocabulary-wide `TG`/`TSV` tables.** The reference precomputes `[V,R]` and `[V,2R]` BF16
  tables (about 762 MB). NInfer computes the same values for the 16 candidates of each proposal row
  and for the sender tokens of the round, which costs well under a tenth of a millisecond on the
  4080 SUPER and keeps the stored head rounding out of a second artifact.
- **One table for greedy and sampling.** The reference's sampled path recomputes the refined rows in
  FP32; NInfer samples the same BF16-rounded table the greedy chain walks. The stored `proposal_q`
  is the distribution the draft was actually drawn from, so the verifier's rejection remains exact
  for that distribution.
- **`kappa` is parsed but not used.** The reference reads it from the config and never applies it.
- **Full proposal head only.** `--lm-head-draft` is rejected: a 131072-row shortlist changes the
  top-16 candidate set and shortens the accepted prefix.

## Runtime wiring

- `SpeculativeBackend::DSpine` shares the masked-draft path (`is_masked_draft_backend`): draft
  context append, `DFlashDecodeState`, workspace planning, commit/materialization and the
  target-verify transaction are the DFlash ones.
- `DFlashDecodeState::candidate_ids` (`[16,K,B]` I32) and `proposal_q` (`[16,K,B]` FP32) are
  allocated for both sparse backends; `linear_topk` writes the candidates in place and the verifier
  reads the same sets the refinement scored.
- Verification uses `speculative_accept_sparse_drafts`, so sampled rounds run distribution-correct
  rejection sampling against the target distribution.
- The proposal workspace is `workspace::dspine_round`: two residual streams (the write must keep
  the pre-write `raw` stream), the per-layer bridge operands, and the refinement buffers.
- The draft block is a `[H, W, B]` tensor; the attention and MLP projections take it flattened to
  `[H, W*B]`, so every concurrency in `1..8` uses the same Op shapes. The anchor row, the
  predecessor row and the candidate columns are all lane-local.
- CUDA Graph profiles for DSpine use the DFlash2 scheme (one executable per width range), because
  the bridge and refinement nodes change route with the visible context inside a range.

Ops and profiles added for this backend:

| Change | Where |
|---|---|
| Sliding window 8192 and the block-causal mask | `sliding_window_attention`, `kv_cache_append` cyclic capacity 8192 |
| Q8 `[5120,4096]` residual projection (the draft attention output) | `linear_add` Q8 plan and launchers |
| Bridge and refinement Ops | `src/ops/dspine/`, `include/ninfer/ops/dspine.h` |

## Numerical qualification

The bridge and refinement were compared against an independent host oracle: the original BF16
checkpoint's tensors, the artifact's Q8 head rows and the engine's own per-layer inputs. One round
of `--spec dspine --draft-tokens 15`, batch 1, `--no-cuda-graph`, on the Qwen3.8-27B artifact:

| Quantity | Result |
|---|---|
| `rms` per layer | max abs error `6.1e-5` |
| `codes` per layer | ≥ 99.95% bit-identical BF16, max abs error `2e-3` |
| `packed` per layer | ≥ 98.9% bit-identical BF16, max abs error `3.1e-2` |
| bridge `out` per layer | ≥ 99.99% bit-identical BF16, max abs error `2.0` on residuals of magnitude ~1e3 |
| `next` (last layer) | 99.999% bit-identical, max abs error `4.8e-7` |
| refinement `base` | max abs error `3.9` on scores of ~1e4 (4e-4 relative) |
| refinement `table` | max abs error `21` on ~7e3 (3e-3 relative, the BF16 `tg` rounding) |
| draft tokens | identical |

Causal prefix stability: with the same prompt, `--draft-tokens 7` produced exactly the first seven
draft tokens of `--draft-tokens 15` in the same round. The current evidence for the mask itself is
the Op oracle: `test_sliding_window_attention` now runs `BlockCausal` at window 8192 for widths
1, 2, 7 and 16 and batches 1, 4 and 8, including contexts on both sides of the window boundary and
CUDA-graph replay, against the naive FP64 predicate.

Measured on the 32 GB RTX 4080 SUPER (80 SM `build-4080s`, 20.6 GB artifact, `--kv-dtype rk4v4-e8`,
thinking off, same code prompt, CUDA Graphs on):

| Backend | Draft | Sampling | Acceptance | tok/round | Decode |
|---|---:|---|---:|---:|---:|
| DFlash2 (`qwen3_8_27b_a8.ninfer`) | 15 | greedy | 63.8% | 10.53 | 226.1 tok/s |
| DSpine (`qwen3_8_27b_a8_dspine.ninfer`) | 15 | greedy | 64.0% | 10.60 | 233.2 tok/s |
| DSpine | 7 | greedy | 80.8% | 6.62 | 157.1 tok/s |
| DFlash2 | 15 | temp 0.8, top-k 20 | 59.4% | 9.77 | 221.7 tok/s |
| DSpine | 15 | temp 0.8, top-k 20 | 54.1% | 9.07 | 198.5 tok/s |

Both artifacts carry the same text weights (the new artifact imports them from the published one),
so the comparison is like-for-like. At equal draft width the two drafts are close and DFlash2 is
sometimes ahead; the published comparison pairs DSpine(16) with DFlash2(8), and at *that* pairing
NInfer reproduces the direction of the published table:

| Prompt (greedy, thinking off) | DSpine 15 | DFlash2 8 | Ratio | DFlash2 15 |
|---|---:|---:|---:|---:|
| merge-sorted function | 10.60 | 7.95 | 1.33 | 10.53 |
| HumanEval-style completion | 9.15 | 7.93 | 1.15 | 10.82 |
| grade-school word problem | 5.54 | 5.54 | 1.00 | 6.54 |

The published evaluation is BF16 with `xhigh` reasoning over 4096 tokens on GSM8K/MATH/HumanEval,
so magnitudes are not comparable; the graded word problem is the one case here where DSpine does not
lead, and a Q8 target (Q8 features and head, versus BF16 in the reference) plausibly costs DSpine
more than DFlash2 because the code bridge re-encodes the target residual at every layer. A BF16
draft backbone is not loadable in NInfer: the fused QKV input projection requires Q8.

Concurrency was exercised through `ninfer-serve --max-concurrency 2 --spec dspine --draft-tokens
15`: two simultaneous requests with different prompts produced distinct, coherent answers and
per-lane draft counters (112 drafted/55 accepted and 295/42), with no startup or round error. The
B=2 CUDA Graph profiles capture and install normally.