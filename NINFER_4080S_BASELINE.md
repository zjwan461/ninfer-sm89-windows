# NInfer on the 32 GB RTX 4080 SUPER — measured baseline

> Card: RTX 4080 SUPER, AD103, **80 SMs**, 32 GB (modded), 736 GB/s, `sm_89`.
> Build: `build-4080s` (`NINFER_TARGET_SM_COUNT=80`).
> Artifact: `qwen3_8_27b_a8.ninfer` (19.0 GiB, int8-prefill).
> Every number below was measured on that machine on 2026-10-25. Before this file the
> repository had **no** 4080S performance record at all, so all of it is new.

## 1. What the 80-SM retarget actually buys

| Build | decode, no speculation | prefill (1972-token prompt) |
|---|---:|---:|
| `build-4080s` (80 SM) | 37.8 tok/s | 3.31k tok/s |
| `build-4090` (128 SM) | 37.7 tok/s | 3.32k tok/s |

Both numbers come from running **both builds on the same 4080 SUPER**.

**Conclusion: the SM constant has no measurable effect on this card.** Decode is
bandwidth-bound (8K context keeps attention small) and prefill came out identical at the
chunk sizes measured, so the "attention tail-wave tax" that motivated
`ninfer-4080s-80sm-retarget-plan.md` does not show up here. The retarget is still correct
to keep (it costs nothing and documents intent), but it should not be billed as a speed win.

## 2. Decode: where the time goes

Repeated 3x per point; spread between runs is under 1%.

| Content | none | k=2 | **k=3** | k=5 |
|---|---:|---:|---:|---:|
| prose | 38.1 | 66.7 | **74.9** | 64.6 |
| code | 37.2 | 78.2 | **86.6** | 63.7 |

`k` = `--draft-tokens`. **k=3 is the optimum for both content types**; k=5 loses because a
draft hop costs more than the extra accepted tokens are worth. `--draft-tokens 3` in the
launcher is therefore correct — a single-run measurement will occasionally suggest k=5 is
faster, and that is noise.

Cost model fitted from k=1..5 (prose):

```
round = 26.2 ms  (one verify forward = one pass over the weights)
      + k x 3.3 ms (one draft hop)
k=3 -> 36.1 ms, 2.65 tokens  -> 73.4 tok/s   (measured 74.9)
```

* One verify forward is at **86% of the bandwidth roofline** (15.9 GiB ÷ 736 GB/s = 23.2 ms
  vs 26.2 ms measured). The engine core is healthy.
* A draft hop (3.3 ms) is ~50% above its own bandwidth floor (LM head 1.27 GB ÷ 736 GB/s =
  1.73 ms + MTP layer ≈ 0.5 ms). That gap is 80-SM GEMV occupancy, and it is the single
  largest remaining decode target (~40% of the round at k=3).
* `--lm-head-draft` earns its keep: k=5 with it is 76.5 tok/s, without it 61.1.
* `--ngram chain` contributes nothing on prose (61.1 vs 61.3 without).

## 3. Prefill

The earlier sweep could not go above 1536 because larger values aborted (section 6). With the
cooperative budget now derived from real occupancy, all values run. Three runs each, median,
1972-token prompt:

| `--prefill-chunk` | prefill |
|---|---:|
| 512 | 3.26k tok/s |
| **1024** | **3.41k tok/s** |
| 2048 | 3.40k |
| 3072 | 3.39k |
| 4096 | 3.39k |

Prefill is flat from 1024 upward, so 1024 stays the launcher's value; the workspace grows with
the chunk, so there is nothing to gain above it. `--prefill-chunk` is now capped at 4096 purely
as a sanity guard.

## 4. Context and concurrency on 32 GB

All measured through `ninfer-serve` with `--kv-capacity auto --kv-dtype rk4v4-e8`.

| max-context | concurrency | peak VRAM | single | aggregate |
|---:|---:|---:|---:|---:|
| 131072 | 3 | 27.5 GB | 64.6 | — |
| 131072 | 4 | 30.0 GB | — | 127.0 (4 clients) |
| **262144** | **2** | **29.5 GB** | **73.0** | **102.9** (2 clients) |
| 262144 | 1 | 24.5 GB | 64.5 | — |

`--kv-capacity auto` resolved 524288 tokens for the 262144/2 pairing, which is exactly
`max_context x concurrency`. VRAM grows ~2.5 GB per extra concurrency slot, **not** with the
host-spill flags: `--host-kv-mib` / `--host-state-slots` are host RAM and do not change the
GPU footprint (measured identical at the same concurrency).

The launcher now uses **262144 / concurrency 2**; `131072 / concurrency 4` is the
alternative when four short concurrent sessions matter more than one long one.

## 5. Comparison with the 4090 figures in the README

| Metric | 4080S (measured) | 4090 (README) | ratio | hardware ratio |
|---|---:|---:|---:|---:|
| decode, MTP | 74.9 (k=3) | 120 | 0.62 | 0.73 (bandwidth) |
| decode, MTP + n-gram | 74.9 | 289 | 0.26 | 0.73 |
| prefill | 3.37k | 5.79k | 0.58 | 0.625 (SM/compute) |

* Prefill (0.58) matches the SM-count ratio — it is a compute-bound path and this card has
  80/128 of the SMs.
* Decode with MTP (0.62) sits between the bandwidth ratio and the compute ratio, because a
  round mixes one bandwidth-bound verify with compute-bound draft hops.
* The 289 "best decode" figure depends on n-gram hits; on prose n-gram contributes nothing
  on either card, so it is not a like-for-like number for this workload.

## 6. The `--prefill-chunk >= 1792` abort: root cause and fix

**Reproduced 2/2**, not a size limit in the parser:

| prompt | chunk 1536 | chunk 1792 |
|---|---|---|
| 532 tokens | exit 0 | exit 0 (single chunk, no split) |
| 1972 tokens | exit 0, 3.32k tok/s | **exit 0xC0000409, 2/2** |

The trigger was `tokens > chunk` (the prefill actually splits) **and** `chunk >= 1792`. A trace
run put the abort on the cooperative launch in
`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu`.

That launch is guarded by a resident-CTA budget (`launch_bf16_prefill_mma`, `SplitK > 1`
branch): the token range is cut into
`max_token_tiles = multiprocessor_count * kResidentCtasPerSm / kCtasPerTokenTile` pieces so that
every cooperative launch fits. **`kResidentCtasPerSm` was a compile-time guess**
(`cooperative_resident_ctas_per_sm()`, 2 for the 27B geometry, 4 for most 35B splits), qualified
on a different card, while a cooperative launch requires every block to be co-resident. When
the real occupancy was lower than the guess, the split produced more pieces than the device
could schedule and the launch failed instead of falling back.

The catalog that guess is checked against is also pinned to the 4090's SM count:

```cpp
static_assert(catalog_is_resident(k27Routes, 128, 3, resident_ctas_27), ...);
```

Replacing that `128` with `kTargetSmCount` makes the **80-SM build fail to compile**, which
independently confirms the 27B cooperative catalog is not resident-safe on 80 SMs at its upper
bound under the old model.

### 6.1 Fix (applied)

The budget now comes from the driver, not from a constant:

```cpp
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, <kernel instantiation>, block.x, smem);
resident_ctas = multiprocessor_count * min(occ_full_tokens, occ_predicated_tokens);
```

Both token-variant instantiations are queried and the smaller figure is used, because the
variant is only known inside the launch helper. If the query fails, the old compile-time
figure is used as a fallback rather than refusing to run.

Verified on both model geometries with a 1972-token prompt, two runs each:

| chunk | before | after |
|---|---|---|
| 1792 | **abort 2/2** | 3.34k tok/s, exit 0 |
| 2048 | **abort 2/2** | 3.39k tok/s, exit 0 |
| 4096 | — | 3.39k tok/s, exit 0 |
| Bonsai (35B geometry, split-32) 2048 | not tested | 3.65k tok/s, exit 0 |

`--prefill-chunk` keeps a sanity cap of 4096 (`kPrefillChunkMaximum`); 8192 is rejected with a
clear message instead of being attempted.

### 6.2 Still open

The `static_assert` in `bf16_gdn_gating_proj_plan.cpp` still says `128`: the catalog really is
not resident-safe at `kTargetSmCount * 3` CTAs per SM, so switching it now would break the 80-SM
build. With the runtime budget measured from occupancy, that constant no longer gates
correctness; it is a catalog-review bound for the reference card. Tightening the catalog (or
re-deriving the assertion for the 3-per-SM model) remains open.

## 7. Draft hop: the "one projection" hypothesis is dead

The hop (one MTP draft step) costs 3.3 ms against a naive 2.2 ms floor, so the LM head
projection looked like the culprit. It dispatches through
`src/ops/linear/q8/shapes/n248320_k5120.cu`; for `tokens <= 8` that table selects
`launch_q8_ksplit<Geometry, 8, C8>` (8-way K split, `Q8KSplitSchedule<8,8,2,...>`), and the
sibling schedules C16/C24/C32/C40 are already compiled in.

Measured (3 runs each, median, `--spec mtp --draft-tokens 3 --lm-head-draft`, prose):

| K-split schedule | decode |
|---|---:|
| C8 (current) | 74.8 tok/s |
| C16 | 74.8 |
| C24 | 74.7 |
| C32 | 74.5 |
| C40 | 74.5 |

**No effect** — the K-split depth is not the lever.

### 7.1 Per-op floor on this card

Configured with `-DNINFER_BUILD_BENCHMARKS=ON` and ran
`ninfer_linear_bench --suite qwen3_6_27b`. Single token (T=1), i.e. the draft-hop shape:

| op | dtype | shape | time | effective GB/s | % of 736 GB/s |
|---|---|---:|---:|---:|---:|
| `27b.output_head` | Q6 | 248320x5120 | 1.51 ms | 659.8 | **90%** |
| `27b.draft_head` | Q4 | 131072x5120 | 0.59 ms | 606.0 | **82%** |
| `27b.mtp_gate_up` | Q8 | 34816x5120 | 0.35 ms | 534.8 | 73% |
| `27b.mtp_attention` | Q8 | 14336x5120 | 0.16 ms | 491.6 | 67% |
| `27b.mtp_input` | Q8 | 5120x10240 | 0.11 ms | 523.4 | 71% |

The bench prints `dram_spec_gbs=1792` because it references an RTX 5090 — the usable column is
effective GB/s. This card is 736 GB/s (11501 MHz x 256-bit / 8, verified with nvidia-smi).

### 7.2 Conclusion: no single win, the hop is already near its floor

A hop is **not one projection**. It is the MTP layer ops plus a head. Summing the known ops
gives ~1.5 ms (draft_head) to ~2.4 ms (output_head) against a measured 3.3 ms per hop; the
remainder is the small ops the suite does not list (down projection, norms, argmax) plus graph
replay. Every large op already runs at 67-90% of the card's bandwidth, and the MTP draft path
**is** CUDA-graph captured (`capture_mtp_decode_batch`, `mtp.cpp:215`), so launch overhead is
already amortized.

The earlier hope of "3.3 -> 2.2 ms (+13%)" assumed a single bf16 pass over the head; the
benchmarks show that assumption was wrong. **The remaining gap to the 4090 is the hardware
ratio** (736/1008 bandwidth, 80/128 SMs), not a missing optimization.

Also worth recording: `linear_topk` (`src/ops/linear_topk/*`, the two-pass producer+merge head)
is **not** on the MTP draft path — it belongs to the DFlash2 proposal. A timing probe installed
there printed nothing during an MTP run, which is how that was established; both probes (the
`linear_topk` timer and the `NINFER_LMHEAD_KSPLIT` override) were reverted.

## 8. Open items

1. **Catalog residency assertion** — the runtime budget is fixed (section 6.1). What remains is
   the static_assert in f16_gdn_gating_proj_plan.cpp, still pinned to 128 SMs: switching it
   to kTargetSmCount fails to compile because the catalog is not resident-safe under the
   older 3-CTA-per-SM model. Either tighten the catalog or re-derive the assertion.
2. **Draft hop** — closed as "near its floor" by section 7. Reopening it would need a kernel
   that beats 67-90% of bandwidth on every one of five ops, which is not a tuning exercise.
3. **`--max-context` 262144 with concurrency 3** does not fit (≈33 GB). Not required: the
   launcher targets 262144 at concurrency 2 by design.
4. **Benchmark targets are off by default** — `-DNINFER_BUILD_BENCHMARKS=ON` is needed for any
   per-op work; worth a line in the build manual.

## 9. Smaller artifacts: what exists, and what this engine can load

The only way to decode faster on this card is to make each token read fewer bytes. Sections 2 and 7
show the current artifact already sits at ~5.0 bits per weight and at 88 % of the bandwidth
roofline, so the remaining lever is a smaller artifact, not a faster kernel. What is actually
available:

| artifact | on-disk | resident | this build | notes |
|---|---:|---:|---|---|
| `qwen3_8_27b_a8.ninfer` (in use) | 19.03 GiB | 15.9 GiB | **loads** | ~5.0 bits/weight, 88 % of roofline |
| `qwen3_8_27b.ninfer` (official) | ~19 GiB | — | loads | same weights, no int8 prefill |
| `qwen3_8_27b_nvfp4.ninfer` (official NVFP4) | **22.09 GiB** | — | loads | **bigger, not smaller**: NVFP4 for MLP + FP8 elsewhere, aimed at Blackwell `a4`; a regression on `sm_89` |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3` (community, 3.5 bits) | 13.99 GiB | **10.9 GiB** | **refuses** | see below |

For reference when judging any replacement: the in-use artifact scores **perplexity 7.0251** on a
113,206-token corpus of this repository's own docs (
infer-perplexity --text <corpus> --context 512
--stride 256, kv fp8-e4m3-r256). A candidate artifact is worth its byte saving only if it stays
within ~2 % of that number.

### 9.1 Why the 3.5-bit artifact does not load here

```
error: weight/000139: unknown tensor layout: gguf_blocks_v1
```

This build accepts exactly five layouts (`src/artifact/formats.cpp`): `contiguous_le_v1`,
`row_split_k128_v1`, `row_scale_v1`, `block_scale_k16_m128x4_v1`, `ternary_row_k128_v1`. The
IQ3_S artifact keeps each tensor in its original **ggml block format** (IQ3_S, IQ4_XS, Q4_K, Q6_K,
IQ2_*), and those `gguf_*` formats exist only in the `iamwavecut/ninfer-all` line, from commit
`8706bfc0` on -- the artifact's own README says stock NInfer builds refuse the file, and that is
exactly what happened.

### 9.2 Measured, using that fork

`iamwavecut/ninfer-all` builds on Windows with MSVC 14.44 + CUDA 13.4 for `sm_89` (794/794 targets),
reusing this checkout's vcpkg tree. With it, the IQ3_S artifact loads at **10.9 GiB** resident
(11.6 GiB once a drafter is loaded) and generates correctly. Three runs each, median, same flags:

| engine + artifact | resident | decode, no spec | decode, MTP k=2 | decode, MTP k=3 | prefill |
|---|---:|---:|---:|---:|---:|
| this build + int8 | 15.9 GiB | 40.4 | 81.5 | 82.7 | **3.39k** |
| fork + int8 | 15.9 GiB | 40.4 | — | 87.7 | 1.39k |
| **fork + IQ3_S** | **10.9 GiB** | **52.5** | **111.6** | 103.0 | 1.71k |

Draft-depth sweep on the fork with IQ3_S: k=1 87.5 (length 1.84), **k=2 111.6 (2.54)**,
k=3 103.0 (2.52), k=5 104.0 (2.89). **k=2 is the optimum there**, not the k=3 this repository's
launcher uses -- the smaller file turns each round faster, so a shallow draft wins.

Two effects separate cleanly:

* **artifact**: same engine, k=3, 103.0 against 87.7 = **1.17x**;
* **engine**: same artifact, k=3, 87.7 against 82.7 = **1.06x** (the fork is slightly ahead);
* **combined best**: 111.6 against 82.7 = **1.35x** on decode.

But the prefill cost is real and large: **1.39-1.71k tok/s against 3.39k** (2.0-2.4x slower). On a
1972-token prompt that is 0.58 s against 1.15 s, so the fork only wins once a request generates
roughly 100+ tokens: at 500 tokens, 4.5 s + 1.15 s beats 6.0 s + 0.58 s. For short answers and for
prompt-heavy work (long-context agents, RAG) the current build is still the faster engine.

Why the fork's prefill is slower is **not** explained by the artifact (it happens with the int8 file
too) and was not investigated; it is likely its own device-profile tuning path.

### 9.3 What porting `gguf_*` into this fork would actually cost

Measured on the fork's tree, for the dense Qwen3.8 path only:

| part | files | lines |
|---|---:|---:|
| `src/ops/linear/gguf/` (bridge + `gguf_linear` + per-type MMQ and VEC kernels) | 44 | 4,726 |
| `third_party/ggml-quants/` (vendored ggml CUDA quant stack: `mmq.cuh`, `mmq-load-tiles.cuh`, `vecdotq.cuh`, `dequantize.cuh`, per-arch MMQ configs) | 29 | 22,400 |
| wiring into `linear.cpp`, `weight.h`, `weight_view.cpp`, `formats.cpp`, `weight_input.cpp` and ~15 `ops/wrapper/*` | ~20 | ~3,000 changed |

Roughly **27,000 lines of vendored and new CUDA/C++ plus edits across ~25 existing files** -- and it
would make this fork carry a second, permanently-diverging copy of llama.cpp's quantization kernels.
That is an integration project, not a change to land alongside a measurement.

The cheaper alternative, if the 1.17x artifact effect is what matters: teach the **Python converter**
to read GGUF blocks and re-encode them as this engine's supported `q4_g64_fp16` / `q5_g64_fp16`
groupwise formats. That keeps the fast prefill and needs no CUDA work, at the cost of a coarser
quantization than the per-tensor ggml mix (expect ~13 GiB resident rather than 10.9, so ~1.15x
rather than 1.17x).

### 9.4 Environment note

Both fork builds (one on `E:`, one on `D:`) were deleted within minutes of finishing, after
producing the numbers above: `Test-Path` alternated between true and false, `icacls` reported
"cannot find the path", and `E:` also kept stale directory entries (`infer-all-src` listed by the
parent while unopenable). No Defender detection event was logged, though its intelligence updated
and its configuration changed mid-run. The second build succeeded only after the binaries were
copied out of the build tree immediately after linking, into
`E:\workspaces\ai\tessera-bench\fork-bin\`; that copy is what produced the table in 9.2. Nothing in
this repository is affected.