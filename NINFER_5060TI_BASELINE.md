# NInfer on the 16 GB RTX 5060 Ti (sm_120a, 36 SM) - measured baseline

> Card: RTX 5060 Ti, GB206, **36 SMs**, 16311 MiB, compute capability 12.0, driver 617.14.
> Build: `build-5060ti` (`CMAKE_CUDA_ARCHITECTURES=120a`, `NINFER_TARGET_SM_COUNT=36`,
> Release, VS 2026 18.11.3 / MSVC 14.51, CUDA 13.4, Windows 11).
> Artifact: `bonsai2_27b_vl_mtp_q4q5.ninfer` (6,869,420,800 B,
> sha256 `96f1f241753d8e6e01c3f742907523a5b78a2c6dd0eee1f81114a61d29b4d671`, components
> text/vision/mtp).
> Every number below was measured on this machine on 2026-11-01. Each figure is a single run on
> the machine's only GPU, with the desktop left as-is (not a headless measurement).

## 1. Results

All decode runs use the code prompt *"Write a Python function that merges two sorted lists, with
a docstring and one example."*, `--kv-dtype rk4v4-e8`, `--no-thinking`, and CUDA graphs on.

| Configuration | Context | Generated | Prefill | Decode |
|---|---:|---:|---:|---:|
| No speculation | 8192 | 128 | 373 tok/s | **59.2 tok/s** |
| MTP k=2 | 8192 | 160 | 441 tok/s | **111.5 tok/s** |
| MTP k=3 | 8192 | 192 | 441 tok/s | **110.2 tok/s** |
| MTP k=2 | 131072 | 160 | 436 tok/s | **111.1 tok/s** |

MTP acceptance: k=2 -> 85.5 %, 2.69 tokens/round, 0 fallbacks; k=3 -> 82.9 %, 3.47 tokens/round.
k=2 and k=3 are within run-to-run noise on this prompt; the launcher keeps k=2.

The first four prefill figures are the short-prompt (30-62 token) path and include the fixed
per-prefill overhead, so they are not a `pp2048` number.

### Long-context needle

A 65,158-token prompt with the sentence *"The secret code is ORCHID-493817."* buried in the
middle, asked at the end, `--max-context 98304 --kv-dtype rk4v4-e8 --max-new 32`:

| Metric | Value |
|---|---:|
| Prefill | **1.32k tok/s** (65,158 prompt tokens in 49.2 s) |
| Decode at depth | **50.6 tok/s** (no speculation) |
| Answer | `ORCHID-493817` - exact |
| KV payload (rk4v4-e8) | 1.59 GiB |
| Planned device total | 7.43 GiB |

This run passes the deep-context split cap (the 36-SM geometry launches 72 splits past 34.5K
keys), so it also checks that the changed wave cap does not drop keys at depth.

For comparison, the pre-build estimate in the 5060 Ti analysis was 56-58 tok/s without
speculation, 87-97 tok/s with MTP k=2 and ~1.5-1.7k tok/s prefill at `pp2048`. The measured
decode lands at the top of / above that range; the 65K prefill is 1.32k tok/s.

## 2. Memory plan (MTP k=2, 131072 context, auto KV)

| Item | Size |
|---|---:|
| Weights (text + vision + MTP) | 6.11 GiB |
| Text KV (rk4v4-e8) | 2.12 GiB |
| MTP KV | 136 MiB |
| GDN state | 147 MiB |
| Replay records | 27 MiB |
| Workspace peak | 153 MiB |
| CUDA Graph allowance | 176 MiB |
| **Planned device total** | **8.88 GiB** |
| Free after startup | 6.10 GiB |

Weight load from `E:` runs at ~137 MiB/s (41-45 s); engine-ready is 42 s cold, 48 s with MTP.
The server reached `/v1/models` in 26 s with the weights still in the OS file cache.

## 3. Correctness checks

- CLI (MTP, `--no-thinking`, 160 tokens) returns a complete `merge_sorted_lists` implementation
  with docstring, loop and example - the requested output.
- Long-context needle: exact `ORCHID-493817` at 65,158 prompt tokens (see section 1).
- HTTP server (`ninfer-serve`, 131072 context, MTP, `--vision`, concurrency 2) answered
  `17 x 23` with `391` and a well-formed OpenAI Chat Completions usage block.

## 4. Limitations of this build

- **NVFP4 W4A4 TMA is not compiled.** MSVC rejects nvcc's host stub for the
  `__grid_constant__ CUtensorMap` parameter (`alignas(128)` -> C2719), so the two TMA launcher
  translation units are excluded and `AllowA4` artifacts run their activations at A16. The
  NVFP4-valued KV modes are compiled but were not exercised. Bonsai (ternary weights,
  `rk4v4-e8` KV) does not use either path. `NINFER_BUILD_NVFP4_TMA=ON` re-enables the TUs on a
  toolchain that accepts the aligned parameter.
- 262144 x 1 lane was not measured; 131072 x 2 is the tested configuration and leaves 6.1 GiB.
- `--prefill-chunk 768` is a starting point for 36 SMs, not a tuned optimum.
- Vision was loaded but no image request was sent yet.
- These are single runs; repeat before quoting a performance delta.

## 5. Reproduce

```bat
:: configure (once)
cmake -S . -B build-5060ti -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DNINFER_TARGET_SM_COUNT=36 ^
  -DCUDAToolkit_ROOT=<cuda-13.x> -DCMAKE_CUDA_COMPILER=<cuda-13.x>/bin/nvcc.exe
cmake --build build-5060ti -j

:: decode probe
build-5060ti\apps\ninfer.exe E:\models\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --prompt "Write a Python function that merges two sorted lists, with a docstring and one example." ^
  --max-context 8192 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 2 --lm-head-draft ^
  --ngram chain --no-thinking --max-new 160
```