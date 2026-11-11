#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

// Shape convention: these Ops are written against row-major logical matrices whose columns are
// contiguous, because they index `row * columns + column`. NInfer tensors store their first
// extent contiguously, so a logical [rows, columns] buffer is a Tensor {columns, rows}.
// The stored bridge weights follow the same rule (a stored [rows, cols] weight is
// {cols, rows}); the 5120-wide 4-D residual blocks are plain {hidden, width, batch}.

// DSpine draft geometry. The bridge and the refinement are registered for exactly the
// Qwen3.8-27B DSpine draft: a 5120-wide residual stream, 1024-wide codes, rank 512 and a
// sixteen-wide candidate head over a 248320-row vocabulary with 248077 public rows.
inline constexpr std::int32_t kDSpineHidden     = 5120;
inline constexpr std::int32_t kDSpineCode       = 1024;
inline constexpr std::int32_t kDSpineRank       = 512;
inline constexpr std::int32_t kDSpineCandidates = 16;
inline constexpr std::int32_t kDSpineMaxColumns = 128; // block width <= 16, batch <= 8
inline constexpr std::int32_t kDSpineTokenDomain = 248077;

/**
 * @brief One bridge read after a draft layer's MLP: RMS-normalize the residual, project it to
 * codes, replace the block's anchor row with the anchor's codebook row, and project the codes
 * into the [gate; predecessor; value] message operands.
 *
 * @details Ideal arithmetic, for block width W, batch B and column `n = t*B + b`:
 *
 *   f       = FP32(residual[:, t, b])
 *   rms[n]  = sqrt(max(mean(f*f), 1.1754943508222875e-38))       // FP32 tiny, not eps
 *   q[d]    = BF16(f[d] / (rms[n] + eps))
 *   p[c,n]  = sum_d FP32(q[d]) * FP32(read[c,d])                 // FP32 accumulation
 *   codes[c,n] = t == 0 ? BF16(sqrt_code * codebook[anchor[b], c])
 *                       : BF16(sqrt_code * p[c,n] / (sqrt(sum_c p[:,n]^2) + eps))
 *   packed[k,n] = BF16(sum_c FP32(codes[c,n]) * FP32(P[k,c]))
 *
 * with `P = [gate; predecessor; value]` (rank rows each). `packed` and `codes` are BF16 because
 * every consumer reproduces this stored rounding; `rms` and `projected` stay FP32.
 *
 * `q` is never materialized: it lives in the caller's `normalized` scratch [hidden, W*B].
 * `residual` is BF16 [hidden,W,B] contiguous, `read` BF16 [code,hidden], `codebook` BF16
 * [vocab,code], and each of `gate`/`predecessor`/`value` BF16 [rank,code]; `anchors` is
 * contiguous I32 [B]. `projected` is caller scratch FP32 [code,W*B]. All tensors are contiguous
 * and pairwise non-overlapping; every output element is overwritten. No device allocation, no
 * persistent state, no synchronization.
 */
void dspine_bridge_read(const Tensor& residual, const Tensor& read, const Tensor& codebook,
                        const Tensor& gate, const Tensor& predecessor, const Tensor& value,
                        const Tensor& anchors, std::int32_t block_width, float sqrt_code, float eps,
                        Tensor& normalized, Tensor& projected, Tensor& codes, Tensor& rms,
                        Tensor& packed, cudaStream_t stream);

/**
 * @brief One bridge write: fold the codebook message back into the residual and normalize the
 * next layer's input.
 *
 * @details For column `n = t*B + b`, with the predecessor row read at column `n - B` (the same
 * request's previous block row):
 *
 *   gate[r] = sigmoid(FP32(packed[r,n]) + FP32(packed[rank+r,n-B]) + FP32(gate_bias[r]))
 *   msg[r]  = t == 0 ? 0 : BF16(gate[r] * FP32(packed[2*rank+r,n-B]))
 *   delta[d]= sum_r FP32(msg[r]) * FP32(write[d,r])              // FP32 accumulation
 *   out[d,n] = BF16(FP32(residual[d,n]) + (t != 0 ? rms[n]*delta[d] : 0))
 *   y[d]    = BF16(FP32(out[d,n]) * rsqrt(mean_d(out[.,n]^2) + eps))
 *   next[d] = BF16(FP32(y[d]) * FP32(next_norm[d]))
 *
 * `next` is materialized into `normalized` [hidden,W,B]; when `packed_tail` is present it also
 * receives rows t>=1 packed as [hidden,(W-1)B] (column `(W-1)b + t-1`), which is the proposal
 * hidden the refinement head consumes. `message` is caller scratch BF16 [rank,W*B].
 *
 * The write reads `residual` and writes `out`; the two must not overlap, because the refinement
 * needs the pre-write raw stream. All tensors are contiguous BF16 as named and pairwise
 * non-overlapping otherwise. No device allocation, no persistent state, no synchronization.
 */
void dspine_bridge_write(const Tensor& residual, const Tensor& rms, const Tensor& packed,
                         const Tensor& gate_bias, const Tensor& write, const Tensor& next_norm,
                         std::int32_t block_width, float eps, Tensor& message, Tensor& out,
                         Tensor& normalized, Tensor& packed_tail, cudaStream_t stream);

// Caller-owned scratch and outputs of dspine_refine_score. Every tensor is contiguous and
// sized for the served block width K and batch B:
//   sender_codes   BF16 [B*(1+(K-1)*16), code]
//   sender_tsv     BF16 [B*(1+(K-1)*16), 2*rank]
//   candidate_rows BF16 [K*B*16, hidden]
//   candidate_tg   BF16 [K*B*16, rank]
//   base           FP32 [16, K*B]
//   table          FP32 [K*B, 16, 16]
struct DSpineRefineBuffers {
    Tensor sender_codes;
    Tensor sender_tsv;
    Tensor candidate_rows;
    Tensor candidate_tg;
    Tensor base;
    Tensor table;
};

/**
 * @brief Serial last-write refinement, scoring phase: build the [K,K] proposal transition table
 * of every proposal row.
 *
 * @details `candidates` holds the stable top-16 ids per proposal column (`linear_topk` order,
 * candidate fastest). For proposal row `i` (column `K*b + i`, block position `t = i+1`) and
 * sender rank `s`, the sender token is the anchor for `i == 0` and `candidates[s, K*b + i - 1]`
 * otherwise. With `c` the candidate rank of row `i`:
 *
 *   e[c]        = BF16(sqrt_code * FP32(codebook[sender,c]))
 *   tsv[r]      = BF16(sum_c FP32(e[c]) * FP32([predecessor; value][r,c]))
 *   m[r]        = sigmoid(FP32(q[r]) + tsv[gate,r] + FP32(gate_bias[r])) * tsv[value,r]
 *   headrow     = BF16-dequantized target head row of candidates[c]
 *   a[h]        = BF16(FP32(headrow[h]) * FP32(norm[h]))
 *   base[c]     = sum_h FP32(headrow[h]) * FP32(norm[h]) * FP32(raw[h])
 *   tg[c,r]     = BF16(sum_h FP32(a[h]) * FP32(write[h,r]))
 *   table[i,s,c] = base[c] + rms * sum_r m[r] * FP32(tg[c,r])
 *
 * `raw` is the last layer's pre-write residual BF16 [hidden,W,B] and `rms` its bridge RMS FP32
 * [W*B]; `packed` is the last bridge read's output BF16 [3*rank,W*B], of which the first `rank`
 * rows are the query half. `head` is the full-vocabulary target head (Q8_G32_FP16 RowSplit or
 * contiguous BF16), `token_domain` bounds its live rows, and `candidates` must be in
 * [0,token_domain). Outputs are completely overwritten, inputs preserved, no device allocation.
 */
void dspine_refine_score(const Tensor& raw, const Tensor& rms, const Tensor& packed,
                         const Tensor& candidates, const Tensor& anchors, const Tensor& codebook,
                         const Tensor& gate_bias, const Tensor& predecessor, const Tensor& value,
                         const Tensor& write, const Tensor& norm, const Weight& head,
                         std::int32_t token_domain, std::int32_t block_width, float sqrt_code,
                         float eps, DSpineRefineBuffers& buffers, cudaStream_t stream);

/**
 * @brief Serial last-write refinement, walk phase: turn the transition table into draft tokens
 * and the proposal distributions the verifier consumes.
 *
 * @details Greedy rows (`configs[b].temperature <= 0`) walk the table as
 * `j = argmax_c table[0, 0, c]` with exact ties resolved to the lower candidate token id, then
 * `j = argmax_c table[i, j, c]`; the selected token is `candidates[j, K*b + i]`.
 *
 * Positive-temperature rows exponentiate `table[i, sender, :] / temperature`, apply the same
 * joint top-k / top-p / min-p filter as the reference proposal distribution (rank by strict
 * descending probability with the lower lane winning exact ties, `before` the mass strictly
 * ahead), renormalize, and draw with the counter-based uniform at
 * `(seed, base_positions[b]+i, kSamplePurposeDSpineProposal)`. `proposal_q[c, K*b+i]` receives
 * exactly the distribution `drafts` was drawn from, so the verifier's standard speculative
 * rejection applies. Greedy rows write the one-hot distribution of their walk.
 *
 * `table` is FP32 [K*B,16,16], `candidates` I32 [16,K*B], `drafts` I32 [K,B], `proposal_q` FP32
 * [16,K,B], `configs` a device-resident SamplingConfig[B] array and `base_positions` I32 [B] the
 * RNG position of the block's first proposal. Outputs are completely overwritten; no device
 * allocation.
 */
void dspine_refine_chain(const Tensor& table, const Tensor& candidates,
                         const SamplingConfig* configs, const Tensor& base_positions,
                         std::int32_t block_width, Tensor& drafts, Tensor& proposal_q,
                         cudaStream_t stream);

} // namespace ninfer::ops