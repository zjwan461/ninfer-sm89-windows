#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/dspine.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dspine_bridge_read_launch(const Tensor& residual, const Tensor& read, const Tensor& codebook,
                               const Tensor& gate, const Tensor& predecessor, const Tensor& value,
                               const Tensor& anchors, std::int32_t block_width, float sqrt_code,
                               float eps, Tensor& normalized, Tensor& projected, Tensor& codes,
                               Tensor& rms, Tensor& packed, cudaStream_t stream);

void dspine_bridge_write_launch(const Tensor& residual, const Tensor& rms, const Tensor& packed,
                                const Tensor& gate_bias, const Tensor& write,
                                const Tensor& next_norm, std::int32_t block_width, float eps,
                                Tensor& message, Tensor& out, Tensor& normalized,
                                Tensor& packed_tail, cudaStream_t stream);

void dspine_refine_score_launch(const Tensor& raw, const Tensor& rms, const Tensor& packed,
                                const Tensor& candidates, const Tensor& anchors,
                                const Tensor& codebook, const Tensor& gate_bias,
                                const Tensor& predecessor, const Tensor& value, const Tensor& write,
                                const Tensor& norm, const Weight& head,
                                std::int32_t token_domain, std::int32_t block_width,
                                float sqrt_code, float eps, DSpineRefineBuffers& buffers,
                                cudaStream_t stream);

void dspine_refine_chain_launch(const Tensor& table, const Tensor& candidates,
                                const SamplingConfig* configs, const Tensor& base_positions,
                                std::int32_t block_width, Tensor& drafts, Tensor& proposal_q,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail