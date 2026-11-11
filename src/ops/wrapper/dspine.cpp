#include "ninfer/ops/dspine.h"

#include "ops/dspine/dspine_launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(const Tensor& tensor, DType dtype, std::initializer_list<std::int32_t> shape,
             const char* op, const char* name) {
    std::int32_t expected[4] = {1, 1, 1, 1};
    std::size_t axis         = 0;
    for (const std::int32_t extent : shape) { expected[axis++] = extent; }
    if (tensor.dtype != dtype || tensor.data == nullptr || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
    for (int index = 0; index < 4; ++index) {
        if (tensor.ne[index] != expected[index]) {
            throw std::invalid_argument(std::string(op) + ": invalid " + name + " shape");
        }
    }
}

void require_blocks(const Tensor& residual, std::int32_t block_width, const char* op) {
    if (block_width < 2 || block_width > 16) {
        throw std::invalid_argument(std::string(op) + ": block width must be in [2,16]");
    }
    const std::int32_t batch = residual.ne[2];
    if (batch < 1 || batch > 8 || residual.ne[0] != kDSpineHidden ||
        residual.ne[1] != block_width || residual.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid residual block geometry");
    }
}

void require_bridge_weights(const Tensor& codebook, const Tensor& gate, const Tensor& predecessor,
                            const Tensor& value, const char* op) {
    // A stored [rows,cols] weight is row-major with the column contiguous, so the Tensor's
    // fastest extent is `cols` (NInfer's ne[0]).
    require(codebook, DType::BF16, {kDSpineCode, 248320}, op, "codebook");
    for (const auto* weight : {&gate, &predecessor, &value}) {
        require(*weight, DType::BF16, {kDSpineCode, kDSpineRank}, op, "bridge projection");
    }
}

void require_head(const Weight& head, std::int32_t token_domain, const char* op) {
    if (token_domain != kDSpineTokenDomain) {
        throw std::invalid_argument(std::string(op) + ": token domain must be 248077");
    }
    if (head.n != 248320 || head.k != kDSpineHidden || head.ndim != 2) {
        throw std::invalid_argument(std::string(op) + ": head must be [248320,5120]");
    }
    if (head.qtype == QType::BF16) {
        if (head.layout != QuantLayout::Contiguous || head.payload == nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 head must be contiguous");
        }
        return;
    }
    if (head.qtype != QType::Q8_G32_FP16 || head.layout != QuantLayout::RowSplit ||
        head.group_size != 32 || head.group != 32 || head.scale_dtype != DType::FP16 ||
        head.qdata == nullptr || head.scales == nullptr || head.qhigh != nullptr) {
        throw std::invalid_argument(std::string(op) + ": unsupported head format");
    }
}

} // namespace

void dspine_bridge_read(const Tensor& residual, const Tensor& read, const Tensor& codebook,
                        const Tensor& gate, const Tensor& predecessor, const Tensor& value,
                        const Tensor& anchors, std::int32_t block_width, float sqrt_code, float eps,
                        Tensor& normalized, Tensor& projected, Tensor& codes, Tensor& rms,
                        Tensor& packed, cudaStream_t stream) {
    constexpr const char* op = "dspine_bridge_read";
    require_blocks(residual, block_width, op);
    const std::int32_t columns = block_width * residual.ne[2];
    require(read, DType::BF16, {kDSpineHidden, kDSpineCode}, op, "read weight");
    require_bridge_weights(codebook, gate, predecessor, value, op);
    require(anchors, DType::I32, {residual.ne[2]}, op, "anchors");
    require(normalized, DType::BF16, {kDSpineHidden, columns}, op, "normalized");
    require(projected, DType::FP32, {columns, kDSpineCode}, op, "projected");
    require(codes, DType::BF16, {columns, kDSpineCode}, op, "codes");
    require(rms, DType::FP32, {columns}, op, "rms");
    require(packed, DType::BF16, {columns, 3 * kDSpineRank}, op, "packed");
    if (!(eps > 0.0F) || !(sqrt_code > 0.0F)) {
        throw std::invalid_argument("dspine_bridge_read: eps and sqrt_code must be positive");
    }
    detail::dspine_bridge_read_launch(residual, read, codebook, gate, predecessor, value, anchors,
                                      block_width, sqrt_code, eps, normalized, projected, codes,
                                      rms, packed, stream);
}

void dspine_bridge_write(const Tensor& residual, const Tensor& rms, const Tensor& packed,
                         const Tensor& gate_bias, const Tensor& write, const Tensor& next_norm,
                         std::int32_t block_width, float eps, Tensor& message, Tensor& out,
                         Tensor& normalized, Tensor& packed_tail, cudaStream_t stream) {
    constexpr const char* op = "dspine_bridge_write";
    require_blocks(residual, block_width, op);
    const std::int32_t batch   = residual.ne[2];
    const std::int32_t columns = block_width * batch;
    require(rms, DType::FP32, {columns}, op, "rms");
    require(packed, DType::BF16, {columns, 3 * kDSpineRank}, op, "packed");
    require(gate_bias, DType::BF16, {kDSpineRank}, op, "gate bias");
    require(write, DType::BF16, {kDSpineRank, kDSpineHidden}, op, "write weight");
    require(next_norm, DType::BF16, {kDSpineHidden}, op, "next norm");
    require(message, DType::BF16, {columns, kDSpineRank}, op, "message");
    require(out, DType::BF16, {kDSpineHidden, block_width, batch}, op, "out");
    require(normalized, DType::BF16, {kDSpineHidden, block_width, batch}, op, "normalized");
    if (packed_tail.data != nullptr) {
        require(packed_tail, DType::BF16, {kDSpineHidden, (block_width - 1) * batch}, op,
                "packed tail");
    }
    if (!(eps > 0.0F)) { throw std::invalid_argument("dspine_bridge_write: eps must be positive"); }
    detail::dspine_bridge_write_launch(residual, rms, packed, gate_bias, write, next_norm,
                                       block_width, eps, message, out, normalized, packed_tail,
                                       stream);
}

void dspine_refine_score(const Tensor& raw, const Tensor& rms, const Tensor& packed,
                         const Tensor& candidates, const Tensor& anchors, const Tensor& codebook,
                         const Tensor& gate_bias, const Tensor& predecessor, const Tensor& value,
                         const Tensor& write, const Tensor& norm, const Weight& head,
                         std::int32_t token_domain, std::int32_t block_width, float sqrt_code,
                         float eps, DSpineRefineBuffers& buffers, cudaStream_t stream) {
    constexpr const char* op = "dspine_refine_score";
    require_blocks(raw, block_width, op);
    const std::int32_t batch   = raw.ne[2];
    const std::int32_t columns = block_width * batch;
    const std::int32_t rows    = (block_width - 1) * batch;
    const std::int32_t sender_rows =
        batch * (1 + (block_width - 2) * kDSpineCandidates);
    require(rms, DType::FP32, {columns}, op, "rms");
    require(packed, DType::BF16, {columns, 3 * kDSpineRank}, op, "packed");
    require(candidates, DType::I32, {kDSpineCandidates, rows}, op, "candidates");
    require(anchors, DType::I32, {batch}, op, "anchors");
    require(codebook, DType::BF16, {kDSpineCode, 248320}, op, "codebook");
    require(gate_bias, DType::BF16, {kDSpineRank}, op, "gate bias");
    require(predecessor, DType::BF16, {kDSpineCode, kDSpineRank}, op, "predecessor projection");
    require(value, DType::BF16, {kDSpineCode, kDSpineRank}, op, "value projection");
    require(write, DType::BF16, {kDSpineRank, kDSpineHidden}, op, "write weight");
    require(norm, DType::BF16, {kDSpineHidden}, op, "norm");
    require_head(head, token_domain, op);
    require(buffers.sender_codes, DType::BF16, {kDSpineCode, sender_rows}, op, "sender codes");
    require(buffers.sender_tsv, DType::BF16, {2 * kDSpineRank, sender_rows}, op, "sender tsv");
    require(buffers.candidate_rows, DType::BF16, {kDSpineHidden, rows * kDSpineCandidates}, op,
            "candidate rows");
    require(buffers.candidate_tg, DType::BF16, {kDSpineRank, rows * kDSpineCandidates}, op,
            "candidate tg");
    require(buffers.base, DType::FP32, {kDSpineCandidates, rows}, op, "base");
    require(buffers.table, DType::FP32,
            {kDSpineCandidates * kDSpineCandidates, rows}, op, "table");
    if (!(eps > 0.0F) || !(sqrt_code > 0.0F)) {
        throw std::invalid_argument("dspine_refine_score: eps and sqrt_code must be positive");
    }
    detail::dspine_refine_score_launch(raw, rms, packed, candidates, anchors, codebook, gate_bias,
                                       predecessor, value, write, norm, head, token_domain,
                                       block_width, sqrt_code, eps, buffers, stream);
}

void dspine_refine_chain(const Tensor& table, const Tensor& candidates,
                         const SamplingConfig* configs, const Tensor& base_positions,
                         std::int32_t block_width, Tensor& drafts, Tensor& proposal_q,
                         cudaStream_t stream) {
    constexpr const char* op = "dspine_refine_chain";
    if (block_width < 2 || block_width > 16) {
        throw std::invalid_argument("dspine_refine_chain: block width must be in [2,16]");
    }
    const std::int32_t batch = drafts.ne[1];
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("dspine_refine_chain: batch must be in [1,8]");
    }
    const std::int32_t rows = (block_width - 1) * batch;
    require(table, DType::FP32, {kDSpineCandidates * kDSpineCandidates, rows}, op, "table");
    require(candidates, DType::I32, {kDSpineCandidates, rows}, op, "candidates");
    require(base_positions, DType::I32, {batch}, op, "base positions");
    require(drafts, DType::I32, {block_width - 1, batch}, op, "drafts");
    require(proposal_q, DType::FP32, {kDSpineCandidates, block_width - 1, batch}, op,
            "proposal q");
    if (configs == nullptr) {
        throw std::invalid_argument("dspine_refine_chain: configs must be non-null");
    }
    detail::dspine_refine_chain_launch(table, candidates, configs, base_positions, block_width,
                                       drafts, proposal_q, stream);
}

} // namespace ninfer::ops