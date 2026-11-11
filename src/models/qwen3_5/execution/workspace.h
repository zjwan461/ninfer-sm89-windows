#pragma once

// Phase allocations shared by actual execution and startup sizing.

#include "core/arena.h"
#include "core/layout.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/dspine.h"

#include <cstdint>

namespace ninfer::models::qwen3_5::execution::workspace {

template <class Allocator>
Tensor matrix(Allocator& allocator, DType dtype, std::int32_t rows, std::int32_t tokens) {
    return allocator.alloc(dtype, {rows, tokens});
}

template <class Allocator>
Tensor vector(Allocator& allocator, DType dtype, std::int32_t elements) {
    return allocator.alloc(dtype, {elements});
}

struct TextPrefillRoots {
    Tensor ids;
    Tensor positions;
    Tensor rope_positions;
    Tensor residual;
    Tensor scatter_indices;
};

template <class Allocator>
TextPrefillRoots text_prefill_roots(Allocator& allocator, const TextConfig& config,
                                    std::int32_t tokens, std::int32_t rope_axes,
                                    std::int32_t scatter_tokens) {
    TextPrefillRoots out;
    out.ids       = vector(allocator, DType::I32, tokens);
    out.positions = vector(allocator, DType::I32, tokens);
    if (rope_axes != 0) { out.rope_positions = matrix(allocator, DType::I32, tokens, rope_axes); }
    out.residual = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    if (scatter_tokens != 0) {
        out.scatter_indices = vector(allocator, DType::I32, scatter_tokens);
    }
    return out;
}

template <class Allocator>
Tensor visual_scatter_indices(Allocator& allocator, std::int32_t tokens) {
    return vector(allocator, DType::I32, tokens);
}

struct TextAttentionProjectionRoots {
    Tensor hidden;
    Tensor query;
    Tensor gate;
    Tensor key;
    Tensor value;
};

template <class Allocator>
TextAttentionProjectionRoots
text_attention_projection(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
    };
}

struct TextAttentionResultRoots {
    Tensor normalized_query;
    Tensor normalized_key;
    Tensor attention;
};

template <class Allocator>
TextAttentionResultRoots text_attention_results(Allocator& allocator, const TextConfig& config,
                                                std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
    };
}

struct GdnControlRoots {
    Tensor hidden;
    Tensor g;
    Tensor beta;
};

template <class Allocator>
GdnControlRoots gdn_control(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::FP32, dimension(config.gdn->linear_num_value_heads), tokens),
        matrix(allocator, DType::FP32, dimension(config.gdn->linear_num_value_heads), tokens),
    };
}

struct GdnProjectionRoots {
    Tensor output_gate;
    Tensor query;
    Tensor key;
    Tensor value;
};

template <class Allocator>
GdnProjectionRoots gdn_projection(Allocator& allocator, const TextConfig& config,
                                  std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.gdn->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.gdn->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens),
    };
}

template <class Allocator>
Tensor gdn_prefill_conv(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.gdn->conv_channels()), tokens);
}

template <class Allocator>
Tensor gdn_recurrent_output(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens);
}

template <class Allocator>
Tensor gdn_normalized_output(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens);
}

template <class Allocator>
Tensor post_mixer_hidden(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
}

struct MtpStemRoots {
    Tensor embedding;
    Tensor normalized_embedding;
    Tensor normalized_hidden;
    Tensor packed_input;
    Tensor residual;
    Tensor attention_hidden;
};

template <class Allocator>
MtpStemRoots mtp_stem(Allocator& allocator, const TextConfig& config, std::int32_t tokens,
                      bool allocate_embedding) {
    MtpStemRoots out;
    if (allocate_embedding) {
        out.embedding = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    }
    out.normalized_embedding =
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.normalized_hidden = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.packed_input = matrix(allocator, DType::BF16, dimension(2ULL * config.hidden_size), tokens);
    out.residual     = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.attention_hidden = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    return out;
}

struct MtpAttentionProjectionRoots {
    Tensor query;
    Tensor key;
    Tensor gate;
    Tensor value;
};

template <class Allocator>
MtpAttentionProjectionRoots mtp_attention_projection(Allocator& allocator, const TextConfig& config,
                                                     std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
    };
}

struct MtpAttentionResultRoots {
    Tensor normalized_query;
    Tensor normalized_key;
    Tensor attention;
};

template <class Allocator>
MtpAttentionResultRoots mtp_attention_results(Allocator& allocator, const TextConfig& config,
                                              std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
    };
}

struct MtpPostAttentionRoots {
    Tensor output;
    Tensor post_mixer_hidden;
};

template <class Allocator>
MtpPostAttentionRoots mtp_post_attention(Allocator& allocator, const TextConfig& config,
                                         std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
    };
}

struct DFlashContextRoots {
    Tensor projected;
    Tensor normalized;
};

template <class Allocator>
DFlashContextRoots dflash_context(Allocator& allocator, const TextConfig& target,
                                  const DraftConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
    };
}

struct DFlashContextLayerRoots {
    Tensor key_raw;
    Tensor value;
    Tensor key;
};

template <class Allocator>
DFlashContextLayerRoots dflash_context_layer(Allocator& allocator, const TextConfig& target,
                                             const DraftConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
    };
}

struct DFlashProposalRoots {
    Tensor ids;
    Tensor positions;
    Tensor residual;
};

template <class Allocator>
DFlashProposalRoots dflash_proposal(Allocator& allocator, const TextConfig& target,
                                    const DraftConfig& config, std::int32_t tokens) {
    return {
        vector(allocator, DType::I32, tokens),
        vector(allocator, DType::I32, tokens),
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
    };
}

struct DFlashAttentionRoots {
    Tensor hidden;
    Tensor query_raw;
    Tensor key_raw;
    Tensor value;
    Tensor query;
    Tensor key;
    Tensor attention;
};

template <class Allocator>
DFlashAttentionRoots dflash_attention(Allocator& allocator, const TextConfig& target,
                                      const DraftConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
    };
}

// Dynamic-convolution outputs remain live through the corresponding attention/MLP branch.
struct DFlash2BranchRoots {
    Tensor prepared;
    Tensor finish_delta;
};

template <class Allocator>
DFlash2BranchRoots dflash2_branch(Allocator& allocator, const TextConfig& target,
                                  const DraftConfig& config, std::int32_t width,
                                  std::int32_t batch) {
    return {allocator.alloc(DType::BF16, {dimension(target.hidden_size), width, batch}),
            allocator.alloc(DType::BF16,
                            {dimension(target.hidden_size / config.dflash2->conv_group_size), 2,
                             width, batch})};
}

struct DFlashMlpRoots {
    Tensor hidden;
    Tensor intermediate;
};

template <class Allocator>
DFlashMlpRoots dflash_mlp(Allocator& allocator, const TextConfig& target, const DraftConfig& config,
                          std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.intermediate_size), tokens),
    };
}

// DSpine draft-block attention. The layer input is the previous bridge write's normalized output,
// so no separate hidden scratch is needed.
struct DSpineAttentionRoots {
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor attention;
};

template <class Allocator>
DSpineAttentionRoots dspine_attention(Allocator& allocator, const TextConfig& target,
                                      const DraftConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
    };
}

// Everything the DSpine proposal phase allocates: two residual streams (the bridge write must keep
// the pre-write raw stream), the per-layer bridge operands, and the refinement buffers.
struct DSpineRoundRoots {
    Tensor residual_a;
    Tensor residual_b;
    Tensor layer_input;
    Tensor read_q;
    Tensor projected;
    Tensor codes;
    Tensor rms;
    Tensor packed;
    Tensor message;
    Tensor proposal_hidden;
    Tensor candidate_scores;
    ops::DSpineRefineBuffers refine;
};

template <class Allocator>
DSpineRoundRoots dspine_round(Allocator& allocator, const TextConfig& target,
                              const DraftConfig& config, std::int32_t width,
                              std::int32_t batch) {
    const auto& bridge = *config.dspine;
    const std::int32_t columns  = width * batch;
    const std::int32_t rows     = (width - 1) * batch;
    const std::int32_t senders  = batch * (1 + (width - 2) * ops::kDSpineCandidates);
    DSpineRoundRoots out;
    // The bridge and refinement ops store a logical [rows,columns] matrix with the *column*
    // contiguous (the DSpine kernels index `row*columns + column`), which is the Tensor
    // {columns, rows}: the fastest extent is the column count.
    const auto row_major = [&allocator](DType dtype, std::int32_t rows, std::int32_t columns) {
        return allocator.alloc(dtype, {columns, rows});
    };
    out.residual_a = allocator.alloc(DType::BF16, {dimension(target.hidden_size), width, batch});
    out.residual_b = allocator.alloc(DType::BF16, {dimension(target.hidden_size), width, batch});
    out.layer_input = allocator.alloc(DType::BF16, {dimension(target.hidden_size), width, batch});
    out.read_q     = matrix(allocator, DType::BF16, dimension(target.hidden_size), columns);
    out.projected =
        row_major(DType::FP32, dimension(bridge.bridge.code_size), columns);
    out.codes = row_major(DType::BF16, dimension(bridge.bridge.code_size), columns);
    out.rms   = vector(allocator, DType::FP32, columns);
    out.packed = row_major(DType::BF16, dimension(3ULL * bridge.bridge.rank), columns);
    out.message = row_major(DType::BF16, dimension(bridge.bridge.rank), columns);
    out.proposal_hidden = matrix(allocator, DType::BF16, dimension(target.hidden_size), rows);
    out.candidate_scores = matrix(allocator, DType::FP32, dimension(ops::kDSpineCandidates), rows);
    out.refine.sender_codes = matrix(allocator, DType::BF16, dimension(bridge.bridge.code_size),
                                     senders);
    out.refine.sender_tsv   = matrix(allocator, DType::BF16, dimension(2ULL * bridge.bridge.rank),
                                   senders);
    out.refine.candidate_rows = matrix(
        allocator, DType::BF16, dimension(target.hidden_size),
        dimension(std::uint64_t(rows) * ops::kDSpineCandidates));
    out.refine.candidate_tg = matrix(allocator, DType::BF16, dimension(bridge.bridge.rank),
                                     dimension(std::uint64_t(rows) * ops::kDSpineCandidates));
    out.refine.base  = matrix(allocator, DType::FP32, dimension(ops::kDSpineCandidates), rows);
    out.refine.table = allocator.alloc(
        DType::FP32, {dimension(ops::kDSpineCandidates * ops::kDSpineCandidates), rows});
    return out;
}

} // namespace ninfer::models::qwen3_5::execution::workspace
