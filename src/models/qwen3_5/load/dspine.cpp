#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

void bind_dspine(Bindings& b, DraftWeights& weights, const DraftConfig& config,
                 const TextConfig& target) {
    const auto& extra = config.dspine.value();
    const auto& code  = extra.bridge;
    const auto h      = target.hidden_size;
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = "dspine/layers/" + std::to_string(i) + "/bridge/";
        weights.layers[i].bridge = DraftBridgeWeights{
            b.direct(p + "read", {code.code_size, h}),
            b.direct(p + "write", {h, code.rank}),
        };
    }
    weights.dspine = DSpineBridgeWeights{
        b.direct("dspine/bridge/codebook", {target.vocab_size, code.code_size}),
        b.direct("dspine/bridge/gate_projection", {code.rank, code.code_size}),
        b.direct("dspine/bridge/gate_bias", {code.rank}),
        b.direct("dspine/bridge/predecessor_projection", {code.rank, code.code_size}),
        b.direct("dspine/bridge/value_projection", {code.rank, code.code_size}),
        b.direct("dspine/bridge/final_read", {code.code_size, h}),
    };
}

} // namespace ninfer::models::qwen3_5::loading