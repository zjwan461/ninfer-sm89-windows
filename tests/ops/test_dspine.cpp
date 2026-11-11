// DSpine bridge and refinement-chain oracles.
//
// The bridge contract quantizes at eight named points (BF16 storage of xnorm, codes, packed,
// message, out and next) and leaves accumulation order private. The oracle below evaluates the
// same formula naively in double, applies exactly those BF16 roundings, and compares the stored
// BF16 output against it. The chain oracle walks an independently computed greedy/sampled table.

#include "ninfer/ops/dspine.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden     = ops::kDSpineHidden;
constexpr std::int32_t kCode       = ops::kDSpineCode;
constexpr std::int32_t kRank       = ops::kDSpineRank;
constexpr std::int32_t kCandidates = ops::kDSpineCandidates;
constexpr std::int32_t kCodebookRows = 248320;
constexpr std::int32_t kBlockWidth  = 8;
constexpr std::int32_t kBatch       = 2;

// BF16 storage is the contract; accumulation order and the pre-rounding FP32 values are private.
constexpr double kStorageAbsolute = 2.0e-6;
constexpr double kStorageRelative = 2.0e-2; // about 2.5 BF16 ULP

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float pattern(std::uint32_t a, std::uint32_t b, std::uint32_t salt, float scale) {
    const std::uint32_t mixed = mix32(a * 0x9e3779b9U ^ b * 0x85ebca6bU ^ salt);
    const int centered        = static_cast<int>((mixed >> 8) & 0xffU) - 128;
    return bf16_to_f32(f32_to_bf16(static_cast<float>(centered) * scale));
}

// Host BF16 rounding of a double, matching the device's round-to-nearest-even conversion.
double round_bf16(double value) {
    return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

// Logical [rows, columns] buffer with `columns` contiguous, as the operations index it.
std::size_t wide_index(std::int32_t row, std::int32_t column, std::int32_t columns) {
    return static_cast<std::size_t>(row) * columns + column;
}

std::size_t hidden_index(std::int32_t element, std::int32_t column) {
    return static_cast<std::size_t>(column) * kHidden + element;
}

double bf16_at(const std::vector<std::uint16_t>& bits, std::size_t index) {
    return static_cast<double>(bf16_to_f32(bits[index]));
}

int bridge_conformance() {
    const std::int32_t columns = kBlockWidth * kBatch;
    const std::int32_t rows    = (kBlockWidth - 1) * kBatch;
    const double sqrt_code     = std::sqrt(static_cast<double>(kCode));
    const double eps           = 1e-6;

    std::vector<std::uint16_t> codebook(static_cast<std::size_t>(kCodebookRows) * kCode, 0);
    std::vector<std::uint16_t> read(static_cast<std::size_t>(kCode) * kHidden);
    std::vector<std::uint16_t> gate(static_cast<std::size_t>(kRank) * kCode);
    std::vector<std::uint16_t> predecessor(gate.size());
    std::vector<std::uint16_t> value(gate.size());
    std::vector<std::uint16_t> gate_bias(kRank);
    std::vector<std::uint16_t> write(static_cast<std::size_t>(kHidden) * kRank);
    std::vector<std::uint16_t> next_norm(kHidden);
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(kHidden) * columns);
    std::vector<std::uint16_t> codes(static_cast<std::size_t>(kCode) * columns, 0);
    std::vector<std::uint16_t> packed(static_cast<std::size_t>(3 * kRank) * columns, 0);
    std::vector<float> rms(columns, 0.0F);
    std::vector<std::uint16_t> normalized(static_cast<std::size_t>(kHidden) * columns, 0);
    std::vector<std::uint16_t> layer_input(static_cast<std::size_t>(kHidden) * columns, 0);
    std::vector<std::uint16_t> projected(static_cast<std::size_t>(kCode) * columns, 0);
    std::vector<std::uint16_t> message(static_cast<std::size_t>(kRank) * columns, 0);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kHidden) * columns, 0);
    std::vector<std::uint16_t> packed_tail(static_cast<std::size_t>(kHidden) * rows, 0);
    std::int32_t anchors[kBatch] = {11, 7301};

    for (std::int32_t c = 0; c < kCode; ++c) {
        for (std::int32_t d = 0; d < kHidden; ++d) {
            read[wide_index(c, d, kHidden)] = f32_to_bf16(pattern(c, d, 17U, 1.0F / 512.0F));
        }
        for (std::int32_t r = 0; r < kRank; ++r) {
            gate[wide_index(r, c, kCode)]        = f32_to_bf16(pattern(r, c, 31U, 1.0F / 256.0F));
            predecessor[wide_index(r, c, kCode)] = f32_to_bf16(pattern(r, c, 47U, 1.0F / 256.0F));
            value[wide_index(r, c, kCode)]       = f32_to_bf16(pattern(r, c, 59U, 1.0F / 256.0F));
        }
    }
    for (std::int32_t r = 0; r < kRank; ++r) {
        gate_bias[r] = f32_to_bf16(pattern(r, 3U, 71U, 1.0F / 128.0F));
    }
    for (std::int32_t d = 0; d < kHidden; ++d) {
        next_norm[d] = f32_to_bf16(pattern(d, 5U, 83U, 1.0F / 64.0F) + 1.0F);
        for (std::int32_t r = 0; r < kRank; ++r) {
            write[wide_index(d, r, kRank)] = f32_to_bf16(pattern(d, r, 97U, 1.0F / 1024.0F));
        }
    }
    for (std::int32_t b = 0; b < kBatch; ++b) {
        for (std::int32_t t = 0; t < kBlockWidth; ++t) {
            const std::int32_t column = t + kBlockWidth * b;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                // The later rows grow, so rms spans a realistic range.
                const float scale = 1.0F + static_cast<float>(t);
                raw[hidden_index(d, column)] =
                    f32_to_bf16(pattern(static_cast<std::uint32_t>(d),
                                        static_cast<std::uint32_t>(column), 109U, 0.25F) *
                                scale);
            }
        }
        // Only the rows the block can name need codebook content.
        for (std::int32_t c = 0; c < kCode; ++c) {
            codebook[wide_index(anchors[b], c, kCode)] =
                f32_to_bf16(pattern(static_cast<std::uint32_t>(anchors[b]),
                                    static_cast<std::uint32_t>(c), 113U, 1.0F / 128.0F));
        }
    }

    GuardedDeviceBuffer d_codebook(codebook.size() * 2), d_read(read.size() * 2),
        d_gate(gate.size() * 2), d_predecessor(predecessor.size() * 2),
        d_value(value.size() * 2), d_gate_bias(gate_bias.size() * 2), d_write(write.size() * 2),
        d_next_norm(next_norm.size() * 2), d_raw(raw.size() * 2), d_codes(codes.size() * 2),
        d_packed(packed.size() * 2), d_rms(rms.size() * 4), d_normalized(normalized.size() * 2),
        d_layer_input(layer_input.size() * 2),
        d_projected(projected.size() * 4), d_message(message.size() * 2), d_out(out.size() * 2),
        d_tail(packed_tail.size() * 2), d_anchors(sizeof(anchors));
    d_codebook.copy_from_host(codebook.data(), d_codebook.bytes());
    d_read.copy_from_host(read.data(), d_read.bytes());
    d_gate.copy_from_host(gate.data(), d_gate.bytes());
    d_predecessor.copy_from_host(predecessor.data(), d_predecessor.bytes());
    d_value.copy_from_host(value.data(), d_value.bytes());
    d_gate_bias.copy_from_host(gate_bias.data(), d_gate_bias.bytes());
    d_write.copy_from_host(write.data(), d_write.bytes());
    d_next_norm.copy_from_host(next_norm.data(), d_next_norm.bytes());
    d_raw.copy_from_host(raw.data(), d_raw.bytes());
    d_anchors.copy_from_host(anchors, d_anchors.bytes());

    Tensor residual(d_raw.data(), DType::BF16, {kHidden, kBlockWidth, kBatch});
    Tensor read_weight(d_read.data(), DType::BF16, {kHidden, kCode});
    Tensor codebook_tensor(d_codebook.data(), DType::BF16, {kCode, kCodebookRows});
    Tensor gate_tensor(d_gate.data(), DType::BF16, {kCode, kRank});
    Tensor predecessor_tensor(d_predecessor.data(), DType::BF16, {kCode, kRank});
    Tensor value_tensor(d_value.data(), DType::BF16, {kCode, kRank});
    Tensor bias_tensor(d_gate_bias.data(), DType::BF16, {kRank});
    Tensor write_tensor(d_write.data(), DType::BF16, {kRank, kHidden});
    Tensor next_norm_tensor(d_next_norm.data(), DType::BF16, {kHidden});
    Tensor anchors_tensor(d_anchors.data(), DType::I32, {kBatch});
    Tensor normalized_tensor(d_normalized.data(), DType::BF16, {kHidden, columns});
    Tensor layer_input_tensor(d_layer_input.data(), DType::BF16, {kHidden, kBlockWidth, kBatch});
    Tensor projected_tensor(d_projected.data(), DType::FP32, {columns, kCode});
    Tensor codes_tensor(d_codes.data(), DType::BF16, {columns, kCode});
    Tensor rms_tensor(d_rms.data(), DType::FP32, {columns});
    Tensor packed_tensor(d_packed.data(), DType::BF16, {columns, 3 * kRank});
    Tensor message_tensor(d_message.data(), DType::BF16, {columns, kRank});
    Tensor out_tensor(d_out.data(), DType::BF16, {kHidden, kBlockWidth, kBatch});
    Tensor tail_tensor(d_tail.data(), DType::BF16, {kHidden, rows});

    ops::dspine_bridge_read(residual, read_weight, codebook_tensor, gate_tensor,
                            predecessor_tensor, value_tensor, anchors_tensor, kBlockWidth,
                            static_cast<float>(sqrt_code), static_cast<float>(eps),
                            normalized_tensor, projected_tensor, codes_tensor, rms_tensor,
                            packed_tensor, nullptr);
    ops::dspine_bridge_write(residual, rms_tensor, packed_tensor, bias_tensor, write_tensor,
                             next_norm_tensor, kBlockWidth, static_cast<float>(eps),
                             message_tensor, out_tensor, layer_input_tensor, tail_tensor, nullptr);
    cuda_synchronize();

    const auto device_codes  = from_device<std::uint16_t>(d_codes.data(), codes.size());
    const auto device_rms    = from_device<float>(d_rms.data(), rms.size());
    const auto device_packed = from_device<std::uint16_t>(d_packed.data(), packed.size());
    const auto device_out    = from_device<std::uint16_t>(d_out.data(), out.size());
    const auto device_tail   = from_device<std::uint16_t>(d_tail.data(), packed_tail.size());

    // Oracle: the documented formula, evaluated in double with the eight BF16 roundings.
    int failures = 0;
    for (std::int32_t b = 0; b < kBatch; ++b) {
        for (std::int32_t t = 0; t < kBlockWidth; ++t) {
            const std::int32_t column = t + kBlockWidth * b;
            double squares            = 0.0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const double f = bf16_at(raw, hidden_index(d, column));
                squares += f * f;
            }
            const double rms_value =
                std::sqrt(std::max(squares / kHidden, 1.1754943508222875e-38));
            const double rms_error = std::abs(rms_value - static_cast<double>(device_rms[column]));
            if (rms_error > 1.0e-5 * std::max(1.0, rms_value)) {
                std::cerr << "dspine bridge: rms[" << column << "] error " << rms_error << '\n';
                ++failures;
            }
            std::vector<double> xnorm(kHidden);
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const double f = bf16_at(raw, hidden_index(d, column));
                xnorm[d]       = round_bf16(f / (rms_value + eps));
            }
            // codes
            std::vector<double> projected(kCode, 0.0);
            double norm_squared = 0.0;
            for (std::int32_t c = 0; c < kCode; ++c) {
                double sum = 0.0;
                for (std::int32_t d = 0; d < kHidden; ++d) {
                    sum += xnorm[d] * bf16_at(read, wide_index(c, d, kHidden));
                }
                projected[c] = sum;
                norm_squared += sum * sum;
            }
            const double norm = std::sqrt(norm_squared);
            for (std::int32_t c = 0; c < kCode; ++c) {
                const double expected =
                    t == 0 ? round_bf16(sqrt_code *
                                        bf16_at(codebook, wide_index(anchors[b], c, kCode)))
                           : round_bf16(sqrt_code * projected[c] / (norm + eps));
                const double got = bf16_at(device_codes, wide_index(c, column, columns));
                if (!(std::abs(got - expected) <=
                      kStorageAbsolute + kStorageRelative * std::abs(expected))) {
                    std::cerr << "dspine bridge: codes[" << column << "][" << c << "] " << got
                              << " vs " << expected << '\n';
                    ++failures;
                    break;
                }
            }
            // packed
            for (std::int32_t r = 0; r < 3 * kRank; ++r) {
                double sum = 0.0;
                for (std::int32_t c = 0; c < kCode; ++c) {
                    const double code =
                        t == 0 ? round_bf16(sqrt_code *
                                            bf16_at(codebook, wide_index(anchors[b], c, kCode)))
                               : round_bf16(sqrt_code * projected[c] / (norm + eps));
                    const std::vector<std::uint16_t>& source =
                        r < kRank ? gate : (r < 2 * kRank ? predecessor : value);
                    const std::int32_t source_row =
                        r < kRank ? r : (r < 2 * kRank ? r - kRank : r - 2 * kRank);
                    sum += code * bf16_at(source, wide_index(source_row, c, kCode));
                }
                const double expected = round_bf16(sum);
                const double got      = bf16_at(device_packed, wide_index(r, column, columns));
                if (!(std::abs(got - expected) <=
                      kStorageAbsolute + kStorageRelative * std::abs(expected))) {
                    std::cerr << "dspine bridge: packed[" << column << "][" << r << "] " << got
                              << " vs " << expected << '\n';
                    ++failures;
                    break;
                }
            }
            // write
            std::vector<double> message(kRank, 0.0);
            if (t > 0) {
                const std::int32_t previous = column - 1;
                for (std::int32_t r = 0; r < kRank; ++r) {
                    const double query = bf16_at(device_packed, wide_index(r, column, columns));
                    const double shift =
                        bf16_at(device_packed, wide_index(kRank + r, previous, columns));
                    const double value_element =
                        bf16_at(device_packed, wide_index(2 * kRank + r, previous, columns));
                    const double gate_value =
                        1.0 / (1.0 + std::exp(-(query + shift +
                                               bf16_at(gate_bias, static_cast<std::size_t>(r)))));
                    message[r] = round_bf16(gate_value * value_element);
                }
            }
            std::vector<double> delta(kHidden, 0.0);
            for (std::int32_t d = 0; d < kHidden; ++d) {
                double sum = 0.0;
                for (std::int32_t r = 0; r < kRank; ++r) {
                    sum += message[r] * bf16_at(write, wide_index(d, r, kRank));
                }
                delta[d] = sum;
            }
            std::vector<double> out_value(kHidden, 0.0);
            int column_failures = 0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const double f = bf16_at(raw, hidden_index(d, column));
                const double expected =
                    round_bf16(t == 0 ? f : f + rms_value * delta[d]);
                const double got = bf16_at(device_out, hidden_index(d, column));
                // `out` is a sum whose terms cancel, so the stored value cannot be more accurate
                // than one BF16 ULP of the residual stream scale.
                const double limit = kStorageAbsolute + kStorageRelative * std::abs(expected) +
                                     2.0e-3 * rms_value;
                if (std::abs(got - expected) > limit && column_failures == 0) {
                    std::cerr << "dspine bridge: out[" << column << "][" << d << "] " << got
                              << " vs " << expected << '\n';
                    ++column_failures;
                }
                out_value[d] = got;
            }
            failures += column_failures;
            if (t > 0) {
                double squares_out = 0.0;
                for (std::int32_t d = 0; d < kHidden; ++d) {
                    squares_out += out_value[d] * out_value[d];
                }
                const double inverse = 1.0 / std::sqrt(squares_out / kHidden + eps);
                const std::int32_t tail_column = (kBlockWidth - 1) * b + t - 1;
                for (std::int32_t d = 0; d < kHidden; ++d) {
                    const double y      = round_bf16(out_value[d] * inverse);
                    const double expect =
                        round_bf16(y * bf16_at(next_norm, static_cast<std::size_t>(d)));
                    const double got = bf16_at(device_tail, hidden_index(d, tail_column));
                    if (!(std::abs(got - expect) <=
                          kStorageAbsolute + kStorageRelative * std::abs(expect))) {
                        std::cerr << "dspine bridge: next[" << column << "][" << d << "] " << got
                                  << " vs " << expect << '\n';
                        ++failures;
                        break;
                    }
                }
            }
        }
    }
    failures += d_codes.verify_guards("dspine bridge codes");
    failures += d_out.verify_guards("dspine bridge out");
    return failures;
}

int chain_conformance() {
    const std::int32_t width  = 5;
    const std::int32_t per_row = width - 1;
    const std::int32_t batch   = 2;
    const std::int32_t rows    = per_row * batch;
    std::vector<std::int32_t> candidates(static_cast<std::size_t>(rows) * kCandidates);
    std::vector<float> table(static_cast<std::size_t>(rows) * kCandidates * kCandidates);
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
            candidates[wide_index(row, candidate, kCandidates)] =
                100 + row * 37 + candidate * 11;
        }
        for (std::int32_t sender = 0; sender < kCandidates; ++sender) {
            for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                const std::int32_t slot = sender * kCandidates + candidate;
                table[wide_index(row, slot, kCandidates * kCandidates)] =
                    pattern(static_cast<std::uint32_t>(row * 97 + sender),
                            static_cast<std::uint32_t>(candidate), 127U, 4.0F);
            }
        }
    }
    GuardedDeviceBuffer d_table(table.size() * 4), d_candidates(candidates.size() * 4),
        d_positions(batch * 4), d_drafts(rows * 4),
        d_q(static_cast<std::size_t>(rows) * kCandidates * 4);
    d_table.copy_from_host(table.data(), d_table.bytes());
    d_candidates.copy_from_host(candidates.data(), d_candidates.bytes());
    const std::int32_t positions[batch] = {64, 512};
    d_positions.copy_from_host(positions, d_positions.bytes());

    Tensor table_tensor(d_table.data(), DType::FP32, {kCandidates * kCandidates, rows});
    Tensor candidate_tensor(d_candidates.data(), DType::I32, {kCandidates, rows});
    Tensor position_tensor(d_positions.data(), DType::I32, {batch});
    Tensor draft_tensor(d_drafts.data(), DType::I32, {per_row, batch});
    Tensor q_tensor(d_q.data(), DType::FP32, {kCandidates, per_row, batch});
    std::array<ops::SamplingConfig, batch> configs{};
    configs[0].temperature = 0.0F;
    configs[1].temperature = 0.8F;
    configs[1].top_k       = 8;
    configs[1].top_p       = 0.9F;
    configs[1].min_p       = 0.05F;
    configs[1].seed        = 12345;
    GuardedDeviceBuffer d_configs(sizeof(configs));
    d_configs.copy_from_host(configs.data(), d_configs.bytes());

    ops::dspine_refine_chain(table_tensor, candidate_tensor,
                             static_cast<const ops::SamplingConfig*>(d_configs.data()),
                             position_tensor, width, draft_tensor, q_tensor, nullptr);
    cuda_synchronize();
    const auto device_drafts =
        from_device<std::int32_t>(d_drafts.data(), static_cast<std::size_t>(rows));
    const auto device_q =
        from_device<float>(d_q.data(), static_cast<std::size_t>(rows) * kCandidates);

    int failures = 0;
    // Greedy lane: host chain walk with the documented tie-break.
    std::int32_t sender = 0;
    for (std::int32_t row = 0; row < per_row; ++row) {
        const std::int32_t token_row = row; // lane 0 -> index = row + per_row * 0
        const float* scores = table.data() + wide_index(token_row, sender * kCandidates,
                                                        kCandidates * kCandidates);
        std::int32_t best = 0;
        for (std::int32_t candidate = 1; candidate < kCandidates; ++candidate) {
            const std::int32_t best_token =
                candidates[wide_index(token_row, best, kCandidates)];
            const std::int32_t token = candidates[wide_index(token_row, candidate, kCandidates)];
            if (scores[candidate] > scores[best] ||
                (scores[candidate] == scores[best] && token < best_token)) {
                best = candidate;
            }
        }
        const std::int32_t expected = candidates[wide_index(token_row, best, kCandidates)];
        const std::int32_t got      = device_drafts[token_row];
        if (expected != got) {
            std::cerr << "dspine chain: greedy draft row " << row << ' ' << got << " vs "
                      << expected << '\n';
            ++failures;
        }
        for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
            const double q = device_q[wide_index(token_row, candidate, kCandidates)];
            const double expected_q = candidate == best ? 1.0 : 0.0;
            if (std::abs(q - expected_q) > 1e-6) {
                std::cerr << "dspine chain: greedy q mismatch\n";
                ++failures;
                break;
            }
        }
        sender = best;
    }
    // Sampled lane: the distribution is a normalized filtered softmax and the token is in support.
    sender = 0;
    for (std::int32_t row = 0; row < per_row; ++row) {
        const std::int32_t token_row = per_row + row;
        const float* scores = table.data() + wide_index(token_row, sender * kCandidates,
                                                        kCandidates * kCandidates);
        double total = 0.0;
        std::int32_t best = 0;
        for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
            total += device_q[wide_index(token_row, candidate, kCandidates)];
            if (scores[candidate] > scores[best]) { best = candidate; }
        }
        if (std::abs(total - 1.0) > 1e-5) {
            std::cerr << "dspine chain: sampled q is not normalized (" << total << ")\n";
            ++failures;
        }
        const std::int32_t got = device_drafts[token_row];
        bool in_support        = false;
        for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
            if (device_q[wide_index(token_row, candidate, kCandidates)] > 0.0 &&
                candidates[wide_index(token_row, candidate, kCandidates)] == got) {
                in_support = true;
                sender     = candidate;
            }
        }
        if (!in_support) {
            std::cerr << "dspine chain: sampled draft is outside the sampled support\n";
            ++failures;
        }
    }
    failures += d_drafts.verify_guards("dspine chain drafts");
    failures += d_q.verify_guards("dspine chain q");
    return failures;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = bridge_conformance() + chain_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " DSpine bridge and chain\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "DSpine bridge and chain: " << error.what() << '\n';
        return 1;
    }
}