#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
// Upper bound on --prefill-chunk. This is now only a sanity guard: the cooperative residency
// budget used to be a compile-time guess that let large chunks abort with 0xC0000409, and the
// launcher capped the value to stay clear of it. The budget now comes from
// cudaOccupancyMaxActiveBlocksPerMultiprocessor, so 1792/2048/3072/4096 all run. Measured on the
// RTX 4080 SUPER, prefill is flat from 1024 up (1024 -> 3.41k tok/s, 2048 -> 3.40k,
// 4096 -> 3.39k) and 512 is worse (3.26k), so 1024 stays the recommended value.
inline constexpr std::uint32_t kPrefillChunkMaximum       = 4096;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
// Drafts one MTP round can verify: the MTP proposal plus host-supplied extensions.
inline constexpr std::uint32_t kMaximumMtpVerifyDrafts = 15;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
