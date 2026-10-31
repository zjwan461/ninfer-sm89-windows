// Fork-local MSVC build: the NVFP4 W4A4 TMA launchers cannot be compiled because nvcc's host
// stub passes a __grid_constant__ CUtensorMap (alignas(128)) by value and MSVC rejects
// over-aligned by-value parameters (C2719). The two launcher translation units are excluded in
// src/ops/CMakeLists.txt when NINFER_BUILD_NVFP4_TMA is off; these definitions keep the
// nvfp4_w4a4 callers linkable. Nothing should reach them: prepare.cpp downgrades AllowA4
// activations to A16 under NINFER_NO_NVFP4_TMA, exactly like the sm_86/sm_89 build.
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "NVFP4 W4A4 TMA is not available in this build: MSVC cannot pass the 128-byte-aligned "
        "CUtensorMap kernel parameter (C2719). Use an A16 artifact or rebuild with "
        "-DNINFER_BUILD_NVFP4_TMA=ON and a toolchain that accepts it.");
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4GeometryId, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    unavailable();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, std::int32_t, float,
                                     cudaStream_t) {
    unavailable();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    unavailable();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4GeometryId, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    unavailable();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                         std::int32_t, float, cudaStream_t) {
    unavailable();
}

} // namespace ninfer::ops::detail