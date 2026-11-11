#pragma once

#include "core/device.h" // kTargetSmCount
#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

// The small-T partial kernels run two CTAs per SM and their grid is (KVHeads, splits, batch), so
// one resident wave on the build's target device is 2 * SMs / KVHeads splits. Upstream's literal
// cap of 85 splits was one wave of the RTX 5090's 170 SMs (4 KV heads x 85 = 340 = 2 x 170); on
// the RTX 4090 it launched 340 CTAs into 256 slots, a second wave at one-third occupancy at every
// depth that reached the cap (design 9.1, 128K decode profile). The fork therefore caps at two
// waves, but two waves are not enough on a narrow card: a split can stage at most 3968 keys, so
// staging the full 262144-key context needs at least 67 splits. The cap is the larger of the
// historical two waves and the whole-wave rounding of those 67 splits:
//   RTX 4090        (128 SM, 4 KV) 2 waves = 128 splits, required 128 -> 128 (unchanged)
//   RTX 4080 SUPER  ( 80 SM, 4 KV) 2 waves =  80 splits, required  80 ->  80 (unchanged)
//   RTX 5060 Ti     ( 36 SM, 4 KV) 2 waves =  36 splits, required  72 ->  72 (4 waves)
// The host split policy (small_t.cu) and the device mirror (small_t.cuh) both use these
// definitions.
inline constexpr int kCausalSmallTCtasPerSm       = 2;
inline constexpr int kCausalSmallTMaxKeysPerSplit = 3968;
inline constexpr int kCausalSmallTMaxKeys         = 262144;
inline constexpr int kCausalSmallTRequiredSplits =
    (kCausalSmallTMaxKeys + kCausalSmallTMaxKeysPerSplit - 1) / kCausalSmallTMaxKeysPerSplit;

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTWaveSplits    = kCausalSmallTCtasPerSm * kTargetSmCount / KVHeadsValue;
    static constexpr int SmallTRequiredSplits =
        (kCausalSmallTRequiredSplits + SmallTWaveSplits - 1) / SmallTWaveSplits * SmallTWaveSplits;
    static constexpr int SmallTMaximumSplits =
        SmallTRequiredSplits > 2 * SmallTWaveSplits ? SmallTRequiredSplits : 2 * SmallTWaveSplits;
    static_assert(SmallTWaveSplits % SmallTSplitScale == 0);
    static_assert(SmallTMaximumSplits * kCausalSmallTMaxKeysPerSplit >= kCausalSmallTMaxKeys);
};

// The two asserts inside CausalAttentionGeometry constrain kTargetSmCount indirectly. Make the
// valid domain explicit so a misconfigured NINFER_TARGET_SM_COUNT fails here with a clear
// message instead of deep inside a template instantiation: even (D256H16Kv2 needs
// SmallTWaveSplits = kTargetSmCount to be divisible by its split scale of 2) and >= 32. At every
// even count in that domain the split cap derived above stages kCausalSmallTMaxKeys keys; the
// build-time check in CMakeLists.txt mirrors the same range.
static_assert(kTargetSmCount % 2 == 0 && kTargetSmCount >= 32,
              "kTargetSmCount must be even and >= 32 so both head geometries can stage "
              "kCausalSmallTMaxKeys keys");

// Rounds a split count past one wave up to whole waves, then applies the cap.
template <typename Geometry>
__host__ __device__ constexpr int causal_small_t_wave_splits(int splits) {
    constexpr int kWave = Geometry::SmallTWaveSplits;
    if (splits > kWave) { splits = (splits + kWave - 1) / kWave * kWave; }
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;

} // namespace ninfer::ops
