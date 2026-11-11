#include "ops/dspine/dspine_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int H  = kDSpineHidden;
constexpr int C  = kDSpineCode;
constexpr int R  = kDSpineRank;
constexpr int KB = kDSpineCandidates;
constexpr int kTile = 16;

__device__ __forceinline__ float bf2f(__nv_bfloat16 value) { return __bfloat162float(value); }

__device__ __forceinline__ __nv_bfloat16 f2bf16(float value) { return __float2bfloat16(value); }

// Plain BP16 multiply-add over a strided vector, used by the scale/norm helpers.
__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
    const int tid = static_cast<int>(threadIdx.x);
    scratch[tid]   = value;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { scratch[tid] += scratch[tid + stride]; }
        __syncthreads();
    }
    return scratch[0];
}

// ---- bridge read -------------------------------------------------------------------------

__global__ __launch_bounds__(256) void read_normalize_kernel(const __nv_bfloat16* __restrict__ x,
                                                             float* __restrict__ rms,
                                                             __nv_bfloat16* __restrict__ q,
                                                             float eps) {
    __shared__ float scratch[256];
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(column) * H;
    float squares           = 0.0F;
    for (int d = tid; d < H; d += 256) {
        const float value = bf2f(x[base + d]);
        squares           = fmaf(value, value, squares);
    }
    squares = block_reduce_sum(squares, scratch);
    const float rms_value =
        sqrtf(fmaxf(squares / static_cast<float>(H), 1.1754943508222875e-38F));
    if (tid == 0) { rms[column] = rms_value; }
    const float inverse = 1.0F / (rms_value + eps);
    for (int d = tid; d < H; d += 256) { q[base + d] = f2bf16(bf2f(x[base + d]) * inverse); }
}

__global__ __launch_bounds__(256) void read_project_kernel(const __nv_bfloat16* __restrict__ q,
                                                           const __nv_bfloat16* __restrict__ read,
                                                           float* __restrict__ projected,
                                                           int columns) {
    __shared__ __nv_bfloat16 weight[kTile][512];
    __shared__ __nv_bfloat16 input[512][kTile];
    const int code0 = static_cast<int>(blockIdx.x) * kTile;
    const int col0  = static_cast<int>(blockIdx.y) * kTile;
    const int tid   = static_cast<int>(threadIdx.x);
    const int code_local = tid >> 4;
    const int col_local  = tid & 15;
    const int code       = code0 + code_local;
    const int column     = col0 + col_local;
    const bool live      = column < columns;
    float accumulator    = 0.0F;
    for (int k0 = 0; k0 < H; k0 += 512) {
        for (int item = tid; item < kTile * 512; item += 256) {
            const int row = item >> 9;
            const int at  = item & 511;
            weight[row][at] = read[static_cast<std::int64_t>(code0 + row) * H + k0 + at];
        }
        for (int item = tid; item < 512 * kTile; item += 256) {
            const int at      = item >> 4;
            const int local   = item & 15;
            const int source  = col0 + local < columns ? col0 + local : 0;
            input[at][local] = q[static_cast<std::int64_t>(source) * H + k0 + at];
        }
        __syncthreads();
#pragma unroll 8
        for (int at = 0; at < 512; ++at) {
            accumulator = fmaf(bf2f(input[at][col_local]), bf2f(weight[code_local][at]),
                               accumulator);
        }
        __syncthreads();
    }
    if (live) { projected[static_cast<std::int64_t>(code) * columns + column] = accumulator; }
}

__global__ __launch_bounds__(256) void read_codes_kernel(const float* __restrict__ projected,
                                                         const __nv_bfloat16* __restrict__ codebook,
                                                         const std::int32_t* __restrict__ anchors,
                                                         __nv_bfloat16* __restrict__ codes,
                                                         int columns, int block_width,
                                                         float sqrt_code, float eps) {
    __shared__ float scratch[256];
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int t      = column % block_width;
    const int b      = column / block_width;
    if (t == 0) {
        const int token = anchors[b];
        for (int c = tid; c < C; c += 256) {
            codes[static_cast<std::int64_t>(c) * columns + column] =
                f2bf16(sqrt_code * bf2f(codebook[static_cast<std::int64_t>(token) * C + c]));
        }
        return;
    }
    float squares = 0.0F;
    for (int c = tid; c < C; c += 256) {
        const float value = projected[static_cast<std::int64_t>(c) * columns + column];
        squares           = fmaf(value, value, squares);
    }
    squares = block_reduce_sum(squares, scratch);
    const float scale = sqrt_code / (sqrtf(squares) + eps);
    for (int c = tid; c < C; c += 256) {
        codes[static_cast<std::int64_t>(c) * columns + column] =
            f2bf16(scale * projected[static_cast<std::int64_t>(c) * columns + column]);
    }
}

__device__ __forceinline__ const __nv_bfloat16* packed_source(const __nv_bfloat16* gate,
                                                             const __nv_bfloat16* predecessor,
                                                             const __nv_bfloat16* value, int row,
                                                             int& source_row) {
    if (row < R) {
        source_row = row;
        return gate;
    }
    if (row < 2 * R) {
        source_row = row - R;
        return predecessor;
    }
    source_row = row - 2 * R;
    return value;
}

__global__ __launch_bounds__(256) void
read_packed_kernel(const __nv_bfloat16* __restrict__ codes,
                   const __nv_bfloat16* __restrict__ gate,
                   const __nv_bfloat16* __restrict__ predecessor,
                   const __nv_bfloat16* __restrict__ value,
                   __nv_bfloat16* __restrict__ packed, int columns) {
    __shared__ __nv_bfloat16 weight[kTile][512];
    __shared__ __nv_bfloat16 input[512][kTile];
    const int row0 = static_cast<int>(blockIdx.x) * kTile;
    const int col0 = static_cast<int>(blockIdx.y) * kTile;
    const int tid  = static_cast<int>(threadIdx.x);
    const int row_local = tid >> 4;
    const int col_local = tid & 15;
    const int row       = row0 + row_local;
    const int column    = col0 + col_local;
    const bool live     = column < columns;
    float accumulator   = 0.0F;
    for (int k0 = 0; k0 < C; k0 += 512) {
        for (int item = tid; item < kTile * 512; item += 256) {
            const int local = item >> 9;
            const int at    = item & 511;
            int source_row  = 0;
            const __nv_bfloat16* source =
                packed_source(gate, predecessor, value, row0 + local, source_row);
            weight[local][at] = source[static_cast<std::int64_t>(source_row) * C + k0 + at];
        }
        for (int item = tid; item < 512 * kTile; item += 256) {
            const int at     = item >> 4;
            const int local  = item & 15;
            const int source = col0 + local < columns ? col0 + local : 0;
            input[at][local] = codes[static_cast<std::int64_t>(k0 + at) * columns + source];
        }
        __syncthreads();
#pragma unroll 8
        for (int at = 0; at < 512; ++at) {
            accumulator = fmaf(bf2f(input[at][col_local]), bf2f(weight[row_local][at]),
                               accumulator);
        }
        __syncthreads();
    }
    if (live) { packed[static_cast<std::int64_t>(row) * columns + column] = f2bf16(accumulator); }
}

// ---- bridge write ------------------------------------------------------------------------

__global__ __launch_bounds__(256) void
write_message_kernel(const __nv_bfloat16* __restrict__ packed,
                     const __nv_bfloat16* __restrict__ gate_bias,
                     __nv_bfloat16* __restrict__ message, int columns, int block_width) {
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int t      = column % block_width;
    if (t == 0) {
        for (int r = tid; r < R; r += 256) {
            message[static_cast<std::int64_t>(r) * columns + column] = f2bf16(0.0F);
        }
        return;
    }
    const int previous = column - 1;
    for (int r = tid; r < R; r += 256) {
        const float query  = bf2f(packed[static_cast<std::int64_t>(r) * columns + column]);
        const float shift  = bf2f(packed[static_cast<std::int64_t>(R + r) * columns + previous]);
        const float value =
            bf2f(packed[static_cast<std::int64_t>(2 * R + r) * columns + previous]);
        const float gate   = 1.0F / (1.0F + __expf(-(query + shift + bf2f(gate_bias[r]))));
        message[static_cast<std::int64_t>(r) * columns + column] = f2bf16(gate * value);
    }
}

__global__ __launch_bounds__(256) void write_out_kernel(const __nv_bfloat16* __restrict__ residual,
                                                        const float* __restrict__ rms,
                                                        const __nv_bfloat16* __restrict__ message,
                                                        const __nv_bfloat16* __restrict__ write,
                                                        __nv_bfloat16* __restrict__ out,
                                                        int columns, int block_width) {
    __shared__ __nv_bfloat16 weight[kTile][512];
    __shared__ __nv_bfloat16 input[512][kTile];
    const int row0 = static_cast<int>(blockIdx.x) * kTile;
    const int col0 = static_cast<int>(blockIdx.y) * kTile;
    const int tid  = static_cast<int>(threadIdx.x);
    const int row_local = tid >> 4;
    const int col_local = tid & 15;
    const int row       = row0 + row_local;
    const int column    = col0 + col_local;
    const bool live     = column < columns;
    float accumulator   = 0.0F;
    for (int item = tid; item < kTile * R; item += 256) {
        const int local = item >> 9;
        const int at    = item & 511;
        weight[local][at] = write[static_cast<std::int64_t>(row0 + local) * R + at];
    }
    for (int item = tid; item < R * kTile; item += 256) {
        const int at     = item >> 4;
        const int local  = item & 15;
        const int source = col0 + local < columns ? col0 + local : 0;
        input[at][local] = message[static_cast<std::int64_t>(at) * columns + source];
    }
    __syncthreads();
#pragma unroll 8
    for (int at = 0; at < R; ++at) {
        accumulator = fmaf(bf2f(input[at][col_local]), bf2f(weight[row_local][at]), accumulator);
    }
    if (live) {
        const int t        = column % block_width;
        const float base   = bf2f(residual[static_cast<std::int64_t>(column) * H + row]);
        const float delta  = t == 0 ? 0.0F : rms[column] * accumulator;
        out[static_cast<std::int64_t>(column) * H + row] = f2bf16(base + delta);
    }
}

__global__ __launch_bounds__(256) void
write_norm_kernel(const __nv_bfloat16* __restrict__ out,
                  const __nv_bfloat16* __restrict__ next_norm,
                  __nv_bfloat16* __restrict__ normalized,
                  __nv_bfloat16* __restrict__ packed_tail, int columns, int block_width,
                  int batch, float eps) {
    __shared__ float scratch[256];
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int t      = column % block_width;
    const int b      = column / block_width;
    const std::int64_t base = static_cast<std::int64_t>(column) * H;
    float squares           = 0.0F;
    for (int d = tid; d < H; d += 256) {
        const float value = bf2f(out[base + d]);
        squares           = fmaf(value, value, squares);
    }
    squares = block_reduce_sum(squares, scratch);
    const float inverse = rsqrtf(squares / static_cast<float>(H) + eps);
    const std::int64_t tail_column =
        t == 0 ? 0 : static_cast<std::int64_t>(block_width - 1) * b + t - 1;
    const bool has_tail = packed_tail != nullptr && t > 0;
    (void)batch;
    for (int d = tid; d < H; d += 256) {
        const __nv_bfloat16 y = f2bf16(bf2f(out[base + d]) * inverse);
        const __nv_bfloat16 next = f2bf16(bf2f(y) * bf2f(next_norm[d]));
        normalized[base + d] = next;
        if (has_tail) { packed_tail[static_cast<std::int64_t>(tail_column) * H + d] = next; }
    }
}

// ---- refinement --------------------------------------------------------------------------

struct HeadView {
    const std::uint8_t* codes;
    const __half* scales;
    const __nv_bfloat16* values;
    std::int32_t qtype;
};

__device__ __forceinline__ float head_element(const HeadView& head, int row, int d) {
    if (head.qtype == static_cast<std::int32_t>(QType::BF16)) {
        return bf2f(head.values[static_cast<std::int64_t>(row) * H + d]);
    }
    const float scale =
        __half2float(head.scales[static_cast<std::int64_t>(row) * (H / 32) + (d >> 5)]);
    return static_cast<float>(static_cast<std::int8_t>(
               head.codes[static_cast<std::int64_t>(row) * H + d])) *
           scale;
}

__global__ __launch_bounds__(256) void
sender_codes_kernel(const __nv_bfloat16* __restrict__ codebook,
                    const std::int32_t* __restrict__ candidates,
                    const std::int32_t* __restrict__ anchors,
                    __nv_bfloat16* __restrict__ sender_codes, int block_width,
                    float sqrt_code) {
    const int candidates_per_row = block_width - 1;
    // One sender row per (block position, candidate rank): the anchor plus every proposal row's
    // sixteen candidates. The rank width is the candidate count, not the row count.
    const int sender_rows = 1 + (candidates_per_row - 1) * kDSpineCandidates;
    const int index       = static_cast<int>(blockIdx.x);
    const int tid         = static_cast<int>(threadIdx.x);
    const int b           = index / sender_rows;
    const int local       = index - b * sender_rows;
    int token             = anchors[b];
    if (local != 0) {
        const int row  = (local - 1) / kDSpineCandidates + 1;
        const int rank = (local - 1) % kDSpineCandidates;
        token          = candidates[rank + KB * ((row - 1) + candidates_per_row * b)];
    }
    for (int c = tid; c < C; c += 256) {
        sender_codes[static_cast<std::int64_t>(index) * C + c] =
            f2bf16(sqrt_code * bf2f(codebook[static_cast<std::int64_t>(token) * C + c]));
    }
}

__global__ __launch_bounds__(128) void
candidate_rows_kernel(const __nv_bfloat16* __restrict__ raw,
                      const __nv_bfloat16* __restrict__ norm,
                      const std::int32_t* __restrict__ candidates, HeadView head,
                      int block_width, float* __restrict__ base,
                      __nv_bfloat16* __restrict__ rows) {
    __shared__ float scratch[128];
    const int candidates_per_row = block_width - 1;
    const int column_index       = static_cast<int>(blockIdx.x); // K*b + i
    const int candidate          = static_cast<int>(blockIdx.y);
    const int tid                = static_cast<int>(threadIdx.x);
    const int b                  = column_index / candidates_per_row;
    const int row                = column_index - b * candidates_per_row;
    const int token              = candidates[candidate + KB * column_index];
    const std::int64_t raw_column =
        static_cast<std::int64_t>(row + 1) + static_cast<std::int64_t>(block_width) * b;
    const std::int64_t raw_base = raw_column * H;
    const std::int64_t row_base = static_cast<std::int64_t>(column_index * KB + candidate) * H;
    float sum                   = 0.0F;
    for (int d = tid; d < H; d += 128) {
        const float weight = head_element(head, token, d);
        const float gain   = weight * bf2f(norm[d]);
        rows[row_base + d] = f2bf16(gain);
        sum                = fmaf(gain, bf2f(raw[raw_base + d]), sum);
    }
    sum = block_reduce_sum(sum, scratch);
    if (tid == 0) { base[candidate + KB * column_index] = sum; }
}

// [rows, C] x [C, 2R] with the operand matrices stored row-major as [2R, C].
template <int MTile, int NTile, int KTile>
__global__ __launch_bounds__(256) void
sender_tsv_kernel(const __nv_bfloat16* __restrict__ codes,
                  const __nv_bfloat16* __restrict__ predecessor,
                  const __nv_bfloat16* __restrict__ value, __nv_bfloat16* __restrict__ tsv,
                  int rows) {
    static_assert(MTile * NTile % 256 == 0);
    constexpr int kPerThread = MTile * NTile / 256;
    __shared__ __nv_bfloat16 a_tile[MTile][KTile];
    __shared__ __nv_bfloat16 b_tile[KTile][NTile];
    const int m0  = static_cast<int>(blockIdx.x) * MTile;
    const int n0  = static_cast<int>(blockIdx.y) * NTile;
    const int tid = static_cast<int>(threadIdx.x);
    const int m_local = tid / (NTile / kPerThread);
    const int n_base  = (tid % (NTile / kPerThread)) * kPerThread;
    float accumulator[kPerThread];
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) { accumulator[i] = 0.0F; }
    for (int k0 = 0; k0 < C; k0 += KTile) {
        for (int item = tid; item < MTile * KTile; item += 256) {
            const int row = item / KTile;
            const int at  = item - row * KTile;
            const int m   = m0 + row;
            a_tile[row][at] = m < rows ? codes[static_cast<std::int64_t>(m) * C + k0 + at]
                                       : f2bf16(0.0F);
        }
        for (int item = tid; item < KTile * NTile; item += 256) {
            const int at  = item / NTile;
            const int col = item - at * NTile;
            const int n   = n0 + col;
            if (n < 2 * R) {
                const __nv_bfloat16* source = n < R ? predecessor : value;
                const int source_row        = n < R ? n : n - R;
                b_tile[at][col] = source[static_cast<std::int64_t>(source_row) * C + k0 + at];
            } else {
                b_tile[at][col] = f2bf16(0.0F);
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int at = 0; at < KTile; ++at) {
            const float a = bf2f(a_tile[m_local][at]);
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) {
                accumulator[i] = fmaf(a, bf2f(b_tile[at][n_base + i]), accumulator[i]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < kPerThread; ++i) {
        const int m = m0 + m_local;
        const int n = n0 + n_base + i;
        if (m < rows && n < 2 * R) {
            tsv[static_cast<std::int64_t>(m) * (2 * R) + n] = f2bf16(accumulator[i]);
        }
    }
}

// [rows, H] x [H, R] with the write matrix stored row-major as [H, R].
template <int MTile, int NTile, int KTile>
__global__ __launch_bounds__(256) void candidate_tg_kernel(const __nv_bfloat16* __restrict__ rows,
                                                           const __nv_bfloat16* __restrict__ write,
                                                           __nv_bfloat16* __restrict__ tg,
                                                           int row_count) {
    constexpr int kPerThread = MTile * NTile / 256;
    __shared__ __nv_bfloat16 a_tile[MTile][KTile];
    __shared__ __nv_bfloat16 b_tile[KTile][NTile];
    const int m0  = static_cast<int>(blockIdx.x) * MTile;
    const int n0  = static_cast<int>(blockIdx.y) * NTile;
    const int tid = static_cast<int>(threadIdx.x);
    const int m_local = tid / (NTile / kPerThread);
    const int n_base  = (tid % (NTile / kPerThread)) * kPerThread;
    float accumulator[kPerThread];
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) { accumulator[i] = 0.0F; }
    for (int k0 = 0; k0 < H; k0 += KTile) {
        for (int item = tid; item < MTile * KTile; item += 256) {
            const int row = item / KTile;
            const int at  = item - row * KTile;
            const int m   = m0 + row;
            a_tile[row][at] =
                m < row_count ? rows[static_cast<std::int64_t>(m) * H + k0 + at] : f2bf16(0.0F);
        }
        for (int item = tid; item < KTile * NTile; item += 256) {
            const int at  = item / NTile;
            const int col = item - at * NTile;
            b_tile[at][col] = write[static_cast<std::int64_t>(k0 + at) * R + n0 + col];
        }
        __syncthreads();
#pragma unroll 4
        for (int at = 0; at < KTile; ++at) {
            const float a = bf2f(a_tile[m_local][at]);
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) {
                accumulator[i] = fmaf(a, bf2f(b_tile[at][n_base + i]), accumulator[i]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < kPerThread; ++i) {
        const int m = m0 + m_local;
        const int n = n0 + n_base + i;
        if (m < row_count) { tg[static_cast<std::int64_t>(m) * R + n] = f2bf16(accumulator[i]); }
    }
}

__global__ __launch_bounds__(256) void table_kernel(const float* __restrict__ base,
                                                    const float* __restrict__ rms,
                                                    const __nv_bfloat16* __restrict__ packed,
                                                    const __nv_bfloat16* __restrict__ gate_bias,
                                                    const __nv_bfloat16* __restrict__ tsv,
                                                    const __nv_bfloat16* __restrict__ tg,
                                                    float* __restrict__ table, int block_width,
                                                    int batch) {
    __shared__ float message[kDSpineCandidates][R];
    const int candidates_per_row = block_width - 1;
    const int sender_rows        = 1 + (candidates_per_row - 1) * kDSpineCandidates;
    const int index              = static_cast<int>(blockIdx.x); // K*b + i
    const int tid                = static_cast<int>(threadIdx.x);
    const int b                  = index / candidates_per_row;
    const int row                = index - b * candidates_per_row;
    const int columns            = block_width * batch;
    const std::int64_t column =
        static_cast<std::int64_t>(row + 1) + static_cast<std::int64_t>(block_width) * b;
    const float rms_value = rms[column];
    for (int item = tid; item < kDSpineCandidates * R; item += 256) {
        const int sender     = item / R;
        const int r          = item - sender * R;
        const int sender_row = row == 0
                                   ? b * sender_rows
                                   : b * sender_rows + 1 +
                                         (row - 1) * kDSpineCandidates + sender;
        const float query = bf2f(packed[static_cast<std::int64_t>(r) * columns + column]);
        const float shift = bf2f(tsv[static_cast<std::int64_t>(sender_row) * (2 * R) + r]);
        const float value = bf2f(tsv[static_cast<std::int64_t>(sender_row) * (2 * R) + R + r]);
        const float gate  = 1.0F / (1.0F + __expf(-(query + shift + bf2f(gate_bias[r]))));
        message[sender][r] = gate * value;
    }
    __syncthreads();
    if (tid < kDSpineCandidates * kDSpineCandidates) {
        const int sender    = tid / kDSpineCandidates;
        const int candidate = tid - sender * kDSpineCandidates;
        const std::int64_t tg_base =
            static_cast<std::int64_t>(index * kDSpineCandidates + candidate) * R;
        float sum = 0.0F;
        for (int r = 0; r < R; ++r) {
            sum = fmaf(message[sender][r], bf2f(tg[tg_base + r]), sum);
        }
        const float unary = base[candidate + kDSpineCandidates * index];
        table[static_cast<std::int64_t>(index) * kDSpineCandidates * kDSpineCandidates +
              sender * kDSpineCandidates + candidate] = unary + rms_value * sum;
    }
}

__global__ __launch_bounds__(32) void chain_kernel(const float* __restrict__ table,
                                                   const std::int32_t* __restrict__ candidates,
                                                   const SamplingConfig* __restrict__ configs,
                                                   const std::int32_t* __restrict__ base_positions,
                                                   std::int32_t* __restrict__ drafts,
                                                   float* __restrict__ proposal_q,
                                                   int block_width) {
    const int candidates_per_row = block_width - 1;
    const int b                  = static_cast<int>(blockIdx.x);
    if (threadIdx.x != 0) { return; }
    const SamplingConfig config = configs[b];
    const int position          = base_positions[b];
    const bool greedy           = !(config.temperature > 0.0F);
    int sender                  = 0;
    for (int row = 0; row < candidates_per_row; ++row) {
        const std::int64_t index =
            static_cast<std::int64_t>(row) + static_cast<std::int64_t>(candidates_per_row) * b;
        const std::int32_t* row_candidates = candidates + kDSpineCandidates * index;
        const float* scores =
            table + index * kDSpineCandidates * kDSpineCandidates + sender * kDSpineCandidates;
        float* q = proposal_q + index * kDSpineCandidates;
        int chosen = 0;
        if (greedy) {
            for (int c = 1; c < kDSpineCandidates; ++c) {
                if (scores[c] > scores[chosen] ||
                    (scores[c] == scores[chosen] && row_candidates[c] < row_candidates[chosen])) {
                    chosen = c;
                }
            }
            for (int c = 0; c < kDSpineCandidates; ++c) { q[c] = c == chosen ? 1.0F : 0.0F; }
        } else {
            float probability[kDSpineCandidates];
            float maximum = -CUDART_INF_F;
            for (int c = 0; c < kDSpineCandidates; ++c) {
                probability[c] = scores[c] / config.temperature;
                maximum        = fmaxf(maximum, probability[c]);
            }
            float peak = 0.0F;
            for (int c = 0; c < kDSpineCandidates; ++c) {
                probability[c] = __expf(probability[c] - maximum);
                peak           = fmaxf(peak, probability[c]);
            }
            const int top_k = config.top_k <= 0 || config.top_k > kDSpineCandidates
                                  ? kDSpineCandidates
                                  : config.top_k;
            float weight[kDSpineCandidates];
            float total = 0.0F;
            int last    = 0;
            for (int c = 0; c < kDSpineCandidates; ++c) {
                int rank     = 0;
                float before = 0.0F;
                for (int d = 0; d < kDSpineCandidates; ++d) {
                    const bool ahead =
                        probability[d] > probability[c] ||
                        (probability[d] == probability[c] && d < c);
                    if (ahead) {
                        ++rank;
                        before += probability[d];
                    }
                }
                bool keep = rank < top_k && before <= config.top_p;
                if (config.min_p > 0.0F) { keep = keep && probability[c] >= config.min_p * peak; }
                weight[c] = keep ? probability[c] : 0.0F;
                total += weight[c];
                if (keep) { last = c; }
            }
            const float uniform =
                sampling_uniform(config.seed, position + row, kSamplePurposeDSpineProposal, 0U);
            const float threshold = uniform * total;
            float cumulative      = 0.0F;
            chosen                = 0;
            for (int c = 0; c < kDSpineCandidates; ++c) {
                cumulative += weight[c];
                if (cumulative <= threshold) { ++chosen; }
            }
            if (chosen > last) { chosen = last; }
            const float inverse = total > 0.0F ? 1.0F / total : 0.0F;
            for (int c = 0; c < kDSpineCandidates; ++c) { q[c] = weight[c] * inverse; }
        }
        drafts[index] = row_candidates[chosen];
        sender        = chosen;
    }
}

} // namespace

void dspine_bridge_read_launch(const Tensor& residual, const Tensor& read, const Tensor& codebook,
                               const Tensor& gate, const Tensor& predecessor, const Tensor& value,
                               const Tensor& anchors, std::int32_t block_width, float sqrt_code,
                               float eps, Tensor& normalized, Tensor& projected, Tensor& codes,
                               Tensor& rms, Tensor& packed, cudaStream_t stream) {
    const int columns = block_width * residual.ne[2];
    read_normalize_kernel<<<columns, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), static_cast<float*>(rms.data),
        static_cast<__nv_bfloat16*>(normalized.data), eps);
    CUDA_CHECK(cudaGetLastError());
    read_project_kernel<<<dim3(C / kTile, (columns + kTile - 1) / kTile), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(read.data), static_cast<float*>(projected.data), columns);
    CUDA_CHECK(cudaGetLastError());
    read_codes_kernel<<<columns, 256, 0, stream>>>(
        static_cast<const float*>(projected.data), static_cast<const __nv_bfloat16*>(codebook.data),
        static_cast<const std::int32_t*>(anchors.data), static_cast<__nv_bfloat16*>(codes.data),
        columns, block_width, sqrt_code, eps);
    CUDA_CHECK(cudaGetLastError());
    read_packed_kernel<<<dim3(3 * R / kTile, (columns + kTile - 1) / kTile), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(codes.data), static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const __nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(packed.data),
        columns);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_bridge_write_launch(const Tensor& residual, const Tensor& rms, const Tensor& packed,
                                const Tensor& gate_bias, const Tensor& write,
                                const Tensor& next_norm, std::int32_t block_width, float eps,
                                Tensor& message, Tensor& out, Tensor& normalized,
                                Tensor& packed_tail, cudaStream_t stream) {
    const int batch = residual.ne[2];
    const int columns = block_width * batch;
    write_message_kernel<<<columns, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed.data),
        static_cast<const __nv_bfloat16*>(gate_bias.data),
        static_cast<__nv_bfloat16*>(message.data), columns, block_width);
    CUDA_CHECK(cudaGetLastError());
    write_out_kernel<<<dim3(H / kTile, (columns + kTile - 1) / kTile), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), static_cast<const float*>(rms.data),
        static_cast<const __nv_bfloat16*>(message.data),
        static_cast<const __nv_bfloat16*>(write.data), static_cast<__nv_bfloat16*>(out.data),
        columns, block_width);
    CUDA_CHECK(cudaGetLastError());
    write_norm_kernel<<<columns, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(out.data),
        static_cast<const __nv_bfloat16*>(next_norm.data),
        static_cast<__nv_bfloat16*>(normalized.data),
        static_cast<__nv_bfloat16*>(packed_tail.data), columns, block_width, batch, eps);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_refine_score_launch(const Tensor& raw, const Tensor& rms, const Tensor& packed,
                                const Tensor& candidates, const Tensor& anchors,
                                const Tensor& codebook, const Tensor& gate_bias,
                                const Tensor& predecessor, const Tensor& value, const Tensor& write,
                                const Tensor& norm, const Weight& head,
                                std::int32_t token_domain, std::int32_t block_width,
                                float sqrt_code, float eps, DSpineRefineBuffers& buffers,
                                cudaStream_t stream) {
    const int batch                = raw.ne[2];
    const int candidates_per_row   = block_width - 1;
    const int columns              = candidates_per_row * batch; // proposal columns
    const int sender_rows_per_lane = 1 + (candidates_per_row - 1) * kDSpineCandidates;
    const HeadView head_view{static_cast<const std::uint8_t*>(head.qdata),
                             static_cast<const __half*>(head.scales),
                             static_cast<const __nv_bfloat16*>(head.payload),
                             static_cast<std::int32_t>(head.qtype)};
    (void)token_domain;
    (void)eps;
    sender_codes_kernel<<<batch * sender_rows_per_lane, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(codebook.data),
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<__nv_bfloat16*>(buffers.sender_codes.data), block_width, sqrt_code);
    CUDA_CHECK(cudaGetLastError());
    candidate_rows_kernel<<<dim3(columns, kDSpineCandidates), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw.data), static_cast<const __nv_bfloat16*>(norm.data),
        static_cast<const std::int32_t*>(candidates.data), head_view, block_width,
        static_cast<float*>(buffers.base.data),
        static_cast<__nv_bfloat16*>(buffers.candidate_rows.data));
    CUDA_CHECK(cudaGetLastError());
    sender_tsv_kernel<8, 64, 256><<<dim3((batch * sender_rows_per_lane + 7) / 8, (2 * R) / 64), 256,
                                   0, stream>>>(
        static_cast<const __nv_bfloat16*>(buffers.sender_codes.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const __nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(buffers.sender_tsv.data), batch * sender_rows_per_lane);
    CUDA_CHECK(cudaGetLastError());
    candidate_tg_kernel<16, 128, 128><<<dim3((columns * kDSpineCandidates + 15) / 16, R / 128), 256,
                                        0, stream>>>(
        static_cast<const __nv_bfloat16*>(buffers.candidate_rows.data),
        static_cast<const __nv_bfloat16*>(write.data),
        static_cast<__nv_bfloat16*>(buffers.candidate_tg.data), columns * kDSpineCandidates);
    CUDA_CHECK(cudaGetLastError());
    table_kernel<<<columns, 256, 0, stream>>>(
        static_cast<const float*>(buffers.base.data), static_cast<const float*>(rms.data),
        static_cast<const __nv_bfloat16*>(packed.data),
        static_cast<const __nv_bfloat16*>(gate_bias.data),
        static_cast<const __nv_bfloat16*>(buffers.sender_tsv.data),
        static_cast<const __nv_bfloat16*>(buffers.candidate_tg.data),
        static_cast<float*>(buffers.table.data), block_width, batch);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_refine_chain_launch(const Tensor& table, const Tensor& candidates,
                                const SamplingConfig* configs, const Tensor& base_positions,
                                std::int32_t block_width, Tensor& drafts, Tensor& proposal_q,
                                cudaStream_t stream) {
    const int batch = drafts.ne[1];
    chain_kernel<<<batch, 32, 0, stream>>>(
        static_cast<const float*>(table.data), static_cast<const std::int32_t*>(candidates.data),
        configs, static_cast<const std::int32_t*>(base_positions.data),
        static_cast<std::int32_t*>(drafts.data), static_cast<float*>(proposal_q.data), block_width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail