#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "../ErrorCheck.h"

namespace {

constexpr int kGqaWarpSize = 32;
constexpr int kGqaWarpsPerBlock = 4;
constexpr int kGqaRowsPerWarp = 8;
constexpr int kGqaTileLen = kGqaWarpsPerBlock * kGqaRowsPerWarp;
constexpr int kGqaTileThreads = kGqaWarpSize * kGqaWarpsPerBlock;
constexpr int kGqaMergeThreads = 32;

__device__ __forceinline__ float gqa_warp_sum(float value) {
    for (int offset = kGqaWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float gqa_bf162_dot(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __bfloat162float(__low2bfloat16(a)) * __bfloat162float(__low2bfloat16(b)) +
           __bfloat162float(__high2bfloat16(a)) * __bfloat162float(__high2bfloat16(b));
}

__host__ __device__ constexpr int32_t gqa_tile_count_for_seq_len(int32_t seq_len) {
    return (seq_len + kGqaTileLen - 1) / kGqaTileLen;
}

__host__ __device__ constexpr int32_t gqa_partial_count_for_seq_len(int32_t seq_len) {
    return gqa_tile_count_for_seq_len(seq_len) * kGqaWarpsPerBlock;
}

__device__ __forceinline__ size_t gqa_partial_idx(int32_t query_head, int32_t tile_idx, int32_t max_tile_count) {
    return static_cast<size_t>(query_head) * static_cast<size_t>(max_tile_count) + static_cast<size_t>(tile_idx);
}

template<Qwen2Size QWEN2_SIZE>
__device__ __forceinline__ const __nv_bfloat16 *gqa_key_ptr(const __nv_bfloat16 *k_cache, int32_t seq_pos,
                                                            int32_t layer_num, int32_t kv_head) {
    using Config = Qwen2Config<QWEN2_SIZE>;
    return k_cache +
        static_cast<size_t>(seq_pos) * (Config::num_layers() * Config::keys_size()) +
        static_cast<size_t>(layer_num) * Config::keys_size() +
        static_cast<size_t>(kv_head) * Config::head_size();
}

template<Qwen2Size QWEN2_SIZE>
__device__ __forceinline__ const __nv_bfloat16 *gqa_value_ptr(const __nv_bfloat16 *v_cache, int32_t seq_pos,
                                                              int32_t layer_num, int32_t kv_head) {
    using Config = Qwen2Config<QWEN2_SIZE>;
    return v_cache +
        static_cast<size_t>(seq_pos) * (Config::num_layers() * Config::values_size()) +
        static_cast<size_t>(layer_num) * Config::values_size() +
        static_cast<size_t>(kv_head) * Config::value_size();
}

__device__ __forceinline__ void merge_online_softmax_state(float other_m, float other_l, float &m, float &l) {
    if (other_l == 0.0f) {
        return;
    }
    if (l == 0.0f) {
        m = other_m;
        l = other_l;
        return;
    }
    const float merged_m = fmaxf(m, other_m);
    l = l * expf(m - merged_m) + other_l * expf(other_m - merged_m);
    m = merged_m;
}

template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_fused_tile_kernel(const __nv_bfloat16 *__restrict__ queries,
                                      const __nv_bfloat16 *__restrict__ k_cache,
                                      const __nv_bfloat16 *__restrict__ v_cache,
                                      float *__restrict__ partial_maxima,
                                      float *__restrict__ partial_denominators,
                                      float2 *__restrict__ partial_weighted_values,
                                      int32_t layer_num,
                                      int32_t seq_len,
                                      int32_t max_partial_count) {
    using Config = Qwen2Config<QWEN2_SIZE>;
    static_assert(Config::head_size() == 64, "warp-per-row GQA kernel expects head_size == 64");
    static_assert(Config::value_size() == 64, "warp-per-row GQA kernel expects value_size == 64");

    const int32_t query_head = static_cast<int32_t>(blockIdx.x);
    const int32_t tile_idx = static_cast<int32_t>(blockIdx.y);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t tile_count = gqa_tile_count_for_seq_len(seq_len);
    if (query_head >= Config::num_query_heads() || tile_idx >= tile_count) {
        return;
    }

    constexpr int32_t kHeadPairs = Config::head_size() / 2;
    __shared__ __nv_bfloat162 query_shared[kHeadPairs];

    if (tid < kHeadPairs) {
        const auto *query_pairs = reinterpret_cast<const __nv_bfloat162 *>(
            queries + static_cast<size_t>(query_head) * Config::head_size());
        query_shared[tid] = query_pairs[tid];
    }
    __syncthreads();

    const int32_t seq_start = tile_idx * kGqaTileLen;
    const int32_t tile_len = min(kGqaTileLen, seq_len - seq_start);
    const int32_t kv_head = query_head * Config::num_kv_heads() / Config::num_query_heads();
    const int32_t lane = tid & (kGqaWarpSize - 1);
    const int32_t warp_id = tid / kGqaWarpSize;

    float local_m = -INFINITY;
    float local_l = 0.0f;
    float2 partial_out = {0.0f, 0.0f};
    const float scale = rsqrtf(static_cast<float>(Config::head_size()));

    for (int32_t row_offset = warp_id; row_offset < tile_len; row_offset += kGqaWarpsPerBlock) {
        const int32_t seq_pos = seq_start + row_offset;

        float dot_partial = 0.0f;
        const auto *key_pairs = reinterpret_cast<const __nv_bfloat162 *>(
            gqa_key_ptr<QWEN2_SIZE>(k_cache, seq_pos, layer_num, kv_head));
        dot_partial = gqa_bf162_dot(query_shared[lane], key_pairs[lane]);
        dot_partial = gqa_warp_sum(dot_partial);

        float score = lane == 0 ? dot_partial * scale : 0.0f;
        score = __shfl_sync(0xffffffffu, score, 0);

        const float next_m = fmaxf(local_m, score);
        const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
        const float new_scale = expf(score - next_m);

        const auto *value_pairs = reinterpret_cast<const __nv_bfloat162 *>(
            gqa_value_ptr<QWEN2_SIZE>(v_cache, seq_pos, layer_num, kv_head));
        const __nv_bfloat162 value_pair = value_pairs[lane];
        partial_out.x = partial_out.x * old_scale + new_scale * __bfloat162float(__low2bfloat16(value_pair));
        partial_out.y = partial_out.y * old_scale + new_scale * __bfloat162float(__high2bfloat16(value_pair));
        local_l = local_l * old_scale + new_scale;
        local_m = next_m;
    }

    const size_t partial_idx = gqa_partial_idx(
        query_head,
        tile_idx * kGqaWarpsPerBlock + warp_id,
        max_partial_count);
    if (lane == 0) {
        const bool warp_active = warp_id < tile_len;
        partial_maxima[partial_idx] = warp_active ? local_m : 0.0f;
        partial_denominators[partial_idx] = warp_active ? local_l : 0.0f;
    }
    partial_weighted_values[partial_idx * kHeadPairs + lane] =
        warp_id < tile_len ? partial_out : float2{0.0f, 0.0f};
}

template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_merge_tile_partials_kernel(const float *__restrict__ partial_maxima,
                                               const float *__restrict__ partial_denominators,
                                               const float2 *__restrict__ partial_weighted_values,
                                               float *__restrict__ weighted_values,
                                               int32_t seq_len,
                                               int32_t max_partial_count) {
    using Config = Qwen2Config<QWEN2_SIZE>;
    static_assert(Config::value_size() == 64, "warp-per-row GQA merge kernel expects value_size == 64");
    constexpr int32_t kHeadPairs = Config::value_size() / 2;

    const int32_t query_head = static_cast<int32_t>(blockIdx.x);
    const int32_t pair_idx = static_cast<int32_t>(threadIdx.x);
    if (query_head >= Config::num_query_heads() || pair_idx >= kHeadPairs) {
        return;
    }

    const int32_t partial_count = gqa_partial_count_for_seq_len(seq_len);
    const size_t head_base = static_cast<size_t>(query_head) * static_cast<size_t>(max_partial_count);
    const int32_t lane = pair_idx & (kGqaWarpSize - 1);

    float merged_m = -INFINITY;
    float merged_l = 0.0f;
    float2 merged_out{0.0f, 0.0f};
    for (int32_t partial_slot = 0; partial_slot < partial_count; ++partial_slot) {
        const size_t partial_idx = head_base + static_cast<size_t>(partial_slot);
        float tile_m = lane == 0 ? partial_maxima[partial_idx] : 0.0f;
        float tile_l = lane == 0 ? partial_denominators[partial_idx] : 0.0f;
        tile_m = __shfl_sync(0xffffffffu, tile_m, 0);
        tile_l = __shfl_sync(0xffffffffu, tile_l, 0);
        if (tile_l == 0.0f) {
            continue;
        }

        const float2 tile_out = partial_weighted_values[partial_idx * kHeadPairs + pair_idx];
        if (merged_l == 0.0f) {
            merged_m = tile_m;
            merged_l = tile_l;
            merged_out = tile_out;
            continue;
        }

        const float next_m = fmaxf(merged_m, tile_m);
        const float merged_scale = expf(merged_m - next_m);
        const float tile_scale = expf(tile_m - next_m);
        merged_out.x = merged_out.x * merged_scale + tile_out.x * tile_scale;
        merged_out.y = merged_out.y * merged_scale + tile_out.y * tile_scale;
        merged_l = merged_l * merged_scale + tile_l * tile_scale;
        merged_m = next_m;
    }

    const size_t out_base = static_cast<size_t>(query_head) * Config::value_size() + pair_idx * 2;
    if (merged_l == 0.0f) {
        weighted_values[out_base] = 0.0f;
        weighted_values[out_base + 1] = 0.0f;
    } else {
        weighted_values[out_base] = merged_out.x / merged_l;
        weighted_values[out_base + 1] = merged_out.y / merged_l;
    }
}

} // namespace

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    std::shared_ptr<CudaBuffer> partial_stats_buffer;
    std::shared_ptr<CudaBuffer> partial_values_buffer;
    int32_t max_seq_len;
    int32_t max_partial_count;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len): max_seq_len(max_seq_len),
                                                       max_partial_count(gqa_partial_count_for_seq_len(max_seq_len)) {
        partial_stats_buffer = std::make_shared<CudaBuffer>(
            static_cast<size_t>(Qwen2Config::num_query_heads()) * max_partial_count * 2 * sizeof(float));
        partial_values_buffer = std::make_shared<CudaBuffer>(
            static_cast<size_t>(Qwen2Config::num_query_heads()) * max_partial_count *
            (Qwen2Config::value_size() / 2) * sizeof(float2));
    }

    /**
     * Scaled dot product attention with grouped queries, see https://arxiv.org/abs/2305.13245.
     * Performs softmax((QK^T)/sqrt(d_k))*V for all queries Q and their associated K and V
     * - dot product each query with its target value throughout the sequence
     * - numerically stable softmax
     * - save a weighted sum of values
     * Does not perform the output projection.
     *
     * All inputs and outputs are row-major
     *
     * @param queries (num_query_heads, head_size)
     * @param k_cache (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache (seq_len, num_layers, num_kv_heads, value_size)
     * @param weighted_values (num_query_heads, value_size) outputs
     * @param layer_num layer index, starting at 0
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
        if (seq_len <= 0) {
            throw std::runtime_error("sequence length must be positive");
        }
        if (seq_len > max_seq_len) {
            throw std::runtime_error("sequence length exceeds allocated GroupQueryAttention scratch space");
        }

        const int32_t tile_count = gqa_tile_count_for_seq_len(seq_len);
        auto *partial_maxima = static_cast<float *>(partial_stats_buffer->data);
        auto *partial_denominators =
            partial_maxima + static_cast<size_t>(Qwen2Config::num_query_heads()) * max_partial_count;
        auto *partial_weighted_values = static_cast<float2 *>(partial_values_buffer->data);

        const dim3 tile_grid(Qwen2Config::num_query_heads(), tile_count);
        gqa_fused_tile_kernel<QWEN2_SIZE><<<tile_grid, kGqaTileThreads, 0, stream>>>(
            queries,
            k_cache,
            v_cache,
            partial_maxima,
            partial_denominators,
            partial_weighted_values,
            layer_num,
            seq_len,
            max_partial_count);
        checkCuda(cudaGetLastError());

        gqa_merge_tile_partials_kernel<QWEN2_SIZE><<<Qwen2Config::num_query_heads(), kGqaMergeThreads, 0, stream>>>(
            partial_maxima,
            partial_denominators,
            partial_weighted_values,
            weighted_values,
            seq_len,
            max_partial_count);
        checkCuda(cudaGetLastError());
    }
};
