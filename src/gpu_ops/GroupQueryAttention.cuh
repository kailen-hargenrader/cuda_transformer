#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "../ErrorCheck.h"

namespace {

constexpr int kGqaTileLen = 64;
constexpr int kGqaTileThreads = 64;
constexpr int kGqaMergeThreads = 64;

__host__ __device__ constexpr int32_t gqa_tile_count_for_seq_len(int32_t seq_len) {
    return (seq_len + kGqaTileLen - 1) / kGqaTileLen;
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
                                      float *__restrict__ partial_weighted_values,
                                      int32_t layer_num,
                                      int32_t seq_len,
                                      int32_t max_tile_count) {
    using Config = Qwen2Config<QWEN2_SIZE>;

    const int32_t query_head = static_cast<int32_t>(blockIdx.x);
    const int32_t tile_idx = static_cast<int32_t>(blockIdx.y);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t tile_count = gqa_tile_count_for_seq_len(seq_len);
    if (query_head >= Config::num_query_heads() || tile_idx >= tile_count) {
        return;
    }

    __shared__ float query_shared[Config::head_size()];
    __shared__ float scores_shared[kGqaTileLen];
    __shared__ float tile_max;
    __shared__ float tile_denom;

    query_shared[tid] = __bfloat162float(queries[static_cast<size_t>(query_head) * Config::head_size() + tid]);
    __syncthreads();

    const int32_t seq_start = tile_idx * kGqaTileLen;
    const int32_t tile_len = min(kGqaTileLen, seq_len - seq_start);
    const int32_t kv_head = query_head * Config::num_kv_heads() / Config::num_query_heads();

    if (tid < tile_len) {
        const auto *key = gqa_key_ptr<QWEN2_SIZE>(k_cache, seq_start + tid, layer_num, kv_head);
        float dot = 0.0f;
        for (int32_t i = 0; i < Config::head_size(); ++i) {
            dot += query_shared[i] * __bfloat162float(key[i]);
        }
        scores_shared[tid] = dot * rsqrtf(static_cast<float>(Config::head_size()));
    }
    __syncthreads();

    const size_t partial_idx = gqa_partial_idx(query_head, tile_idx, max_tile_count);
    if (tid == 0) {
        float local_m = -INFINITY;
        float local_l = 0.0f;
        for (int32_t i = 0; i < tile_len; ++i) {
            const float score = scores_shared[i];
            const float next_m = fmaxf(local_m, score);
            local_l = local_l * expf(local_m - next_m) + expf(score - next_m);
            local_m = next_m;
        }
        tile_max = local_m;
        tile_denom = local_l;
        partial_maxima[partial_idx] = local_m;
        partial_denominators[partial_idx] = local_l;
    }
    __syncthreads();

    float partial_out = 0.0f;
    for (int32_t i = 0; i < tile_len; ++i) {
        const auto *value = gqa_value_ptr<QWEN2_SIZE>(v_cache, seq_start + i, layer_num, kv_head);
        partial_out += expf(scores_shared[i] - tile_max) * __bfloat162float(value[tid]);
    }
    partial_weighted_values[partial_idx * Config::value_size() + tid] = partial_out;
}

template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_merge_tile_partials_kernel(const float *__restrict__ partial_maxima,
                                               const float *__restrict__ partial_denominators,
                                               const float *__restrict__ partial_weighted_values,
                                               float *__restrict__ weighted_values,
                                               int32_t seq_len,
                                               int32_t max_tile_count) {
    using Config = Qwen2Config<QWEN2_SIZE>;

    const int32_t query_head = static_cast<int32_t>(blockIdx.x);
    const int32_t value_idx = static_cast<int32_t>(threadIdx.x);
    if (query_head >= Config::num_query_heads() || value_idx >= Config::value_size()) {
        return;
    }

    const int32_t tile_count = gqa_tile_count_for_seq_len(seq_len);
    const size_t head_base = static_cast<size_t>(query_head) * static_cast<size_t>(max_tile_count);

    float merged_m = -INFINITY;
    float merged_l = 0.0f;
    float merged_out = 0.0f;
    for (int32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const size_t partial_idx = head_base + static_cast<size_t>(tile_idx);
        const float tile_m = partial_maxima[partial_idx];
        const float tile_l = partial_denominators[partial_idx];
        if (tile_l == 0.0f) {
            continue;
        }

        const float tile_out = partial_weighted_values[partial_idx * Config::value_size() + value_idx];
        if (merged_l == 0.0f) {
            merged_m = tile_m;
            merged_l = tile_l;
            merged_out = tile_out;
            continue;
        }

        const float next_m = fmaxf(merged_m, tile_m);
        const float merged_scale = expf(merged_m - next_m);
        const float tile_scale = expf(tile_m - next_m);
        merged_out = merged_out * merged_scale + tile_out * tile_scale;
        merged_l = merged_l * merged_scale + tile_l * tile_scale;
        merged_m = next_m;
    }

    weighted_values[static_cast<size_t>(query_head) * Config::value_size() + value_idx] =
        merged_l == 0.0f ? 0.0f : merged_out / merged_l;
}

} // namespace

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    std::shared_ptr<CudaBuffer> partial_stats_buffer;
    std::shared_ptr<CudaBuffer> partial_values_buffer;
    int32_t max_seq_len;
    int32_t max_tile_count;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len): max_seq_len(max_seq_len),
                                                       max_tile_count(gqa_tile_count_for_seq_len(max_seq_len)) {
        partial_stats_buffer = std::make_shared<CudaBuffer>(
            static_cast<size_t>(Qwen2Config::num_query_heads()) * max_tile_count * 2 * sizeof(float));
        partial_values_buffer = std::make_shared<CudaBuffer>(
            static_cast<size_t>(Qwen2Config::num_query_heads()) * max_tile_count *
            Qwen2Config::value_size() * sizeof(float));
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
            partial_maxima + static_cast<size_t>(Qwen2Config::num_query_heads()) * max_tile_count;
        auto *partial_weighted_values = static_cast<float *>(partial_values_buffer->data);

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
            max_tile_count);
        checkCuda(cudaGetLastError());

        gqa_merge_tile_partials_kernel<QWEN2_SIZE><<<Qwen2Config::num_query_heads(), kGqaMergeThreads, 0, stream>>>(
            partial_maxima,
            partial_denominators,
            partial_weighted_values,
            weighted_values,
            seq_len,
            max_tile_count);
        checkCuda(cudaGetLastError());
    }
};
