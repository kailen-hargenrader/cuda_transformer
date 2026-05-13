#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "../ErrorCheck.h"

namespace {

constexpr int kGqaScoreThreads = 128;
constexpr int kGqaStatsThreads = 256;
constexpr int kGqaValueThreads = 64;

template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_score_kernel(const __nv_bfloat16 *__restrict__ queries,
                                 const __nv_bfloat16 *__restrict__ k_cache,
                                 float *__restrict__ scores,
                                 int32_t layer_num,
                                 int32_t seq_len,
                                 int32_t max_seq_len) {
    using Config = Qwen2Config<QWEN2_SIZE>;

    const int32_t query_head = static_cast<int32_t>(blockIdx.x);
    const int32_t seq_pos = static_cast<int32_t>(blockIdx.y) * kGqaScoreThreads + static_cast<int32_t>(threadIdx.x);
    if (query_head >= Config::num_query_heads() || seq_pos >= seq_len) {
        return;
    }

    const int32_t kv_head = query_head * Config::num_kv_heads() / Config::num_query_heads();
    const auto *query = queries + static_cast<size_t>(query_head) * Config::head_size();
    const auto *key = k_cache +
        static_cast<size_t>(seq_pos) * (Config::num_layers() * Config::keys_size()) +
        static_cast<size_t>(layer_num) * Config::keys_size() +
        static_cast<size_t>(kv_head) * Config::head_size();

    float dot = 0.0f;
    for (int32_t i = 0; i < Config::head_size(); ++i) {
        dot += __bfloat162float(query[i]) * __bfloat162float(key[i]);
    }

    const float scale = rsqrtf(static_cast<float>(Config::head_size()));
    scores[static_cast<size_t>(query_head) * max_seq_len + seq_pos] = dot * scale;
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
__global__ void gqa_online_softmax_stats_kernel(const float *__restrict__ scores,
                                                float *__restrict__ max_scores,
                                                float *__restrict__ denominators,
                                                int32_t seq_len,
                                                int32_t max_seq_len) {
    using Config = Qwen2Config<QWEN2_SIZE>;

    const int32_t query_head = static_cast<int32_t>(blockIdx.x);
    if (query_head >= Config::num_query_heads()) {
        return;
    }

    const auto *head_scores = scores + static_cast<size_t>(query_head) * max_seq_len;
    float local_m = -INFINITY;
    float local_l = 0.0f;

    for (int32_t seq_pos = static_cast<int32_t>(threadIdx.x); seq_pos < seq_len; seq_pos += kGqaStatsThreads) {
        const float score = head_scores[seq_pos];
        const float next_m = fmaxf(local_m, score);
        local_l = local_l * expf(local_m - next_m) + expf(score - next_m);
        local_m = next_m;
    }

    __shared__ float thread_m[kGqaStatsThreads];
    __shared__ float thread_l[kGqaStatsThreads];
    thread_m[threadIdx.x] = local_m;
    thread_l[threadIdx.x] = local_l;
    __syncthreads();

    if (threadIdx.x == 0) {
        float merged_m = -INFINITY;
        float merged_l = 0.0f;
        for (int32_t i = 0; i < kGqaStatsThreads; ++i) {
            merge_online_softmax_state(thread_m[i], thread_l[i], merged_m, merged_l);
        }
        max_scores[query_head] = merged_m;
        denominators[query_head] = merged_l;
    }
}

template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_weighted_sum_kernel(const float *__restrict__ scores,
                                        const float *__restrict__ max_scores,
                                        const float *__restrict__ denominators,
                                        const __nv_bfloat16 *__restrict__ v_cache,
                                        float *__restrict__ weighted_values,
                                        int32_t layer_num,
                                        int32_t seq_len,
                                        int32_t max_seq_len) {
    using Config = Qwen2Config<QWEN2_SIZE>;

    const int32_t query_head = static_cast<int32_t>(blockIdx.x);
    const int32_t value_idx = static_cast<int32_t>(threadIdx.x);
    if (query_head >= Config::num_query_heads() || value_idx >= Config::value_size()) {
        return;
    }

    const int32_t kv_head = query_head * Config::num_kv_heads() / Config::num_query_heads();
    const float max_score = max_scores[query_head];
    const float denominator = denominators[query_head];
    const auto *head_scores = scores + static_cast<size_t>(query_head) * max_seq_len;

    float acc = 0.0f;
    for (int32_t seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
        const auto *value = v_cache +
            static_cast<size_t>(seq_pos) * (Config::num_layers() * Config::values_size()) +
            static_cast<size_t>(layer_num) * Config::values_size() +
            static_cast<size_t>(kv_head) * Config::value_size();
        const float weight = expf(head_scores[seq_pos] - max_score) / denominator;
        acc += weight * __bfloat162float(value[value_idx]);
    }

    weighted_values[static_cast<size_t>(query_head) * Config::value_size() + value_idx] = acc;
}

} // namespace

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    std::shared_ptr<CudaBuffer> score_buffer;
    std::shared_ptr<CudaBuffer> online_softmax_buffer;
    int32_t max_seq_len;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len): max_seq_len(max_seq_len) {
        score_buffer = std::make_shared<CudaBuffer>(
            static_cast<size_t>(Qwen2Config::num_query_heads()) * max_seq_len * sizeof(float));
        online_softmax_buffer = std::make_shared<CudaBuffer>(
            static_cast<size_t>(Qwen2Config::num_query_heads()) * 2 * sizeof(float));
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
        if (seq_len > max_seq_len) {
            throw std::runtime_error("sequence length exceeds allocated GroupQueryAttention scratch space");
        }

        auto *scores = static_cast<float *>(score_buffer->data);
        auto *max_scores = static_cast<float *>(online_softmax_buffer->data);
        auto *denominators = max_scores + Qwen2Config::num_query_heads();

        const dim3 score_grid(
            Qwen2Config::num_query_heads(),
            (seq_len + kGqaScoreThreads - 1) / kGqaScoreThreads);
        gqa_score_kernel<QWEN2_SIZE><<<score_grid, kGqaScoreThreads, 0, stream>>>(
            queries, k_cache, scores, layer_num, seq_len, max_seq_len);
        checkCuda(cudaGetLastError());

        gqa_online_softmax_stats_kernel<QWEN2_SIZE><<<Qwen2Config::num_query_heads(), kGqaStatsThreads, 0, stream>>>(
            scores, max_scores, denominators, seq_len, max_seq_len);
        checkCuda(cudaGetLastError());

        gqa_weighted_sum_kernel<QWEN2_SIZE><<<Qwen2Config::num_query_heads(), kGqaValueThreads, 0, stream>>>(
            scores, max_scores, denominators, v_cache, weighted_values, layer_num, seq_len, max_seq_len);
        checkCuda(cudaGetLastError());
    }
};
