#pragma once

#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

namespace {

constexpr int kResidualAddThreads = 256;

__global__ void add_residual_in_place_kernel(__nv_bfloat16 *hidden_state,
                                             const __nv_bfloat16 *__restrict__ residual,
                                             int32_t len) {
    const int32_t stride = static_cast<int32_t>(gridDim.x) * kResidualAddThreads;
    for (int32_t i = static_cast<int32_t>(blockIdx.x) * kResidualAddThreads + static_cast<int32_t>(threadIdx.x);
         i < len; i += stride) {
        const float hidden = __bfloat162float(hidden_state[i]);
        const float update = __bfloat162float(residual[i]);
        hidden_state[i] = __float2bfloat16(hidden + update);
    }
}

inline void add_residual_in_place(const std::shared_ptr<CudaBuffer> &hidden_state,
                                  const std::shared_ptr<CudaBuffer> &residual,
                                  cudaStream_t stream) {
    const int32_t len = static_cast<int32_t>(hidden_state->size / sizeof(__nv_bfloat16));
    const int32_t blocks = (len + kResidualAddThreads - 1) / kResidualAddThreads;
    add_residual_in_place_kernel<<<blocks, kResidualAddThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(hidden_state->data),
        static_cast<const __nv_bfloat16 *>(residual->data),
        len);
    checkCuda(cudaGetLastError());
}

} // namespace

template<Qwen2Size QWEN2_SIZE>
class Qwen2Layer {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num),
    input_layernorm(Qwen2Config::hidden_size()),
    post_attention_layernorm(Qwen2Config::hidden_size()),
    group_query_attention(max_seq_len) {
        norm_buffer = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        queries_buffer = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(__nv_bfloat16));
        projection_buffer = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        weighted_values_buffer = std::make_shared<CudaBuffer>(
            Qwen2Config::num_query_heads() * Qwen2Config::value_size() * sizeof(float));
        gate_buffer = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        up_buffer = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
    }

    uint32_t layer_num;
    LayerNorm input_layernorm;                              // (hidden_size,)
    std::shared_ptr<CudaBuffer> q_proj_weight;              // (queries_size, hidden_size)
    std::shared_ptr<CudaBuffer> q_proj_bias;                // (queries_size,)
    std::shared_ptr<CudaBuffer> k_proj_weight;              // (keys_size, hidden_size)
    std::shared_ptr<CudaBuffer> k_proj_bias;                // (keys_size,)
    std::shared_ptr<CudaBuffer> v_proj_weight;              // (values_size, hidden_size)
    std::shared_ptr<CudaBuffer> v_proj_bias;                // (values_size,)
    std::shared_ptr<CudaBuffer> o_proj_weight;              // (hidden_size, queries_size)
    LayerNorm post_attention_layernorm;                     // (hidden_size,)
    std::shared_ptr<CudaBuffer> up_proj_weight;             // (intermediate_size, intermediate_size)
    std::shared_ptr<CudaBuffer> gate_proj_weight;           // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> down_proj_weight;           // (hidden_size, intermediate_size)
    GroupQueryAttention<QWEN2_SIZE> group_query_attention;
    std::shared_ptr<CudaBuffer> norm_buffer;                // (hidden_size,)
    std::shared_ptr<CudaBuffer> queries_buffer;             // (queries_size,)
    std::shared_ptr<CudaBuffer> projection_buffer;          // (hidden_size,)
    std::shared_ptr<CudaBuffer> weighted_values_buffer;     // (num_query_heads, value_size) float32
    std::shared_ptr<CudaBuffer> gate_buffer;                // (intermediate_size,)
    std::shared_ptr<CudaBuffer> up_buffer;                  // (intermediate_size,)

    /**
     * Pass the hidden state through this layer. Modifies the hidden state in-place.
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param hidden_state current hidden state bf16 (hidden_size,)
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
        auto *k_cache_ptr = static_cast<__nv_bfloat16 *>(k_cache->data);
        auto *v_cache_ptr = static_cast<__nv_bfloat16 *>(v_cache->data);
        auto *queries_ptr = static_cast<__nv_bfloat16 *>(queries_buffer->data);
        auto *weighted_values_ptr = static_cast<float *>(weighted_values_buffer->data);

        const size_t k_cache_offset =
            static_cast<size_t>(seq_len - 1) * (Qwen2Config::num_layers() * Qwen2Config::keys_size()) +
            static_cast<size_t>(layer_num) * Qwen2Config::keys_size();
        const size_t v_cache_offset =
            static_cast<size_t>(seq_len - 1) * (Qwen2Config::num_layers() * Qwen2Config::values_size()) +
            static_cast<size_t>(layer_num) * Qwen2Config::values_size();
        auto *new_keys_ptr = k_cache_ptr + k_cache_offset;
        auto *new_values_ptr = v_cache_ptr + v_cache_offset;

        input_layernorm.normalize_hidden_state(hidden_state, norm_buffer, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::queries_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16 *>(q_proj_weight->data),
            static_cast<__nv_bfloat16 *>(q_proj_bias->data),
            static_cast<__nv_bfloat16 *>(norm_buffer->data),
            queries_ptr,
            stream);
        RoPE::apply_rope_to_qk(
            queries_ptr,
            Qwen2Config::num_query_heads(),
            Qwen2Config::head_size(),
            seq_len - 1,
            Qwen2Config::rope_theta_base(),
            stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::keys_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16 *>(k_proj_weight->data),
            static_cast<__nv_bfloat16 *>(k_proj_bias->data),
            static_cast<__nv_bfloat16 *>(norm_buffer->data),
            new_keys_ptr,
            stream);
        RoPE::apply_rope_to_qk(
            new_keys_ptr,
            Qwen2Config::num_kv_heads(),
            Qwen2Config::head_size(),
            seq_len - 1,
            Qwen2Config::rope_theta_base(),
            stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::values_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16 *>(v_proj_weight->data),
            static_cast<__nv_bfloat16 *>(v_proj_bias->data),
            static_cast<__nv_bfloat16 *>(norm_buffer->data),
            new_values_ptr,
            stream);

        group_query_attention.sdpa(
            queries_ptr,
            k_cache_ptr,
            v_cache_ptr,
            weighted_values_ptr,
            static_cast<int32_t>(layer_num),
            seq_len,
            stream);

        MatrixVectorMultiply::bf16_matmul<float>(
            Qwen2Config::hidden_size(),
            Qwen2Config::queries_size(),
            static_cast<__nv_bfloat16 *>(o_proj_weight->data),
            nullptr,
            weighted_values_ptr,
            static_cast<__nv_bfloat16 *>(projection_buffer->data),
            stream);
        add_residual_in_place(hidden_state, projection_buffer, stream);

        post_attention_layernorm.normalize_hidden_state(hidden_state, norm_buffer, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16 *>(gate_proj_weight->data),
            nullptr,
            static_cast<__nv_bfloat16 *>(norm_buffer->data),
            static_cast<__nv_bfloat16 *>(gate_buffer->data),
            stream);
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16 *>(up_proj_weight->data),
            nullptr,
            static_cast<__nv_bfloat16 *>(norm_buffer->data),
            static_cast<__nv_bfloat16 *>(up_buffer->data),
            stream);
        SiLUMult::silu_mult_in_place(gate_buffer, up_buffer, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::hidden_size(),
            Qwen2Config::intermediate_size(),
            static_cast<__nv_bfloat16 *>(down_proj_weight->data),
            nullptr,
            static_cast<__nv_bfloat16 *>(gate_buffer->data),
            static_cast<__nv_bfloat16 *>(projection_buffer->data),
            stream);
        add_residual_in_place(hidden_state, projection_buffer, stream);
    }
};
