#pragma once

#include <memory>

#include "Qwen2Layer.cuh"
#include "Qwen2Config.h"
#include "../ErrorCheck.h"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/ArgMax.cuh"
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include <stdexcept>

template<Qwen2Size QWEN2_SIZE>
class Qwen2Model {
    cudaStream_t stream;
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    using Qwen2Layer = Qwen2Layer<QWEN2_SIZE>;

    Qwen2Model() {
        checkCuda(cudaStreamCreate(&stream));
        hidden_state = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        logits = std::make_shared<CudaBuffer>(Qwen2Config::vocab_size() * sizeof(__nv_bfloat16));
    }

    ~Qwen2Model() {
        checkCuda(cudaStreamDestroy(stream));
    }

    std::shared_ptr<CudaBuffer> embedding_weight; // (vocab_size, hidden_size)
    std::shared_ptr<Qwen2Layer> layers[Qwen2Config::num_layers()];
    LayerNorm final_layernorm{Qwen2Config::hidden_size()}; // (hidden_size,)
    std::shared_ptr<CudaBuffer> hidden_state; // (hidden_size,)
    std::shared_ptr<CudaBuffer> logits; // (vocab_size,)
    ArgMax logits_argmax{Qwen2Config::vocab_size()};

    /**
     *
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param seq_len current sequence length
     * @param input_tok_id last token in the sequence
     * @param temperature Sampling parameter. Always set to 0, for deterministic (greedy) decoding, see https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting.
     *                    You do not need to implement any other sampling methods.
     * @return
     */
    int32_t forward(const std::shared_ptr<CudaBuffer> &k_cache, const std::shared_ptr<CudaBuffer> &v_cache, int32_t seq_len, int32_t input_tok_id, float temperature) {
        if (temperature != 0.0f) {
            throw std::runtime_error("temperature > 0 is not supported");
        }
        if (!Qwen2Config::embedding_tying()) {
            throw std::runtime_error("current implementation requires tied embeddings");
        }

        auto *embedding_ptr = static_cast<__nv_bfloat16 *>(embedding_weight->data);
        auto *hidden_state_ptr = static_cast<__nv_bfloat16 *>(hidden_state->data);
        const auto *embedding_row = embedding_ptr + static_cast<size_t>(input_tok_id) * Qwen2Config::hidden_size();
        checkCuda(cudaMemcpyAsync(
            hidden_state_ptr,
            embedding_row,
            hidden_state->size,
            cudaMemcpyDefault,
            stream));

        for (auto &layer : layers) {
            layer->forward(k_cache, v_cache, hidden_state, seq_len, stream);
        }

        final_layernorm.normalize_hidden_state(hidden_state, hidden_state, stream);
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::vocab_size(),
            Qwen2Config::hidden_size(),
            embedding_ptr,
            nullptr,
            hidden_state_ptr,
            static_cast<__nv_bfloat16 *>(logits->data),
            stream);

        int32_t *next_token = logits_argmax.bf16_argmax(logits, stream);
        checkCuda(cudaStreamSynchronize(stream));
        return *next_token;
    }
};