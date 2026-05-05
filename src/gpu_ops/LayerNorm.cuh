#pragma once

#include "../CudaBuffer.cuh"
#include <memory>

/**
 * Layer normalization without bias, as used in the T5 paper https://arxiv.org/pdf/1910.10683
 * Internally, uses float32 for all calculations, and only rounds back to bfloat16 at the end.
 */
class LayerNorm {
    /// Scratch for partial sums followed by one float holding the inverse RMS.
    std::shared_ptr<CudaBuffer> temp_space;
    int32_t partial_count;

public:
    /// epsilon to add in denominator square root, for numerical stability
    static constexpr float EPS = 1.0e-6f;

    /// GPU bf16 vector of shape (hidden_size,)
    std::shared_ptr<CudaBuffer> weights;

    /**
     * Initialize temporary space
     */
    explicit LayerNorm(int32_t len);

    /**
     * Apply variance correction and scaling factors to hidden state
     * @param hidden_state GPU bf16 input
     * @param output Location to write output
     * @param stream CUDA stream for asycnhronous operation
     */
    void normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream);
};
