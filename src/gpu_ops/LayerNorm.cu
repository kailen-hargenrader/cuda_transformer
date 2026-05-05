#include "LayerNorm.cuh"
#include "../ErrorCheck.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cstddef>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void layernorm_partial_sum_kernel(const __nv_bfloat16 *in, int32_t len, float *partials) {
    float sum = 0.0f;
    const int32_t stride = static_cast<int32_t>(gridDim.x) * kThreads;
    for (int32_t i = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
         i < len; i += stride) {
        const float x = __bfloat162float(in[i]);
        sum += x * x;
    }

    sum = warp_sum(sum);

    __shared__ float warp_sums[kThreads / 32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < kThreads / 32 ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            partials[blockIdx.x] = block_sum;
        }
    }
}

__global__ void layernorm_finalize_kernel(const float *partials, int32_t partial_count, int32_t len, float *inv_rms) {
    float sum = 0.0f;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < partial_count; i += kThreads) {
        sum += partials[i];
    }

    sum = warp_sum(sum);

    __shared__ float warp_sums[kThreads / 32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total = lane < kThreads / 32 ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            *inv_rms = rsqrtf(total / static_cast<float>(len) + LayerNorm::EPS);
        }
    }
}

__global__ void layernorm_scale_kernel(const __nv_bfloat16 *in, const __nv_bfloat16 *weights, __nv_bfloat16 *out,
                                       int32_t len, const float *inv_rms_ptr) {
    const float inv_rms = *inv_rms_ptr;
    const int32_t i = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
    if (i < len) {
        const float x = __bfloat162float(in[i]);
        const float w = __bfloat162float(weights[i]);
        out[i] = __float2bfloat16(w * x * inv_rms);
    }
}

size_t temp_space_size_for_partials(int32_t partial_count) {
    return static_cast<size_t>(partial_count + 1) * sizeof(float);
}

void layout_temp_space(void *base, int32_t partial_count, float **partials, float **inv_rms) {
    *partials = static_cast<float *>(base);
    *inv_rms = *partials + partial_count;
}

} // namespace

LayerNorm::LayerNorm(int32_t len) {
    partial_count = std::max(1, (len + kThreads - 1) / kThreads);
    temp_space = std::make_shared<CudaBuffer>(temp_space_size_for_partials(partial_count));
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state,
                                       const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    const auto *in = static_cast<const __nv_bfloat16 *>(hidden_state->data);
    auto *out = static_cast<__nv_bfloat16 *>(output->data);
    const auto *w = static_cast<const __nv_bfloat16 *>(weights->data);
    const int32_t len = static_cast<int32_t>(hidden_state->size / sizeof(__nv_bfloat16));

    float *partials{};
    float *inv_rms{};
    layout_temp_space(temp_space->data, partial_count, &partials, &inv_rms);

    layernorm_partial_sum_kernel<<<partial_count, kThreads, 0, stream>>>(in, len, partials);
    checkCuda(cudaGetLastError());
    layernorm_finalize_kernel<<<1, kThreads, 0, stream>>>(partials, partial_count, len, inv_rms);
    checkCuda(cudaGetLastError());

    const int32_t scale_blocks = (len + kThreads - 1) / kThreads;
    layernorm_scale_kernel<<<scale_blocks, kThreads, 0, stream>>>(in, w, out, len, inv_rms);
    checkCuda(cudaGetLastError());
}
