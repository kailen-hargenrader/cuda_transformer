#include "RoPE.cuh"
#include "../ErrorCheck.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cmath>

namespace {

constexpr int kThreads = 256;

__global__ void rope_kernel(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim, int32_t position_idx,
                            float theta_base) {
    extern __shared__ float inv_freq[];

    const int32_t half = head_dim / 2;
    if (threadIdx.x == 0) {
        inv_freq[0] = 1.0f;
        const float ratio = powf(theta_base, -1.0f / static_cast<float>(half));
        for (int32_t i = 1; i < half; ++i) {
            inv_freq[i] = inv_freq[i - 1] * ratio;
        }
    }
    __syncthreads();

    const int32_t total_pairs = num_heads * half;
    const int32_t stride = static_cast<int32_t>(gridDim.x) * kThreads;
    for (int32_t pair_idx = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
         pair_idx < total_pairs; pair_idx += stride) {
        const int32_t head = pair_idx / half;
        const int32_t j = pair_idx - head * half;
        __nv_bfloat16 *row = x + static_cast<size_t>(head) * static_cast<size_t>(head_dim);

        float s;
        float c;
        sincosf(static_cast<float>(position_idx) * inv_freq[j], &s, &c);

        const float a = __bfloat162float(row[j]);
        const float b = __bfloat162float(row[j + half]);
        row[j] = __float2bfloat16(a * c - b * s);
        row[j + half] = __float2bfloat16(b * c + a * s);
    }
}

} // namespace

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim, int32_t position_idx,
                            float theta_base, cudaStream_t stream) {
    const int32_t half = head_dim / 2;
    const int32_t total_pairs = num_heads * half;
    const int32_t blocks = std::max(1, (total_pairs + kThreads - 1) / kThreads);
    const size_t smem_size = static_cast<size_t>(half) * sizeof(float);
    rope_kernel<<<blocks, kThreads, smem_size, stream>>>(x, num_heads, head_dim, position_idx, theta_base);
    checkCuda(cudaGetLastError());
}
