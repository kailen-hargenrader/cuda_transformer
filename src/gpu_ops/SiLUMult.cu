#include "SiLUMult.cuh"
#include "../ErrorCheck.h"
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_mult_vec_kernel(__nv_bfloat162 *x, const __nv_bfloat162 *y, int32_t pair_count) {
    const int32_t stride = static_cast<int32_t>(gridDim.x) * kThreads;
    for (int32_t i = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
         i < pair_count; i += stride) {
        const __nv_bfloat162 xv = x[i];
        const __nv_bfloat162 yv = y[i];

        const float x0 = __bfloat162float(__low2bfloat16(xv));
        const float x1 = __bfloat162float(__high2bfloat16(xv));
        const float y0 = __bfloat162float(__low2bfloat16(yv));
        const float y1 = __bfloat162float(__high2bfloat16(yv));

        x[i] = __floats2bfloat162_rn(silu(x0) * y0, silu(x1) * y1);
    }
}

__global__ void silu_mult_scalar_kernel(__nv_bfloat16 *x, const __nv_bfloat16 *y, int32_t len) {
    const int32_t stride = static_cast<int32_t>(gridDim.x) * kThreads;
    for (int32_t i = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
         i < len; i += stride) {
        const float xv = __bfloat162float(x[i]);
        const float yv = __bfloat162float(y[i]);
        x[i] = __float2bfloat16(silu(xv) * yv);
    }
}

__global__ void silu_mult_tail_kernel(__nv_bfloat16 *x, const __nv_bfloat16 *y, int32_t idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float xv = __bfloat162float(x[idx]);
        const float yv = __bfloat162float(y[idx]);
        x[idx] = __float2bfloat16(silu(xv) * yv);
    }
}

} // namespace

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y,
                                  cudaStream_t stream) {
    auto *x_ptr = static_cast<__nv_bfloat16 *>(x->data);
    const auto *y_ptr = static_cast<const __nv_bfloat16 *>(y->data);
    const int32_t len = static_cast<int32_t>(x->size / sizeof(__nv_bfloat16));
    const int blocks = std::max(1, (len + kThreads - 1) / kThreads);

    const bool aligned = (reinterpret_cast<uintptr_t>(x_ptr) % alignof(__nv_bfloat162) == 0) &&
                         (reinterpret_cast<uintptr_t>(y_ptr) % alignof(__nv_bfloat162) == 0);
    if (aligned && len >= 2) {
        const int32_t pair_count = len / 2;
        silu_mult_vec_kernel<<<blocks, kThreads, 0, stream>>>(reinterpret_cast<__nv_bfloat162 *>(x_ptr),
                                                              reinterpret_cast<const __nv_bfloat162 *>(y_ptr),
                                                              pair_count);
        checkCuda(cudaGetLastError());
        if ((len & 1) != 0) {
            silu_mult_tail_kernel<<<1, 1, 0, stream>>>(x_ptr, y_ptr, len - 1);
            checkCuda(cudaGetLastError());
        }
        return;
    }

    silu_mult_scalar_kernel<<<blocks, kThreads, 0, stream>>>(x_ptr, y_ptr, len);
    checkCuda(cudaGetLastError());
}
