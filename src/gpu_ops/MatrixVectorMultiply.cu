#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"
#include <cuda_bf16.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kTileK = 256;

__device__ __forceinline__ float vec_to_float(float v) {
    return v;
}

__device__ __forceinline__ float vec_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template<typename input_float_t>
__global__ void bf16_matvec_kernel(int32_t m, int32_t k, const __nv_bfloat16 *__restrict__ mat,
                                   const __nv_bfloat16 *__restrict__ bias,
                                   const input_float_t *__restrict__ vec, __nv_bfloat16 *__restrict__ out) {
    extern __shared__ char smem_raw[];
    auto *vec_tile = reinterpret_cast<input_float_t *>(smem_raw);

    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int32_t row = static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + warp_id;

    float sum = 0.0f;
    const __nv_bfloat16 *row_ptr = row < m ? mat + static_cast<size_t>(row) * static_cast<size_t>(k) : nullptr;

    for (int32_t tile_start = 0; tile_start < k; tile_start += kTileK) {
        const int32_t tile_len = min(kTileK, k - tile_start);
        for (int32_t j = static_cast<int32_t>(threadIdx.x); j < tile_len; j += kThreads) {
            vec_tile[j] = vec[tile_start + j];
        }
        __syncthreads();

        if (row < m) {
            for (int32_t j = lane; j < tile_len; j += kWarpSize) {
                sum += __bfloat162float(row_ptr[tile_start + j]) * vec_to_float(vec_tile[j]);
            }
        }
        __syncthreads();
    }

    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }

    if (row < m && lane == 0) {
        if (bias != nullptr) {
            sum += __bfloat162float(bias[row]);
        }
        out[row] = __float2bfloat16(sum);
    }
}

} // namespace

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias,
                                       input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    const int blocks = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const size_t smem_size = static_cast<size_t>(kTileK) * sizeof(input_float_t);
    bf16_matvec_kernel<input_float_t><<<blocks, kThreads, smem_size, stream>>>(m, k, mat, bias, vec, out);
    checkCuda(cudaGetLastError());
}

template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat,
                                                               __nv_bfloat16 *bias, __nv_bfloat16 *vec,
                                                               __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat,
                                                       __nv_bfloat16 *bias, float *vec, __nv_bfloat16 *out,
                                                       cudaStream_t stream);
