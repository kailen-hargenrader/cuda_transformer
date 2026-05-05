#include "ArgMax.cuh"
#include "../ErrorCheck.h"
#include <algorithm>
#include <climits>
#include <cuda_bf16.h>
#include <cmath>

namespace {

constexpr int kBlock = 256;
constexpr int kItemsPerThread = 8;
constexpr int kItemsPerBlock = kBlock * kItemsPerThread;

struct ArgMaxPair {
    float value;
    int32_t index;
};

__device__ __forceinline__ void argmax_update(float &best_v, int32_t &best_i, float value, int32_t index) {
    if (value > best_v || (value == best_v && index < best_i)) {
        best_v = value;
        best_i = index;
    }
}

__device__ __forceinline__ void warp_reduce_argmax(float &best_v, int32_t &best_i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_v = __shfl_down_sync(0xffffffffu, best_v, offset);
        const int32_t other_i = __shfl_down_sync(0xffffffffu, best_i, offset);
        argmax_update(best_v, best_i, other_v, other_i);
    }
}

__device__ __forceinline__ void write_block_result(float best_v, int32_t best_i, ArgMaxPair *out) {
    __shared__ float warp_vals[kBlock / 32];
    __shared__ int32_t warp_idx[kBlock / 32];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;

    warp_reduce_argmax(best_v, best_i);
    if (lane == 0) {
        warp_vals[warp_id] = best_v;
        warp_idx[warp_id] = best_i;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_v = lane < kBlock / 32 ? warp_vals[lane] : -HUGE_VALF;
        int32_t block_i = lane < kBlock / 32 ? warp_idx[lane] : INT_MAX;
        warp_reduce_argmax(block_v, block_i);
        if (lane == 0) {
            out[blockIdx.x].value = block_v;
            out[blockIdx.x].index = block_i;
        }
    }
}

__global__ void argmax_bf16_pass(const __nv_bfloat16 *data, int32_t len, ArgMaxPair *out) {
    const int32_t base = static_cast<int32_t>(blockIdx.x) * kItemsPerBlock;
    float best_v = -HUGE_VALF;
    int32_t best_i = INT_MAX;

    for (int32_t offset = static_cast<int32_t>(threadIdx.x); offset < kItemsPerBlock; offset += kBlock) {
        const int32_t i = base + offset;
        if (i < len) {
            argmax_update(best_v, best_i, __bfloat162float(data[i]), i);
        }
    }

    write_block_result(best_v, best_i, out);
}

__global__ void argmax_pair_pass(const ArgMaxPair *in, int32_t len, ArgMaxPair *out) {
    const int32_t base = static_cast<int32_t>(blockIdx.x) * kItemsPerBlock;
    float best_v = -HUGE_VALF;
    int32_t best_i = INT_MAX;

    for (int32_t offset = static_cast<int32_t>(threadIdx.x); offset < kItemsPerBlock; offset += kBlock) {
        const int32_t i = base + offset;
        if (i < len) {
            argmax_update(best_v, best_i, in[i].value, in[i].index);
        }
    }

    write_block_result(best_v, best_i, out);
}

int32_t partial_count_for_len(int32_t len) {
    return std::max(1, (len + kItemsPerBlock - 1) / kItemsPerBlock);
}

size_t temp_size_for_len(int32_t len) {
    const size_t pairs_bytes = static_cast<size_t>(partial_count_for_len(len)) * sizeof(ArgMaxPair);
    return pairs_bytes * 2;
}

void layout_buffers(void *base, ArgMaxPair **a, ArgMaxPair **b, int32_t max_partials) {
    *a = static_cast<ArgMaxPair *>(base);
    *b = *a + max_partials;
}

} // namespace

ArgMax::ArgMax(int32_t len) {
    temp_space = std::make_shared<CudaBuffer>(temp_size_for_len(len));
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    const auto *raw_bf16 = static_cast<const __nv_bfloat16 *>(bf16_data->data);
    int32_t current_count = static_cast<int32_t>(bf16_data->size / sizeof(__nv_bfloat16));
    const int32_t max_partials = partial_count_for_len(current_count);

    ArgMaxPair *buffer_a{};
    ArgMaxPair *buffer_b{};
    layout_buffers(temp_space->data, &buffer_a, &buffer_b, max_partials);

    int32_t next_count = partial_count_for_len(current_count);
    argmax_bf16_pass<<<next_count, kBlock, 0, stream>>>(raw_bf16, current_count, buffer_a);
    checkCuda(cudaGetLastError());

    ArgMaxPair *current = buffer_a;
    ArgMaxPair *next = buffer_b;
    current_count = next_count;

    while (current_count > 1) {
        next_count = partial_count_for_len(current_count);
        argmax_pair_pass<<<next_count, kBlock, 0, stream>>>(current, current_count, next);
        checkCuda(cudaGetLastError());

        ArgMaxPair *tmp = current;
        current = next;
        next = tmp;
        current_count = next_count;
    }

    return &current[0].index;
}
