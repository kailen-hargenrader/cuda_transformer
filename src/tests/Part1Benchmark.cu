#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/ArgMax.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/SiLUMult.cuh"
#include "../qwen2/Qwen2Config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using BenchConfig = Qwen2Config<QWEN2_0_5B>;

enum class ImplMode {
    Baseline,
    Optimized,
    Both,
};

struct Options {
    std::string op = "all";
    std::string shape = "all";
    ImplMode impl = ImplMode::Both;
    int warmup = 20;
    int iters = 200;
    bool verify = true;
};

Options parse_options(int argc, char **argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--op") {
            opts.op = need_value(arg);
        } else if (arg == "--shape") {
            opts.shape = need_value(arg);
        } else if (arg == "--warmup") {
            opts.warmup = std::stoi(need_value(arg));
        } else if (arg == "--iters") {
            opts.iters = std::stoi(need_value(arg));
        } else if (arg == "--impl") {
            const std::string value = need_value(arg);
            if (value == "baseline") {
                opts.impl = ImplMode::Baseline;
            } else if (value == "optimized") {
                opts.impl = ImplMode::Optimized;
            } else if (value == "both") {
                opts.impl = ImplMode::Both;
            } else {
                throw std::runtime_error("unknown impl: " + value);
            }
        } else if (arg == "--help") {
            std::cout
                << "Usage: part1bench [--op all|argmax|matvec|layernorm|rope|silu] "
                << "[--shape all|q_proj|ffn_up|logits] [--impl baseline|optimized|both] "
                << "[--warmup N] [--iters N] [--skip-verify]\n";
            std::exit(0);
        } else if (arg == "--skip-verify") {
            opts.verify = false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opts;
}

bool bf16_close(__nv_bfloat16 a, __nv_bfloat16 b, float rtol = 1e-2f, float atol = 1e-4f) {
    const float af = __bfloat162float(a);
    const float bf = __bfloat162float(b);
    return fabsf(af - bf) <= atol + rtol * fabsf(bf);
}

void check_bf16_same(const __nv_bfloat16 *a, const __nv_bfloat16 *b, int32_t len, const std::string &name) {
    for (int32_t i = 0; i < len; ++i) {
        if (!bf16_close(a[i], b[i])) {
            throw std::runtime_error(name + " mismatch at index " + std::to_string(i));
        }
    }
}

template<typename T>
void fill_random(T *dst, int32_t len, std::mt19937 &gen, std::normal_distribution<float> &dist);

template<>
void fill_random(__nv_bfloat16 *dst, int32_t len, std::mt19937 &gen, std::normal_distribution<float> &dist) {
    for (int32_t i = 0; i < len; ++i) {
        dst[i] = __float2bfloat16(dist(gen));
    }
}

template<typename Fn>
float time_kernel_ms(cudaStream_t stream, int warmup, int iters, Fn &&fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    checkCuda(cudaStreamSynchronize(stream));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    checkCuda(cudaEventRecord(stop, stream));
    checkCuda(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(iters);
}

void print_result(const std::string &op, const std::string &shape, const std::string &impl, float ms) {
    std::cout << op << "," << shape << "," << impl << "," << ms << "\n";
}

namespace baseline {

constexpr int kBlock = 256;

template<typename input_float_t>
__global__ void matvec_kernel(int32_t m, int32_t k, const __nv_bfloat16 *__restrict__ mat,
                              const __nv_bfloat16 *__restrict__ bias,
                              const input_float_t *__restrict__ vec, __nv_bfloat16 *__restrict__ out) {
    extern __shared__ char smem_raw[];
    auto *vec_smem = reinterpret_cast<input_float_t *>(smem_raw);

    for (int32_t j = static_cast<int32_t>(threadIdx.x); j < k; j += static_cast<int32_t>(blockDim.x)) {
        vec_smem[j] = vec[j];
    }
    __syncthreads();

    const int32_t row = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) +
                        static_cast<int32_t>(threadIdx.x);
    if (row >= m) {
        return;
    }

    float sum = 0.0f;
    if (bias != nullptr) {
        sum = __bfloat162float(bias[row]);
    }
    const __nv_bfloat16 *row_ptr = mat + static_cast<size_t>(row) * static_cast<size_t>(k);
    for (int32_t j = 0; j < k; ++j) {
        sum += __bfloat162float(row_ptr[j]) * static_cast<float>(vec_smem[j]);
    }
    out[row] = __float2bfloat16(sum);
}

template<typename input_float_t>
void matvec(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, input_float_t *vec,
            __nv_bfloat16 *out, cudaStream_t stream) {
    const int blocks = (m + kBlock - 1) / kBlock;
    const size_t smem = static_cast<size_t>(k) * sizeof(input_float_t);
    matvec_kernel<input_float_t><<<blocks, kBlock, smem, stream>>>(m, k, mat, bias, vec, out);
    checkCuda(cudaGetLastError());
}

__global__ void rope_kernel(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim, int32_t position_idx,
                            float theta_base) {
    const int32_t half = head_dim / 2;
    const int32_t total_pairs = num_heads * half;
    const int32_t tid = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) +
                        static_cast<int32_t>(threadIdx.x);
    const int32_t stride = static_cast<int32_t>(gridDim.x) * static_cast<int32_t>(blockDim.x);

    for (int32_t p = tid; p < total_pairs; p += stride) {
        const int32_t head = p / half;
        const int32_t j = p - head * half;
        __nv_bfloat16 *row = x + static_cast<size_t>(head) * static_cast<size_t>(head_dim);

        const float theta_idx_frac = static_cast<float>(j) / static_cast<float>(half);
        const float theta = powf(theta_base, -theta_idx_frac);
        const float angle = theta * static_cast<float>(position_idx);
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float a = __bfloat162float(row[j]);
        const float b = __bfloat162float(row[j + half]);
        row[j] = __float2bfloat16(a * c - b * s);
        row[j + half] = __float2bfloat16(b * c + a * s);
    }
}

void rope(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim, int32_t position_idx, float theta_base,
          cudaStream_t stream) {
    constexpr int threads = 256;
    const int32_t total_pairs = num_heads * (head_dim / 2);
    const int32_t blocks = std::max(1, (total_pairs + threads - 1) / threads);
    rope_kernel<<<blocks, threads, 0, stream>>>(x, num_heads, head_dim, position_idx, theta_base);
    checkCuda(cudaGetLastError());
}

__global__ void silu_kernel(__nv_bfloat16 *x, const __nv_bfloat16 *y, int32_t len) {
    const int32_t i = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) +
                      static_cast<int32_t>(threadIdx.x);
    if (i < len) {
        const float xv = __bfloat162float(x[i]);
        const float yv = __bfloat162float(y[i]);
        x[i] = __float2bfloat16((xv / (1.0f + expf(-xv))) * yv);
    }
}

void silu(std::shared_ptr<CudaBuffer> x, std::shared_ptr<CudaBuffer> y, cudaStream_t stream) {
    constexpr int threads = 256;
    const int32_t len = static_cast<int32_t>(x->size / sizeof(__nv_bfloat16));
    const int blocks = (len + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(static_cast<__nv_bfloat16 *>(x->data),
                                                static_cast<const __nv_bfloat16 *>(y->data), len);
    checkCuda(cudaGetLastError());
}

struct LayerNormOp {
    std::shared_ptr<CudaBuffer> scratch;
    std::shared_ptr<CudaBuffer> weights;

    explicit LayerNormOp(int32_t len) : scratch(std::make_shared<CudaBuffer>(sizeof(float))) {
        (void)len;
    }
};

__global__ void layernorm_sum_kernel(const __nv_bfloat16 *in, int32_t len, float *sum_sq_out) {
    const int32_t i = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) +
                      static_cast<int32_t>(threadIdx.x);
    const int32_t stride = static_cast<int32_t>(gridDim.x) * static_cast<int32_t>(blockDim.x);
    float acc = 0.0f;
    for (int32_t j = i; j < len; j += stride) {
        const float x = __bfloat162float(in[j]);
        acc += x * x;
    }
    __shared__ float smem[256];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sum_sq_out, smem[0]);
    }
}

__global__ void layernorm_scale_kernel(const __nv_bfloat16 *in, const __nv_bfloat16 *weights,
                                       __nv_bfloat16 *out, int32_t len, float inv_rms) {
    const int32_t i = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) +
                      static_cast<int32_t>(threadIdx.x);
    if (i < len) {
        out[i] = __float2bfloat16(__bfloat162float(weights[i]) * __bfloat162float(in[i]) * inv_rms);
    }
}

void layernorm(LayerNormOp &op, const std::shared_ptr<CudaBuffer> &hidden_state,
               const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    const int32_t len = static_cast<int32_t>(hidden_state->size / sizeof(__nv_bfloat16));
    float *sum_ptr = static_cast<float *>(op.scratch->data);
    checkCuda(cudaMemsetAsync(sum_ptr, 0, sizeof(float), stream));
    constexpr unsigned threads = 256;
    const unsigned blocks = std::min(1024U, std::max(1U, (static_cast<unsigned>(len) + threads - 1) / threads));
    layernorm_sum_kernel<<<blocks, threads, 0, stream>>>(static_cast<const __nv_bfloat16 *>(hidden_state->data),
                                                         len, sum_ptr);
    checkCuda(cudaGetLastError());
    float sum_host = 0.0f;
    checkCuda(cudaMemcpyAsync(&sum_host, sum_ptr, sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaStreamSynchronize(stream));
    const float inv_rms = rsqrtf(sum_host / static_cast<float>(len) + LayerNorm::EPS);
    const unsigned scale_blocks = (static_cast<unsigned>(len) + threads - 1) / threads;
    layernorm_scale_kernel<<<scale_blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(hidden_state->data), static_cast<const __nv_bfloat16 *>(op.weights->data),
        static_cast<__nv_bfloat16 *>(output->data), len, inv_rms);
    checkCuda(cudaGetLastError());
}

struct ArgMaxOp {
    std::shared_ptr<CudaBuffer> temp;

    explicit ArgMaxOp(int32_t len) {
        const int32_t blocks = std::max(1, (len + kBlock - 1) / kBlock);
        size_t off = static_cast<size_t>(blocks) * sizeof(float);
        off = (off + alignof(int32_t) - 1) & ~(alignof(int32_t) - 1);
        temp = std::make_shared<CudaBuffer>(off + static_cast<size_t>(blocks) * sizeof(int32_t) + sizeof(int32_t));
    }
};

__device__ __forceinline__ void argmax_update(float &best_v, int32_t &best_i, float value, int32_t index) {
    if (value > best_v || (value == best_v && index < best_i)) {
        best_v = value;
        best_i = index;
    }
}

__global__ void argmax_per_block_kernel(const __nv_bfloat16 *data, int32_t len, float *partial_vals,
                                        int32_t *partial_idx) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x) * kBlock + static_cast<int32_t>(threadIdx.x);
    float v = -HUGE_VALF;
    int32_t i = INT_MAX;
    if (idx < len) {
        v = __bfloat162float(data[idx]);
        i = idx;
    }

    __shared__ float smem_v[kBlock];
    __shared__ int32_t smem_i[kBlock];
    smem_v[threadIdx.x] = v;
    smem_i[threadIdx.x] = i;
    __syncthreads();

    for (unsigned stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            argmax_update(smem_v[threadIdx.x], smem_i[threadIdx.x], smem_v[threadIdx.x + stride],
                          smem_i[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_vals[blockIdx.x] = smem_v[0];
        partial_idx[blockIdx.x] = smem_i[0];
    }
}

__global__ void argmax_merge_kernel(const float *partial_vals, const int32_t *partial_idx, int32_t len,
                                    int32_t *out_idx) {
    float best_v = -HUGE_VALF;
    int32_t best_i = INT_MAX;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < len; i += kBlock) {
        argmax_update(best_v, best_i, partial_vals[i], partial_idx[i]);
    }

    __shared__ float smem_v[kBlock];
    __shared__ int32_t smem_i[kBlock];
    smem_v[threadIdx.x] = best_v;
    smem_i[threadIdx.x] = best_i;
    __syncthreads();

    for (unsigned stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            argmax_update(smem_v[threadIdx.x], smem_i[threadIdx.x], smem_v[threadIdx.x + stride],
                          smem_i[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_idx = smem_i[0];
    }
}

int32_t *argmax(ArgMaxOp &op, const std::shared_ptr<CudaBuffer> &data, cudaStream_t stream) {
    const int32_t len = static_cast<int32_t>(data->size / sizeof(__nv_bfloat16));
    const int32_t blocks = std::max(1, (len + kBlock - 1) / kBlock);

    auto *partial_vals = static_cast<float *>(op.temp->data);
    size_t off = static_cast<size_t>(blocks) * sizeof(float);
    off = (off + alignof(int32_t) - 1) & ~(alignof(int32_t) - 1);
    auto *partial_idx = reinterpret_cast<int32_t *>(static_cast<char *>(op.temp->data) + off);
    auto *out_idx = partial_idx + blocks;

    argmax_per_block_kernel<<<blocks, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16 *>(data->data), len,
                                                           partial_vals, partial_idx);
    checkCuda(cudaGetLastError());
    argmax_merge_kernel<<<1, kBlock, 0, stream>>>(partial_vals, partial_idx, blocks, out_idx);
    checkCuda(cudaGetLastError());
    return out_idx;
}

} // namespace baseline

void bench_argmax(const Options &opts, cudaStream_t stream) {
    if (opts.op != "all" && opts.op != "argmax") {
        return;
    }
    if (opts.shape != "all" && opts.shape != "logits") {
        return;
    }

    constexpr int32_t len = static_cast<int32_t>(BenchConfig::vocab_size());
    auto input = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 100.0f);
    fill_random(static_cast<__nv_bfloat16 *>(input->data), len, gen, dist);

    baseline::ArgMaxOp baseline_op(len);
    ArgMax optimized_op(len);

    if (opts.verify) {
        int32_t *baseline_idx = baseline::argmax(baseline_op, input, stream);
        int32_t *optimized_idx = optimized_op.bf16_argmax(input, stream);
        checkCuda(cudaStreamSynchronize(stream));
        if (*baseline_idx != *optimized_idx) {
            throw std::runtime_error("argmax mismatch");
        }
    }

    if (opts.impl == ImplMode::Baseline || opts.impl == ImplMode::Both) {
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters,
                                        [&]() { (void)baseline::argmax(baseline_op, input, stream); });
        print_result("argmax", "logits", "baseline", ms);
    }
    if (opts.impl == ImplMode::Optimized || opts.impl == ImplMode::Both) {
        const float ms =
            time_kernel_ms(stream, opts.warmup, opts.iters, [&]() { (void)optimized_op.bf16_argmax(input, stream); });
        print_result("argmax", "logits", "optimized", ms);
    }
}

template<typename input_float_t>
void bench_matvec_case(const Options &opts, cudaStream_t stream, const std::string &shape_name, int32_t m, int32_t k,
                       bool use_bias) {
    if (opts.shape != "all" && opts.shape != shape_name) {
        return;
    }

    auto mat = std::make_shared<CudaBuffer>(static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(__nv_bfloat16));
    auto vec = std::make_shared<CudaBuffer>(static_cast<size_t>(k) * sizeof(input_float_t));
    std::shared_ptr<CudaBuffer> bias = use_bias ? std::make_shared<CudaBuffer>(static_cast<size_t>(m) * sizeof(__nv_bfloat16))
                                                : nullptr;
    auto out_baseline = std::make_shared<CudaBuffer>(static_cast<size_t>(m) * sizeof(__nv_bfloat16));
    auto out_optimized = std::make_shared<CudaBuffer>(static_cast<size_t>(m) * sizeof(__nv_bfloat16));

    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 1.0f);
    fill_random(static_cast<__nv_bfloat16 *>(mat->data), m * k, gen, dist);
    fill_random(static_cast<input_float_t *>(vec->data), k, gen, dist);
    if (bias) {
        fill_random(static_cast<__nv_bfloat16 *>(bias->data), m, gen, dist);
    }

    if (opts.verify) {
        baseline::matvec(m, k, static_cast<__nv_bfloat16 *>(mat->data),
                         bias ? static_cast<__nv_bfloat16 *>(bias->data) : nullptr,
                         static_cast<input_float_t *>(vec->data), static_cast<__nv_bfloat16 *>(out_baseline->data),
                         stream);
        MatrixVectorMultiply::bf16_matmul(
            m, k, static_cast<__nv_bfloat16 *>(mat->data), bias ? static_cast<__nv_bfloat16 *>(bias->data) : nullptr,
            static_cast<input_float_t *>(vec->data), static_cast<__nv_bfloat16 *>(out_optimized->data), stream);
        checkCuda(cudaStreamSynchronize(stream));
        check_bf16_same(static_cast<__nv_bfloat16 *>(out_baseline->data),
                        static_cast<__nv_bfloat16 *>(out_optimized->data), m, "matvec " + shape_name);
    }

    if (opts.impl == ImplMode::Baseline || opts.impl == ImplMode::Both) {
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters, [&]() {
            baseline::matvec(m, k, static_cast<__nv_bfloat16 *>(mat->data),
                             bias ? static_cast<__nv_bfloat16 *>(bias->data) : nullptr,
                             static_cast<input_float_t *>(vec->data), static_cast<__nv_bfloat16 *>(out_baseline->data),
                             stream);
        });
        print_result("matvec", shape_name, "baseline", ms);
    }

    if (opts.impl == ImplMode::Optimized || opts.impl == ImplMode::Both) {
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters, [&]() {
            MatrixVectorMultiply::bf16_matmul(
                m, k, static_cast<__nv_bfloat16 *>(mat->data), bias ? static_cast<__nv_bfloat16 *>(bias->data) : nullptr,
                static_cast<input_float_t *>(vec->data), static_cast<__nv_bfloat16 *>(out_optimized->data), stream);
        });
        print_result("matvec", shape_name, "optimized", ms);
    }
}

void bench_matvec(const Options &opts, cudaStream_t stream) {
    if (opts.op != "all" && opts.op != "matvec") {
        return;
    }
    bench_matvec_case<__nv_bfloat16>(opts, stream, "q_proj", BenchConfig::queries_size(), BenchConfig::hidden_size(), true);
    bench_matvec_case<__nv_bfloat16>(opts, stream, "ffn_up", BenchConfig::intermediate_size(), BenchConfig::hidden_size(),
                                     false);
    bench_matvec_case<__nv_bfloat16>(opts, stream, "logits", BenchConfig::vocab_size(), BenchConfig::hidden_size(),
                                     false);
}

void bench_layernorm(const Options &opts, cudaStream_t stream) {
    if (opts.op != "all" && opts.op != "layernorm") {
        return;
    }
    if (opts.shape != "all" && opts.shape != "hidden896") {
        return;
    }

    constexpr int32_t len = BenchConfig::hidden_size();
    auto input = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto weights = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto out_baseline = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto out_optimized = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));

    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 1.0f);
    fill_random(static_cast<__nv_bfloat16 *>(input->data), len, gen, dist);
    fill_random(static_cast<__nv_bfloat16 *>(weights->data), len, gen, dist);

    baseline::LayerNormOp baseline_op(len);
    baseline_op.weights = weights;
    LayerNorm optimized_op(len);
    optimized_op.weights = weights;

    if (opts.verify) {
        baseline::layernorm(baseline_op, input, out_baseline, stream);
        optimized_op.normalize_hidden_state(input, out_optimized, stream);
        checkCuda(cudaStreamSynchronize(stream));
        check_bf16_same(static_cast<__nv_bfloat16 *>(out_baseline->data),
                        static_cast<__nv_bfloat16 *>(out_optimized->data), len, "layernorm");
    }

    if (opts.impl == ImplMode::Baseline || opts.impl == ImplMode::Both) {
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters,
                                        [&]() { baseline::layernorm(baseline_op, input, out_baseline, stream); });
        print_result("layernorm", "hidden896", "baseline", ms);
    }
    if (opts.impl == ImplMode::Optimized || opts.impl == ImplMode::Both) {
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters,
                                        [&]() { optimized_op.normalize_hidden_state(input, out_optimized, stream); });
        print_result("layernorm", "hidden896", "optimized", ms);
    }
}

void bench_rope(const Options &opts, cudaStream_t stream) {
    if (opts.op != "all" && opts.op != "rope") {
        return;
    }
    if (opts.shape != "all" && opts.shape != "query14x64") {
        return;
    }

    constexpr int32_t num_heads = BenchConfig::num_query_heads();
    constexpr int32_t head_dim = BenchConfig::head_size();
    constexpr int32_t len = num_heads * head_dim;
    auto source = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto baseline_buf = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto optimized_buf = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));

    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 1.0f);
    fill_random(static_cast<__nv_bfloat16 *>(source->data), len, gen, dist);
    std::memcpy(baseline_buf->data, source->data, source->size);
    std::memcpy(optimized_buf->data, source->data, source->size);

    constexpr int32_t position_idx = 13;
    constexpr float theta_base = 1e6f;
    if (opts.verify) {
        baseline::rope(static_cast<__nv_bfloat16 *>(baseline_buf->data), num_heads, head_dim, position_idx, theta_base,
                       stream);
        RoPE::apply_rope_to_qk(static_cast<__nv_bfloat16 *>(optimized_buf->data), num_heads, head_dim, position_idx,
                               theta_base, stream);
        checkCuda(cudaStreamSynchronize(stream));
        check_bf16_same(static_cast<__nv_bfloat16 *>(baseline_buf->data),
                        static_cast<__nv_bfloat16 *>(optimized_buf->data), len, "rope");
    }

    if (opts.impl == ImplMode::Baseline || opts.impl == ImplMode::Both) {
        std::memcpy(baseline_buf->data, source->data, source->size);
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters, [&]() {
            baseline::rope(static_cast<__nv_bfloat16 *>(baseline_buf->data), num_heads, head_dim, position_idx,
                           theta_base, stream);
        });
        print_result("rope", "query14x64", "baseline", ms);
    }
    if (opts.impl == ImplMode::Optimized || opts.impl == ImplMode::Both) {
        std::memcpy(optimized_buf->data, source->data, source->size);
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters, [&]() {
            RoPE::apply_rope_to_qk(static_cast<__nv_bfloat16 *>(optimized_buf->data), num_heads, head_dim, position_idx,
                                   theta_base, stream);
        });
        print_result("rope", "query14x64", "optimized", ms);
    }
}

void bench_silu(const Options &opts, cudaStream_t stream) {
    if (opts.op != "all" && opts.op != "silu") {
        return;
    }
    if (opts.shape != "all" && opts.shape != "ffn4864") {
        return;
    }

    constexpr int32_t len = BenchConfig::intermediate_size();
    auto source_x = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto source_y = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto baseline_x = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));
    auto optimized_x = std::make_shared<CudaBuffer>(static_cast<size_t>(len) * sizeof(__nv_bfloat16));

    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 10.0f);
    fill_random(static_cast<__nv_bfloat16 *>(source_x->data), len, gen, dist);
    fill_random(static_cast<__nv_bfloat16 *>(source_y->data), len, gen, dist);
    std::memcpy(baseline_x->data, source_x->data, source_x->size);
    std::memcpy(optimized_x->data, source_x->data, source_x->size);

    if (opts.verify) {
        baseline::silu(baseline_x, source_y, stream);
        SiLUMult::silu_mult_in_place(optimized_x, source_y, stream);
        checkCuda(cudaStreamSynchronize(stream));
        check_bf16_same(static_cast<__nv_bfloat16 *>(baseline_x->data),
                        static_cast<__nv_bfloat16 *>(optimized_x->data), len, "silu");
    }

    if (opts.impl == ImplMode::Baseline || opts.impl == ImplMode::Both) {
        std::memcpy(baseline_x->data, source_x->data, source_x->size);
        const float ms =
            time_kernel_ms(stream, opts.warmup, opts.iters, [&]() { baseline::silu(baseline_x, source_y, stream); });
        print_result("silu", "ffn4864", "baseline", ms);
    }
    if (opts.impl == ImplMode::Optimized || opts.impl == ImplMode::Both) {
        std::memcpy(optimized_x->data, source_x->data, source_x->size);
        const float ms = time_kernel_ms(stream, opts.warmup, opts.iters,
                                        [&]() { SiLUMult::silu_mult_in_place(optimized_x, source_y, stream); });
        print_result("silu", "ffn4864", "optimized", ms);
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options opts = parse_options(argc, argv);
        cudaStream_t stream{};
        checkCuda(cudaStreamCreate(&stream));

        std::cout << "op,shape,impl,avg_ms\n";
        bench_argmax(opts, stream);
        bench_matvec(opts, stream);
        bench_layernorm(opts, stream);
        bench_rope(opts, stream);
        bench_silu(opts, stream);

        checkCuda(cudaStreamDestroy(stream));
    } catch (const std::exception &e) {
        std::cerr << "part1bench error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
