# transformer

Caltech CS179 Transformer Project

# Background

I recommend you read [Attention Is All You Need](https://arxiv.org/abs/1706.03762) many times, this paper is critical to all transformer models.

We will be implementing the [Qwen2](https://arxiv.org/pdf/2407.10671) model, an open-weights LLMs.
Qwen2's architecture is copied from [Meta's Llama](https://arxiv.org/abs/2407.21783), except Qwen2 uses bias in the QKV matrices.
Both of these architectures implement [grouped-query attention](https://arxiv.org/abs/2305.13245),
[layer normalization](https://arxiv.org/abs/1607.06450v1) with zero mean,
[SwiGLU feed forward networks](https://arxiv.org/abs/2002.05202), and
[rotary positional embeddings](https://arxiv.org/abs/2104.09864).

# Implementation

We will only implement autoregressive decoding, which supports inference (generation) with one token at a time.
With single-token decoding, we avoid the masked attention operator which is the focus of most transformer optimization such as [FlashAttention](https://arxiv.org/pdf/2205.14135).
Additionally, we will not support batching.

We will use the [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) data type to store
model weights and key/value cache to reduce memory bandwidth relative to `float32`. However,
internally, we will use `float32` for accumulation within kernels, to minimize unnecessary floating-point rounding errors.

For simplicity, all tensors in this implementation will use row-major ordering, i.e. the memory is contiguous along the last dimension.
However, this layout is not optimal for performance is all cases.

# Real world LLM inference

This project is missing many features in real-world production LLM inference, such as:
- Batching
- Prefill phase with masked matrix multiply
- Tensor cores
- Quantization
- Advanced model architectures, such as Mixture-of-Experts (MoE)
- Multi-GPU and multi-node parallelization
- KV cache offloading
- Sampling methods, such as Top-K

# Assignment
You will implement all kernels necessary for the LLM. Libraries are not allowed (such as cuBLAS, CUTLASS, cub, and thrust),
you must code all kernels from scratch.

## Part 1 (first week)

For the questions, cite sources you used. To submit, zip your repository to `~/lab5_2025_submission.zip`.

### Question 1.1 (5 points)
In this assignment, we will not be using tensor cores, because they require advanced data transfer layouts.
Instead, we will implement matrix-vector multiply with standard fused-multiply-add operators.
What is ratio of BF16 tensor core FLOPS to BF16 non-tensor core FLOPS on an A100-PCIE-40GB GPU?
Note: NVIDIA and AMD marketing both try to inflate their performance by measuring "sparse" tensor core operations, but nobody uses those.


The A100 has 312 TFLOPS when using tensor cores and 39 TFLOPS when using CUDA cores. Thus the ratio is 8:1.


### Question 1.2 (5 points)
What is the expected speedup of tensor cores vs non-tensor cores for matrix-vector multiplication on an A100-PCIE-40GB GPU?
Make an argument based on arithmetic intensity (FLOPS is not the whole story).
Assume the matrix and vector are read from off-chip memory.


For matrix vector multiplication the arithmetic intensity is 1 FLOP/Byte. This is because I only need to multiply each matrix element by one element in the vector. The full utilization of the cuda cores comes when the memory load speed per single byte matches the speed at which all operations can be done on that byte. For the A100, this requires about 50 FLOPs/Byte. Thus, GEMV is extremely memory bound and therefore there is pretty much no speedup for using tensor cores.


### Coding (80 points)
Implement GPU operators:
- ArgMax
- LayerNorm
- MatrixVectorMultiply
- RoPE
- SiLUMult

### Profiling (10 points)
Profile all your kernels with `ncu`, with input sizes matching what you'd expect for Qwen2 0.5B.
For each kernel, provide a screenshot and explain something interesting you noticed.
For example:
- Explain why your kernel is memory-bandwidth limited, latency/occupancy-limited, compute-limited, or limited by some other overhead.
- Explain why your kernel has suboptimal memory accesses, and a potential strategy to improve the kernel with expected performance increase.
- Explain which kernels are the most important to optimize, and which ones are less important.
- Explain how the performance would be different in another scenario (e.g. longer sequence length, larger model, increased batch size)
- Explain similarities across the kernels



ALL SCREENSHOTS IN profiling/profile_screenshots!!!

Argmax: This kernel is pretty lightweight compared to the matvec multiply, so it is not very important to optimize. On SOL it is hard to see if our kernel is optimal because we do not fill the grid (we will do this in parallel over the sequence dimension later). However, we can see that the kernel is memory bound by looking at the long scoreboard stalls. Each warp sits idle for 16.6 cycles on average waiting for memory from the L1 cache.

LayerNorm: This kernel is also lightweight compared to the matvec multiply. It is similar to the Argmax kernel in that it is memory bound. Each warp sits idle for 15.4 cycles on average waiting for cache misses. One inefficient part of layernorm is that naively it is two pass. If I were to optimize it futher I would do an online layernorm (one pass) called Welford's Algorithm.

MatVec Multiply: This kernel is highly efficient. It shows high memory and compute throughput. The bottleneck is the VRAM store speed. This is not really a problem that can be fixed. However, we could optimize further by loading 2 bf16 from VRAM at a time instead of 1.

RoPE: This kernel takes 4us which is extremely small compared to the Matvec multiply. Thus, it is not worth trying to optimize. Even still, there are no obvious optimizations besides increasing the gridsize when there is a larger task. One thing that is nice about this operation is that it is completely parallelizable, so it will not become a problem with scale. One thing that could be improved regardless is that we load real and imaginary parts 
separately instead of as a float2. Also, if we could just recompute these values if that's faster.

Silu: This kernel, similar to RoPE is lightweight compared to Matvec multiply and is completely parallel. Thus, it is not worth trying to hyperoptimize.



## Part 2 (second week)

To submit, zip your repository to `~/lab6_2025_submission.zip`.

### Question 2.1 (3 points)
List all the matrix-vector multiplies in a Qwen2 0.5B layer, including the (M, K) dimensions of the matrix.
(Do not include grouped-query attention).

**Answer:**

A Qwen2 0.5B layer includes the following matrix-vector multiplies (excluding grouped-query attention):

1. **Q, K, V projections**
   - `q_proj`: `(896, 896)`
   - `k_proj`: `(128, 896)`
   - `v_proj`: `(128, 896)`
2. **Output projection**
   - `o_proj`: `(896, 896)`
3. **MLP projections**
   - `gate_proj`: `(4864, 896)`
   - `up_proj`: `(4864, 896)`
   - `down_proj`: `(896, 4864)`

These come from the `Qwen2Config` values:
- `hidden_size = 896`
- `num_query_heads = 14`
- `num_kv_heads = 2`
- `head_size = 64`
- `intermediate_size = 4864`

So the projection output sizes are:
- `queries_size = 14 * 64 = 896`
- `keys_size = 2 * 64 = 128`
- `values_size = 2 * 64 = 128`

### Question 2.2 (2 points)
Treating each query head as a row of a matrix, what are the dimensions of the matrix-matrix multiply in a
Qwen2 0.5B layer grouped-query attention operation? Assume current sequence length is 1234 tokens.

**Answer:**

For Qwen2 0.5B, each query head has size `64`, and there are `14` query heads total.

Treating each query head as a row, the attention-score matmul is:

- Queries: `(14, 64)`
- Keys across the context: `(64, 1234)`
- Result: `(14, 1234)`

Because this is grouped-query attention, the `2` KV heads are shared across the `14` query heads, but the score matrix still has one row per query head.

### Question 2.3 (5 points)
Assuming off-chip memory bandwidth is the limiting factor, what is the theoretical minimum inference latency (in ms)
for Qwen2 0.5B on an A100-PCIE-40GB, with BF16 weights? Assume small sequence length (i.e. KV cache size is negligible).

**Answer:**

Using the actual Qwen2 0.5B config, the model has:

- Token embedding: `151936 * 896 = 136,134,656` parameters
- Each layer: `14,912,384` parameters
- Number of layers: `24`
- Final layernorm: `896` parameters

Total parameter count:

- `136,134,656 + 24 * 14,912,384 + 896 = 494,032,768`

BF16 uses 2 bytes per parameter, so total weight size is:

- `494,032,768 * 2 = 988,065,536` bytes ≈ `0.988 GB`

A100-PCIE-40GB memory bandwidth: ~1550 GB/s

Theoretical minimum time to read all weights = (model size) / (bandwidth)

= 0.988 GB / 1550 GB/s ≈ 0.000637 s ≈ 0.637 ms

So, the theoretical minimum inference latency is approximately **0.64 ms**.

### Question 2.4 (5 points)
Determine the sequence length at which the KV cache becomes non-negligible in terms of performance;
specifically, at what sequence length in Qwen2 0.5B would the KV cache become 10% the size of the model parameters?

**Answer:**

Model parameter size from above is `988,065,536` bytes, so 10% is:

- `98,806,553.6` bytes

Each token in the KV cache stores a key and value for every layer:

- `num_layers = 24`
- `num_kv_heads = 2`
- `head_size = 64`
- BF16 = `2` bytes per value

Bytes per token:

- `24 * 2 * 64 * 2 * 2 = 12,288` bytes

Sequence length where total KV cache is 10% of model weights:

- `98,806,553.6 / 12,288 ≈ 8,041`

So the KV cache becomes about 10% of the model size at roughly **8,000 tokens**.

### Coding (75 points)
Complete:
- GroupQueryAttention
  - Must use online numerically stable softmax, see section 3.1 of [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- Qwen2Layer
- Qwen2Model

You should not allocate or free any memory inside `Qwen2Model::forward`;
scratch space should be allocated only at model initialization, in constructors.

Test by running `./transformer`, and 100 tokens will be produced, matching the python reference implementation 

Once working, you can run `./transformer --interactive --max-seq-len 10000` to send messages with a chatbot interface.

### Profiling (10 points)

Once working, profile your implementation with:
```bash
ncu --set full --nvtx --nvtx-include last_token/ -c100 -o profile ./transformer
```
(may adjust -c parameter to number of kernels per layer)

How many microseconds per layer does your implementation take?
What is the slowest part of the layer and why?
Include screenshots of the something interesting you notice, and explain.

## Assignment notes

- Your kernels must fully occupy the GPU when possible (i.e. do not launch with only 1 block, launch with many).
- Kernels should have optimal memory access (coalesced gmem, and no smem bank conflicts) when possible.
- Always use CUDA streams when launching kernels, such as:
  - `my_kernel<<<grid_dim, block_dim, 0, stream>>>(my_arg);`
- Use the test cases and python reference to check the correctness of your implementation.

## Debugging tips

- Add print statements in the python implementation and equivalents in the CUDA implementation, such as:
```python
print('after q proj:', queries[0, 0])
```
corresponding to
```c++
std::cerr << "after q proj: " << static_cast<float>(*static_cast<__nv_bfloat16*>(queries->data)) << std::endl;
```
and check when they diverge.

- All GPU memory is allocated with `cudaMallocManaged`, which allows you to access the GPU memory from the CPU.
  Therefore, with plain GDB, we can run:
  - `CUDA_LAUNCH_BLOCKING=1 gdb ./transformer`
  - Set breakpoints
  - Save a tensor to disk: `dump binary memory /tmp/queries.bin queries->data ((uint8_t*)queries->data)+queries->size`
  - Load the tensor in python: `torch.from_file('/tmp/queries.bin',size=head_size*num_query_heads,dtype=torch.bfloat16).reshape(num_query_heads, head_size)`

## Test cases
Run all test cases with:
- `cd build`
- `cmake --build .`
- `ctest`

For tests that are failing, you can run them individually to see which elements were incorrect, for example:
- `cd build`
- `cmake --build .`
- `./silumulttest`

Failing output:
```
difference at index 0: GPU calculated 10.875, CPU calculated 108.5
difference at index 1: GPU calculated -5.65625, CPU calculated 0.332031
difference at index 2: GPU calculated 7.3125, CPU calculated -76.5
...
```

Also, note that passing all test cases does not mean you will get an A.
The test cases only check for correctness, not for performance.
If your kernels have needless suboptimal memory access, poor occupancy, or other performance issues,
the tests will still pass, but you will not get a good grade.

## Author
Sam Foxman 2025
