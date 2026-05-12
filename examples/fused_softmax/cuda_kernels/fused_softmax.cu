#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#define THREAD_PER_BLOCK 256

constexpr int cdiv(int n, int d) { return (n + d - 1) / d; }

__device__ __forceinline__ bool ptr_is_16b_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

__device__ __forceinline__ float horiz_max4(float4 v) {
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

__device__ __forceinline__ float horiz_sum4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

// 读 4 个 float：对齐则用 float4，否则标量（避免未对齐 vector load 的 UB）
__device__ __forceinline__ float4 load4(const float* p, int n) {
    float4 v{0.f, 0.f, 0.f, 0.f};
    if (n >= 4 && ptr_is_16b_aligned(p)) {
        return *reinterpret_cast<const float4*>(p);
    }
    if (n >= 1) v.x = p[0];
    if (n >= 2) v.y = p[1];
    if (n >= 3) v.z = p[2];
    if (n >= 4) v.w = p[3];
    return v;
}

__device__ __forceinline__ void store4(float* p, float4 v, int n) {
    if (n >= 4 && ptr_is_16b_aligned(p)) {
        *reinterpret_cast<float4*>(p) = v;
        return;
    }
    if (n >= 1) p[0] = v.x;
    if (n >= 2) p[1] = v.y;
    if (n >= 3) p[2] = v.z;
    if (n >= 4) p[3] = v.w;
}

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_softmax_fp32_kernel_warp_vec4(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols)
{
    extern __shared__ float shared[];
    const int idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int num_warps = block_size / 32;
    const float* row_in = x + static_cast<int64_t>(idx) * cols;
    float* row_out = output + static_cast<int64_t>(idx) * cols;

    // 按「连续 4 元组」轮转分配 chunk：chunk c 由线程 (c % block_size) 处理，合并访问友好
    const int nvec = (cols + 3) / 4;

    // 1) 行 max
    float max_val = -INFINITY;
    for (int c = tid; c < nvec; c += block_size) {
        const int base = c * 4;
        const int rem = cols - base;
        const int n = rem < 4 ? rem : 4;
        const float4 vx = load4(row_in + base, n);
        if (n >= 4) {
            max_val = fmaxf(max_val, horiz_max4(vx));
        } else {
            if (n >= 1) max_val = fmaxf(max_val, vx.x);
            if (n >= 2) max_val = fmaxf(max_val, vx.y);
            if (n >= 3) max_val = fmaxf(max_val, vx.z);
        }
    }
    shared[tid] = max_val;
    __syncthreads();
    max_val = warpReduceMax(max_val);
    if ((tid % 32) == 0) {
        shared[tid / 32] = max_val;
    }
    __syncthreads();
    if (tid < 32) {
        max_val = (tid < num_warps) ? shared[tid] : -INFINITY;
        max_val = warpReduceMax(max_val);
        if (tid == 0) {
            shared[0] = max_val;
        }
    }
    __syncthreads();
    const float row_max = shared[0];

    // 2) 寄存器里累加 exp（双 exp 路径里第一次）
    float partial_exp_sum = 0.0f;
    for (int c = tid; c < nvec; c += block_size) {
        const int base = c * 4;
        const int rem = cols - base;
        const int n = rem < 4 ? rem : 4;
        const float4 vx = load4(row_in + base, n);
        if (n >= 4) {
            float4 e{expf(vx.x - row_max), expf(vx.y - row_max), expf(vx.z - row_max), expf(vx.w - row_max)};
            partial_exp_sum += horiz_sum4(e);
        } else {
            if (n >= 1) partial_exp_sum += expf(vx.x - row_max);
            if (n >= 2) partial_exp_sum += expf(vx.y - row_max);
            if (n >= 3) partial_exp_sum += expf(vx.z - row_max);
        }
    }
    shared[tid] = partial_exp_sum;
    __syncthreads();
    partial_exp_sum = warpReduceSum(partial_exp_sum);
    if ((tid % 32) == 0) {
        shared[tid / 32] = partial_exp_sum;
    }
    __syncthreads();
    if (tid < 32) {
        partial_exp_sum = (tid < num_warps) ? shared[tid] : 0.0f;
        partial_exp_sum = warpReduceSum(partial_exp_sum);
        if (tid == 0) {
            shared[0] = partial_exp_sum;
        }
    }
    __syncthreads();
    const float row_sum = shared[0];

    // 3) 再 exp 一次并写回（与标量 warp 版相同策略）
    const float inv_sum = 1.f / row_sum;
    for (int c = tid; c < nvec; c += block_size) {
        const int base = c * 4;
        const int rem = cols - base;
        const int n = rem < 4 ? rem : 4;
        const float4 vx = load4(row_in + base, n);
        float4 vo{0.f, 0.f, 0.f, 0.f};
        if (n >= 4) {
            vo.x = expf(vx.x - row_max) * inv_sum;
            vo.y = expf(vx.y - row_max) * inv_sum;
            vo.z = expf(vx.z - row_max) * inv_sum;
            vo.w = expf(vx.w - row_max) * inv_sum;
        } else {
            if (n >= 1) vo.x = expf(vx.x - row_max) * inv_sum;
            if (n >= 2) vo.y = expf(vx.y - row_max) * inv_sum;
            if (n >= 3) vo.z = expf(vx.z - row_max) * inv_sum;
        }
        store4(row_out + base, vo, n);
    }
}

__global__ void fused_softmax_fp32_kernel_warp(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols)
{
    extern __shared__ float shared[];
    const int idx = blockIdx.x;     // range [0, N]
    const int tid = threadIdx.x;    // range [0, THREAD_PER_BLOCK)
    const int block_size = blockDim.x;
    const int num_warps = block_size / 32;
    const float* inp = x + idx * cols;
    // 1 reduce
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += block_size) {
        max_val = fmaxf(max_val, inp[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    // warp shuffle 只在 32 lane 内有效；256 线程要先 warp 内 max，再把每个 warp 的结果写到 shared，二次归约
    max_val = warpReduceMax(max_val);
    if ((tid % 32) == 0) {
        shared[tid / 32] = max_val;
    }
    __syncthreads();
    // 必须凑满一个 warp 参与 shuffle；lane 8..31 用 -inf 占位
    if (tid < 32) {
        max_val = (tid < num_warps) ? shared[tid] : -INFINITY;
        max_val = warpReduceMax(max_val);
        if (tid == 0) {
            shared[0] = max_val;
        }
    }
    __syncthreads();
    const float row_max = shared[0];

    // 2 各线程在寄存器里累加自己负责列上的 exp，不再先写全局 output
    float partial_exp_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        partial_exp_sum += expf(inp[i] - row_max);
    }
    shared[tid] = partial_exp_sum;
    __syncthreads();
    partial_exp_sum = warpReduceSum(partial_exp_sum);
    if ((tid % 32) == 0) {
        shared[tid / 32] = partial_exp_sum;
    }
    __syncthreads();
    if (tid < 32) {
        partial_exp_sum = (tid < num_warps) ? shared[tid] : 0.0f;
        partial_exp_sum = warpReduceSum(partial_exp_sum);
        if (tid == 0) {
            shared[0] = partial_exp_sum;
        }
    }
    __syncthreads();
    const float row_sum = shared[0];

    // 3 再算一遍 exp 并除以行和，一次性写 output（多一次 exp，少一轮全局读写）
    float inv_sum = 1.f / row_sum;
    for (int i = tid; i < cols; i += block_size) {
        output[idx * cols + i] = expf(inp[i] - row_max) * inv_sum;
    }
}

__global__ void fused_softmax_fp32_kernel_shm(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols)
{
    extern __shared__ float shared[];
    const int idx = blockIdx.x;     // range [0, N]
    const int tid = threadIdx.x;    // range [0, THREAD_PER_BLOCK)
    const int block_size = blockDim.x;
    const float* inp = x + idx * cols;
    // 1 reduce
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += block_size) {
        max_val = fmaxf(max_val, inp[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    // 2 各线程在寄存器里累加自己负责列上的 exp，不再先写全局 output
    float partial_exp_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        partial_exp_sum += expf(inp[i] - row_max);
    }
    shared[tid] = partial_exp_sum;
    __syncthreads();
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    const float row_sum = shared[0];
    __syncthreads();

    // 3 再算一遍 exp 并除以行和，一次性写 output（多一次 exp，少一轮全局读写）
    float inv_sum = 1.f / row_sum;
    for (int i = tid; i < cols; i += block_size) {
        output[idx * cols + i] = expf(inp[i] - row_max) * inv_sum;
    }
}

__global__ void fused_softmax_fp32_kernel_native(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols)
{
    const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id < rows) {
        const float* inp_row = x + row_id * cols;
        float* out_row = output + row_id * cols;
        float max_val = -INFINITY;
        for (int i = 0; i < cols; i++) {
            max_val = fmaxf(max_val, inp_row[i]);
        }
        float sum_val = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum_val += expf(inp_row[i] - max_val);
        }
        for (int i = 0; i < cols; i++) {
            out_row[i] = expf(inp_row[i] - max_val) / sum_val;
        }
    }
}

torch::Tensor fused_softmax_fp32_native(torch::Tensor x) {
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    auto output = torch::empty_like(x);

    float* xp = x.data_ptr<float>();
    float* op = output.data_ptr<float>();

    const int block_num = cdiv(rows, THREAD_PER_BLOCK);
    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);
    fused_softmax_fp32_kernel_native<<<grid, block>>>(xp, op, rows, cols);
    return output;
}

torch::Tensor fused_softmax_fp32_shm(torch::Tensor x) {
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    auto output = torch::empty_like(x);

    float* xp = x.data_ptr<float>();
    float* op = output.data_ptr<float>();

    const int block_num = rows;
    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);
    // extern __shared__ 必须在 launch 第三参指定字节数，否则 shared 长度为 0，会越界/未定义行为
    const size_t shm_bytes = static_cast<size_t>(THREAD_PER_BLOCK) * sizeof(float);
    fused_softmax_fp32_kernel_shm<<<grid, block, shm_bytes>>>(xp, op, rows, cols);
    return output;
}

torch::Tensor fused_softmax_fp32_warp(torch::Tensor x) {
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    auto output = torch::empty_like(x);

    float* xp = x.data_ptr<float>();
    float* op = output.data_ptr<float>();

    const int block_num = rows;
    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);
    // extern __shared__ 必须在 launch 第三参指定字节数，否则 shared 长度为 0，会越界/未定义行为
    const size_t shm_bytes = static_cast<size_t>(THREAD_PER_BLOCK) * sizeof(float);
    fused_softmax_fp32_kernel_warp<<<grid, block, shm_bytes>>>(xp, op, rows, cols);
    return output;
}

torch::Tensor fused_softmax_fp32_warp_vec4(torch::Tensor x) {
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    auto output = torch::empty_like(x);

    float* xp = x.data_ptr<float>();
    float* op = output.data_ptr<float>();

    const int block_num = rows;
    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);
    const size_t shm_bytes = static_cast<size_t>(THREAD_PER_BLOCK) * sizeof(float);
    fused_softmax_fp32_kernel_warp_vec4<<<grid, block, shm_bytes>>>(xp, op, rows, cols);
    return output;
}
