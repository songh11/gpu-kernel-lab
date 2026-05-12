#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#define THREAD_PER_BLOCK 256

constexpr int cdiv(int n, int d) { return (n + d - 1) / d; }
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void vector_add_fp32_kernel_scalar(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = x[idx] + y[idx];
    }
}


__global__ void vector_add_fp32_kernel_vec4(
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ output,
    int N) {

    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = FLOAT4(x[idx]);
        float4 reg_b = FLOAT4(y[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(output[idx]) = reg_c;
    } else if (idx < N) {
        for (int i = 0; (idx + i) < N; i++) {
            output[idx + i] = x[idx + i] + y[idx + i];
        }
    }
}

static bool ptr16_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) % 16u) == 0u;
}


torch::Tensor vector_add_fp32_native(torch::Tensor x, torch::Tensor y) {
    const int length = static_cast<int>(x.numel());
    auto output = torch::empty_like(x);

    if (length == 0) {
        return output;
    }

    float* xp = x.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    float* op = output.data_ptr<float>();

    const int block_num = cdiv(length, THREAD_PER_BLOCK);
    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);
    vector_add_fp32_kernel_scalar<<<grid, block>>>(xp, yp, op, length);
    return output;
}

torch::Tensor vector_add_fp32_vec4(torch::Tensor x, torch::Tensor y) {
    const int length = static_cast<int>(x.numel());
    auto output = torch::empty_like(x);

    if (length == 0) {
        return output;
    }

    float* xp = x.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    float* op = output.data_ptr<float>();

    TORCH_CHECK(
        ptr16_aligned(xp) && ptr16_aligned(yp) && ptr16_aligned(op),
        "vector_add_fp32_vec4 requires 16-byte aligned x, y, output (use vector_add_fp32_native otherwise).");

    // idx = 4 * tid：每个线程最多覆盖 4 个 float，至少要 ceil(N/4) 个线程角色。
    const int vec_n = cdiv(length, 4);
    const int block_num = cdiv(vec_n, THREAD_PER_BLOCK);
    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);
    vector_add_fp32_kernel_vec4<<<grid, block>>>(xp, yp, op, length);
    return output;
}
