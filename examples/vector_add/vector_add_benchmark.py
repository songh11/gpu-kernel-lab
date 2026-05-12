from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

import torch
import triton
import triton.language as tl

# 公共参数
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
DTYPE = torch.float32

# triton
from examples.vector_add.triton_kernels.vector_add_triton import add_triton

# tilelang
from examples.vector_add.tilelang_kernels.vector_add_tilelang import add_tilelang

# CUDA 注册
from torch.utils.cpp_extension import load
# Load the CUDA kernel as a python module
file_list = ['./cuda_kernels/vector_add.cu']
add_cuda = load(name='add_cuda', sources=['./cuda_kernels/main.cpp'] + file_list, extra_cuda_cflags=['-O2'])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # x 轴名称
        x_vals=[2**i for i in range(12, 28, 1)],  # x 的值
        line_arg='provider',  # 参数名称，其值对应于绘图中的不同线条。
        line_vals=['triton', 'torch', 'cuda_native', 'cuda_vec4', 'tilelang'],
        line_names=['Triton', 'Torch', 'cuda(native)', 'cuda(float4)', 'TileLang'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-'), ('purple', '-')],
        ylabel='GB/s',  # y 轴标签名称。
        plot_name='vector-add-performance',  # 绘图名称。也用作保存绘图的文件名。
        args={},  # 不在 `x_names` 和 `y_name` 中的函数参数值。
    ))
def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=DTYPE)
    y = torch.randn(size, device=DEVICE, dtype=DTYPE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y, warmup=25, rep=100, quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x, y), warmup=25, rep=100, quantiles=quantiles)
    elif provider == "cuda_native":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_cuda.vector_add_fp32_native(x, y), warmup=25, rep=100, quantiles=quantiles)
    elif provider == "cuda_vec4":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_cuda.vector_add_fp32_vec4(x, y), warmup=25, rep=100, quantiles=quantiles)
    elif provider == "tilelang":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_tilelang(x, y), warmup=25, rep=100, quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)

if __name__ == "__main__":
    # data compare
    x = torch.randn(512, device=DEVICE, dtype=DTYPE)
    y = torch.randn(512, device=DEVICE, dtype=DTYPE)
    output_torch = x + y
    output_triton = add_triton(x, y)
    output_cuda_native = add_cuda.vector_add_fp32_native(x, y)
    output_cuda_vec4 = add_cuda.vector_add_fp32_vec4(x, y)
    output_tilelang = add_tilelang(x, y)
    print(output_torch[:10])
    print(output_triton[:10])
    print(output_cuda_native[:10])
    print(output_cuda_vec4[:10])
    print(output_tilelang[:10])
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    print(f'The maximum difference between torch and cuda native is '
        f'{torch.max(torch.abs(output_torch - output_cuda_native))}')
    print(f'The maximum difference between torch and tilelang is '
        f'{torch.max(torch.abs(output_torch - output_tilelang))}')

    benchmark.run(show_plots=True, print_data=True, save_path="./")