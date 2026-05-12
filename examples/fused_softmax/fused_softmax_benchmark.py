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

# seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# triton
from examples.fused_softmax.triton_kernels.fused_softmax_triton import fused_softmax_triton

# tilelang
from examples.fused_softmax.tilelang_kernels.fused_softmax_tilelang import fused_softmax_tilelang

# CUDA 注册
from torch.utils.cpp_extension import load
# Load the CUDA kernel as a python module
file_list = ['./cuda_kernels/fused_softmax.cu']
fused_softmax_cuda = load(name='fused_softmax_cuda', sources=['./cuda_kernels/main.cpp'] + file_list, extra_cuda_cflags=['-O2'])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100, 2)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'cuda_shm', 'cuda_warp', 'cuda_warp_vec4', 'tilelang'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "cuda(shm)", "cuda(warp)", "cuda(warp_vec4)", "TileLang"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    # 不要用「当前线程的 secondary CUDA stream」配合 triton.testing.do_bench：
    # do_bench 内部用 Triton 的 di.Event / di.synchronize()，往往不能可靠地覆盖
    # PyTorch 在非默认流上排队的 kernel；测到的 ms 会偏短 → GB/s 虚高，且 numel 变大时
    # 近似「GB/s 随 N 线性涨」的假曲线。教程里若用 stream，需保证与计时后端一致。
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: fused_softmax_triton(x))
    elif provider == 'cuda_shm':
        ms = triton.testing.do_bench(lambda: fused_softmax_cuda.fused_softmax_fp32_shm(x))
    elif provider == 'cuda_warp':
        ms = triton.testing.do_bench(lambda: fused_softmax_cuda.fused_softmax_fp32_warp(x))
    elif provider == 'cuda_warp_vec4':
        ms = triton.testing.do_bench(lambda: fused_softmax_cuda.fused_softmax_fp32_warp_vec4(x))
    elif provider == 'tilelang':
        ms = triton.testing.do_bench(lambda: fused_softmax_tilelang(x))
    else:
        raise ValueError(f"unknown provider: {provider}")
    gbps = lambda t_ms: 2 * x.numel() * x.element_size() * 1e-9 / (t_ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    # data compare
    x = torch.randn(4096, 512, device=DEVICE, dtype=DTYPE)
    output_torch = torch.softmax(x, axis=-1)
    output_triton = fused_softmax_triton(x)
    output_cuda_shm = fused_softmax_cuda.fused_softmax_fp32_shm(x)
    output_cuda_warp = fused_softmax_cuda.fused_softmax_fp32_warp(x)
    output_cuda_warp_vec4 = fused_softmax_cuda.fused_softmax_fp32_warp_vec4(x)
    output_tilelang = fused_softmax_tilelang(x)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    print(f'The maximum difference between torch and cuda shm is '
        f'{torch.max(torch.abs(output_torch - output_cuda_shm))}')
    print(f'The maximum difference between torch and cuda warp is '
        f'{torch.max(torch.abs(output_torch - output_cuda_warp))}')
    print(f'The maximum difference between torch and cuda warp vec4 is '
        f'{torch.max(torch.abs(output_torch - output_cuda_warp_vec4))}')
    print(f'The maximum difference between torch and tilelang is '
        f'{torch.max(torch.abs(output_torch - output_tilelang))}')
    benchmark.run(show_plots=True, print_data=True, save_path="./")