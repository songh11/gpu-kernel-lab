import torch
import tilelang
import tilelang.language as T

# 与 CUDA warp 版类似：每行一个 block，列方向多线程分片；需 N 与 block 关系合法
_THREADS = 256


def _fused_softmax(M, N, threads: int, dtype):
    """对 (M, N) 在最后一维做 softmax；grid=M，每 block threads 个线程。"""
    @T.prim_func
    def main(
        x: T.Tensor((M, N), dtype),
        out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, threads=threads) as m:
            row_buf = T.alloc_fragment((threads,), dtype)
            row_max_s = T.alloc_fragment((1,), dtype)
            row_sum_s = T.alloc_fragment((1,), dtype)

            # 1) 每线程跨列步进求局部 max，再块内 reduce_max
            # 注意：勿用循环内反复绑定的 SSA 标量（如 local_max = ...），会触发
            # "Immutable variable ... outside defining region"；用 fragment 槽位做累加。
            for j in T.Parallel(threads):
                row_buf[j] = T.Cast(dtype, -1.0e30)
                for k in T.serial(T.ceildiv(N, threads)):
                    n = j + k * threads
                    row_buf[j] = T.if_then_else(
                        n < N,
                        T.max(row_buf[j], x[m, n]),
                        row_buf[j],
                    )
            T.reduce_max(row_buf, row_max_s, dim=0, clear=True)

            # 2) 局部 exp 之和，再 reduce_sum
            for j in T.Parallel(threads):
                rm = row_max_s[0]
                row_buf[j] = T.Cast(dtype, 0.0)
                for k in T.serial(T.ceildiv(N, threads)):
                    n = j + k * threads
                    row_buf[j] = row_buf[j] + T.if_then_else(
                        n < N,
                        T.exp(x[m, n] - rm),
                        T.Cast(dtype, 0.0),
                    )
            T.reduce_sum(row_buf, row_sum_s, dim=0, clear=True)

            # 3) 再算 exp 写回（与当前 CUDA/Triton 教程一致的双 exp 路径）
            for j in T.Parallel(threads):
                rm = row_max_s[0]
                inv = T.Cast(dtype, 1.0) / row_sum_s[0]
                for k in T.serial(T.ceildiv(N, threads)):
                    n = j + k * threads
                    if n < N:
                        out[m, n] = T.exp(x[m, n] - rm) * inv

    return main


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        M = T.dynamic("M", "int32")
        N = T.dynamic("N", "int32")
        program = _fused_softmax(M, N, _THREADS, T.float32)
        _kernel = tilelang.compile(
            program, out_idx=-1, target="cuda", execution_backend="cython"
        )
    return _kernel


def fused_softmax_tilelang(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    assert x.is_cuda
    assert x.dtype == torch.float32
    kernel = _get_kernel()
    return kernel(x)
