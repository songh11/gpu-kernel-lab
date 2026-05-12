import torch
import tilelang
import tilelang.language as T

_THREADS = 256


def _elementwise_add(N, threads: int, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as (b_x):
            for i in T.Parallel(threads):
                n = b_x * threads + i
                if n < N:
                    C[n] = A[n] + B[n]

    return main


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        program = _elementwise_add(T.dynamic("N"), _THREADS, T.float32)
        _kernel = tilelang.compile(
            program, out_idx=-1, target="cuda", execution_backend="cython"
        )
    return _kernel


def add_tilelang(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.device == y.device
    assert x.dtype == y.dtype
    kernel = _get_kernel()
    return kernel(x, y)
