import torch
from deep_gemm.jit import generate, build


def tile_gemm(lhs, rhs, out):
    m, k = lhs.shape
    k_, n = rhs.shape
    m_, n_ = out.shape

    # shape check
    assert m == m_ and n == n_ and k == k_


includes = ('"/workspace/tiny-cuda-examples/100-days-of-cuda/day10/tile_matmul.cuh"',)

template = """
dim3 block_size(TILE_SIZE, TILE_SIZE);
dim3 grid_size((n + block_size.x - 1) / block_size.x,
               (m + block_size.y - 1) / block_size.y);

tiledMatmulKernel<<<grid_size, block_size>>>(lhs, rhs, out, m, n, k);
"""

arg_defs = (
    ("lhs", torch.float32),
    ("rhs", torch.float32),
    ("out", torch.float32),
    ("m", int),
    ("n", int),
    ("k", int),
)


code = generate(includes, arg_defs, template)

runtime = build("tile_gemm", arg_defs, code)

print(runtime)
