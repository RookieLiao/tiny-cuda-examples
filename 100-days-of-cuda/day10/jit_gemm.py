import torch
from deep_gemm.jit import generate, build

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


def tile_gemm(lhs, rhs):
    m, k = lhs.shape
    k_, n = rhs.shape

    out = torch.empty(m, n, device=lhs.device)
    # shape check
    assert k == k_

    code = generate(includes, arg_defs, template)
    runtime = build("tile_gemm", arg_defs, code)

    runtime(lhs, rhs, out, m, n, k)

    return out


if __name__ == "__main__":
    a = torch.randn(32, 256, dtype=torch.float32, device="cuda")
    b = torch.randn(256, 128, dtype=torch.float32, device="cuda")
    c = tile_gemm(a, b)
    c_ref = torch.matmul(a, b)
    assert torch.allclose(c, c_ref)
