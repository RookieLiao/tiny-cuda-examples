import torch
from deep_gemm.jit import generate, build

includes = ('"/workspace/tiny-cuda-examples/100-days-of-cuda/day12/smem_matmul.cuh"',)

template = """
dim3 block_size(256);
dim3 grid_size((n + 128 - 1) / 128,
               (m + 128 - 1) / 128);

sgemm_128x128x8<<<grid_size, block_size>>>(m, n, k, lhs, lda, rhs, ldb, out, ldc);
"""

arg_defs = (
    ("lhs", torch.float32),
    ("rhs", torch.float32),
    ("out", torch.float32),
    ("m", int),
    ("n", int),
    ("k", int),
    ("lda", int),
    ("ldb", int),
    ("ldc", int),
)


def tile_gemm(lhs, rhs):
    m, k = lhs.shape
    k_, n = rhs.shape

    out = torch.empty(m, n, device=lhs.device)
    lda = lhs.stride(0)
    ldb = rhs.stride(0)
    ldc = out.stride(0)

    # shape check
    assert k == k_

    code = generate(includes, arg_defs, template)
    runtime = build("tile_gemm", arg_defs, code)
    runtime(lhs, rhs, out, m, n, k, lda, ldb, ldc)

    return out


if __name__ == "__main__":
    a = torch.ones(128, 256, dtype=torch.float32, device="cuda")
    b = torch.randn(256, 128, dtype=torch.float32, device="cuda")
    c = tile_gemm(a, b)
    c_ref = torch.matmul(a, b)
    assert torch.allclose(c, c_ref, atol=1e-4, rtol=1e-4)
    print("Success!")
