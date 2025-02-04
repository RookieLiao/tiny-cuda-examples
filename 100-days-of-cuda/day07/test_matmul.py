import torch
from matmul_cuda import naive_matmul, tiled_matmul

# Test parameters
m = 1024
k = 2048
n = 4096


def test_matmul():
    # Create random input tensors
    A = torch.rand(m, k, dtype=torch.float32, device="cuda")
    B = torch.rand(k, n, dtype=torch.float32, device="cuda")
    C = torch.empty(m, n, dtype=torch.float32, device="cuda")

    # Launch custom kernel
    # naive_matmul(A, B, C)
    tiled_matmul(A, B, C)

    # Compute reference using PyTorch's matmul
    C_ref = torch.matmul(A, B)

    # Check correctness
    assert torch.allclose(
        C, C_ref, atol=1e-5
    ), "Matrix multiplication kernel is not correct"
    print("Test passed!")


def benchmark_matmul():
    A = torch.rand(m, k, device="cuda", dtype=torch.float32)
    B = torch.rand(k, n, device="cuda", dtype=torch.float32)
    C = torch.empty(m, n, device="cuda", dtype=torch.float32)

    # Warm up
    for _ in range(10):
        naive_matmul(A, B, C)
        torch.matmul(A, B)
        tiled_matmul(A, B, C)

    # Benchmark
    steps = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(steps):
        naive_matmul(A, B, C)
    end.record()

    torch.cuda.synchronize()
    time_naive = start.elapsed_time(end) / steps

    start.record()
    for _ in range(steps):
        tiled_matmul(A, B, C)
    end.record()

    torch.cuda.synchronize()
    time_tiled = start.elapsed_time(end) / steps

    # Compare with PyTorch
    start.record()
    for _ in range(steps):
        torch.matmul(A, B)
    end.record()

    torch.cuda.synchronize()
    time_torch = start.elapsed_time(end) / steps

    print(f"Naive matmul time: {time_naive:.3f} ms")
    print(f"Tiled matmul time: {time_tiled:.3f} ms")
    print(f"PyTorch time: {time_torch:.3f} ms")


if __name__ == "__main__":
    # test_matmul()
    benchmark_matmul()
