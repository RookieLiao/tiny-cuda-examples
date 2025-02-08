import torch
import sys

sys.path.append("../day07")
from matmul_cuda import naive_matmul, tiled_matmul
from torch.profiler import profile, record_function, ProfilerActivity


m = 1024
k = 2048
n = 4096

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("test_tiled_trace_" + str(prof.step_num) + ".json")


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


def profile_matmul():
    A = torch.rand(m, k, dtype=torch.float32, device="cuda")
    B = torch.rand(k, n, dtype=torch.float32, device="cuda")
    C = torch.empty(m, n, dtype=torch.float32, device="cuda")

    print("=============")
    print("Profiling naive_matmul")
    print("=============")

    with torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step
        schedule= torch.profiler.schedule(wait=1, warmup=10, active=20, repeat=1),
        on_trace_ready=trace_handler,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
    ) as p:
        for _ in range(40):
            # naive_matmul(A, B, C)
            tiled_matmul(A, B, C)
            # send a signal to the profiler that the next iteration has started
            p.step()

    with torch.autograd.profiler.profile(use_device="cuda") as prof:
        naive_matmul(A, B, C)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("=============")
    print("Profiling tiled_matmul")
    print("=============")

    with torch.autograd.profiler.profile(use_device="cuda") as prof:
        tiled_matmul(A, B, C)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    for _ in range(10):
        naive_matmul(A, B, C)

    for _ in range(10):
        tiled_matmul(A, B, C)

if __name__ == "__main__":
    profile_matmul()
