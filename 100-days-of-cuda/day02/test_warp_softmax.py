import torch
from warp_softmax_cuda import warp_softmax_launcher

# Test parameters
rows = 1024
cols = 8192

# Create random input tensor
x = torch.rand(rows, cols, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)

# Launch custom kernel
warp_softmax_launcher(x, y)

# Compute reference using PyTorch's softmax
z = torch.nn.functional.softmax(x, dim=1)

# Check correctness
assert torch.allclose(y, z, atol=1e-5), "Warp Softmax kernel is not correct"

print("Test passed!")


steps = 10
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

# Initialize the matrix
x = torch.rand(rows, cols, device="cuda", dtype=torch.float32)

# Warm up
for _ in range(10):
    # _ = torch.nn.functional.softmax(x, dim=-1)
    warp_softmax_launcher(x, y)

for i in range(steps):
    start_events[i].record()
    # _ = torch.nn.functional.softmax(x, dim=-1)
    warp_softmax_launcher(x, y)
    end_events[i].record()

torch.cuda.synchronize()
times = [start_events[i].elapsed_time(end_events[i]) for i in range(steps)]
print(f"Softmax computation time (average): {sum(times) / steps} ms")
