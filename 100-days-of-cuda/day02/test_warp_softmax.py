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
