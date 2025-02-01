import torch
from dtype_aware_softmax_cuda import dtype_aware_softmax

# Test parameters
rows = 1024
cols = 8192

# Create random input tensor
x = torch.rand(rows, cols, dtype=torch.float16, device="cuda")
y = torch.empty_like(x)

# Launch custom kernel
dtype_aware_softmax(x, y)

# Compute reference using PyTorch's softmax
z = torch.nn.functional.softmax(x, dim=1)

# Check correctness
assert torch.allclose(y, z, atol=1e-5), "Warp Softmax kernel is not correct"

print("Test passed!")
