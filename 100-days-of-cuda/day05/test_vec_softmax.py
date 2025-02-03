import torch
from vec_softmax_cuda import vec_softmax

# Test parameters
rows = 10
cols = 1024  # Must be divisible by 4 for half4

# Create input tensor (FP16)
x = torch.rand(rows, cols, dtype=torch.float16, device="cuda")
y_custom = torch.empty_like(x)

# Run custom kernel
vec_softmax(x, y_custom)

# Reference implementation
y_ref = torch.nn.functional.softmax(x, dim=-1)

# Check results (using relaxed tolerance for FP16)
assert torch.allclose(y_custom, y_ref, atol=1e-3), "Vectorized softmax kernel failed"
