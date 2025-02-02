import torch
from block_softmax_cuda import block_softmax

# Configuration
rows, cols = 128, 10240
x = torch.rand(rows, cols, device="cuda", dtype=torch.float32)
y_torch = torch.nn.functional.softmax(x, dim=1)
y_custom = torch.empty_like(x)

# Kernel call
block_softmax(x, y_custom)

# Verification
assert torch.allclose(y_custom, y_torch, atol=1e-5), "Block softmax mismatch"
print("Validation passed!")
