import torch

from softmax_cuda import softmax_kernel_launcher

x = torch.rand(2048, 8192, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)

softmax_kernel_launcher(x, y)

z = torch.nn.functional.softmax(x, dim=1)

assert torch.allclose(y, z, atol=1e-5), "Softmax kernel is not correct"
