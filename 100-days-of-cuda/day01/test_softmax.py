import torch

from softmax_cuda import softmax_kernel_launcher

rows = 1024
cols = 8192

x = torch.rand(rows, cols, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)

softmax_kernel_launcher(x, y)

z = torch.nn.functional.softmax(x, dim=1)

assert torch.allclose(y, z, atol=1e-5), "Softmax kernel is not correct"

steps = 10
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

# Warm up
for _ in range(10):
    _ = torch.nn.functional.softmax(x, dim=-1)
    # softmax_kernel_launcher(x, y)

for i in range(steps):
    start_events[i].record()
    _ = torch.nn.functional.softmax(x, dim=-1)
    # softmax_kernel_launcher(x, y)
    end_events[i].record()

torch.cuda.synchronize()
times = [start_events[i].elapsed_time(end_events[i]) for i in range(steps)]
print(f"Softmax computation time (average): {sum(times) / steps} ms")

