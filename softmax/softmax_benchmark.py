import torch
import numpy as np

import subprocess

DEVICE_INDEX = torch.cuda.current_device()


def get_supported_clocks():
    result = subprocess.run(
        "nvidia-smi -q -d SUPPORTED_CLOCKS | grep -E 'Graphics.*MHz|Memory.*MHz'",
        shell=True,
        capture_output=True,
        text=True,
    )
    # Extract clock values using more precise parsing
    clocks = []
    for line in result.stdout.split("\n"):
        if line.strip():
            # Example line: "Graphics                  : 2100 MHz"
            parts = line.split(":")
            if len(parts) >= 2:
                clock_value = parts[1].strip().split()[0]
                clocks.append(int(clock_value))
    return sorted(list(set(clocks)))  # Remove duplicates and sort


def set_clock_speed():
    """Set to a sustainable clock speed for consistent measurements"""
    supported_clocks = get_supported_clocks()

    # Choose median clock speed from supported values for stable benchmarking
    CLOCK_SPEED = int(np.median(supported_clocks))

    subprocess.run(f"nvidia-smi -pm ENABLED -i {DEVICE_INDEX}", shell=True)
    subprocess.run(f"nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE_INDEX}", shell=True)

    # Verify stability
    torch.cuda.empty_cache()
    test_tensor = torch.randn(10000, 10000, device="cuda")
    for _ in range(100):  # Stress test
        _ = test_tensor @ test_tensor
    current_clock = torch.cuda.clock_rate(DEVICE_INDEX)
    assert abs(current_clock - CLOCK_SPEED) < 50, (
        f"Clock speed not stable when setting {CLOCK_SPEED} but got {current_clock}"
    )


def reset_clock_speed():
    """Reset GPU clock speed to default values."""
    subprocess.run(f"nvidia-smi -pm DISABLED -i {DEVICE_INDEX}", shell=True)
    subprocess.run(f"nvidia-smi -rgc -i {DEVICE_INDEX}", shell=True)


def get_l2_cache_bytes():
    """Return L2 cache size in bytes based on GPU model"""
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()

    # Common GPU L2 cache sizes (in MB)
    l2_size_mb = {
        "a100": 40,  # A100/A800
        "h100": 50,  # H100/H800
        "rtx 4090": 72,  # RTX 4090
        "rtx 3090": 6,  # RTX 3090/Ti
        "a6000": 6,  # RTX A6000
        "v100": 6,  # V100
        "titan": 5,  # Titan series
    }

    for key in l2_size_mb:
        if key in device_name:
            return l2_size_mb[key] * (1024**2)

    # Fallback for unknown devices: use device properties or default to 40MB
    try:
        return torch.cuda.get_device_properties(0).l2_cache_size
    except AttributeError:
        return 40 * (1024**2)  # Conservative default


# Update the cache allocation line
x = torch.empty(int(get_l2_cache_bytes()), dtype=torch.int8, device="cuda")


def flush_cache():
    x.zero_()


steps = 10
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

# Initialize the matrix
matrix = torch.randn(1024, 32768, device="cuda", dtype=torch.float32)

# Warm up
for _ in range(10):
    _ = torch.nn.functional.softmax(matrix, dim=-1)

kernel_count = 10

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for _ in range(steps * kernel_count):
        _ = torch.nn.functional.softmax(matrix, dim=-1)


for i in range(steps):
    flush_cache()
    start_events[i].record()
    graph.replay()
    end_events[i].record()

torch.cuda.synchronize()
times = [start_events[i].elapsed_time(end_events[i]) for i in range(steps)]
print(f"Softmax computation time (average): {sum(times) / (steps * kernel_count)} ms")

reset_clock_speed()
