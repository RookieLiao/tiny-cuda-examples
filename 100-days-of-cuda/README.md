# Project Progress and Tasks

### Mandatory and Optional Tasks
| Day   | Task Description                                                                                     |
|-------|-----------------------------------------------------------------------------------------------------|
| D10   | **Mandatory FA2-Forward**: Implement forward pass for FA2 (e.g., a custom neural network layer).    |
| D20   | **Mandatory FA2-Backwards**: Implement backward pass for FA2 (e.g., gradient computation).          |

---

### Project Progress by Day
| Day   | Files & Summaries         |
|-------|---------------------------|
| day1  | Naive Softmax kernel implementation - A basic CUDA kernel for computing softmax, with each thread handling one row |
| day2  | Warp Softmax kernel implementation - Optimized version using warp-level primitives for better performance (~26x speedup) |
| day3  | Dtype-aware Softmax kernel implementation - Use template to handle different data types (float16, bfloat16, float32) |
| day4  | Block Softmax kernel implementation - Optimized version for very large column size |
| day5  | Vectorized Softmax kernel implementation - Use half4 to vector read and write |
| day6  | Matrix Multiplication - Naive implementation |
| day7  | Matrix Multiplication - Tiled implementation |
| day8  | Matrix Multiplication - Profiling |
| day9  | Try to optimize tiled matmul according to this [blog](https://salykova.github.io/sgemm-gpu) and the result is not correct yet |

test it


#### How to load into PyTorch:
1. **Implement CUDA Kernel**:
   - Set up grid/block dimensions
   - Create kernel launch wrapper
   - Handle torch::Tensor inputs

2. **Create C++ Bindings**:
   - Include Torch headers
   - Declare PYBIND11_MODULE
   - Expose kernel launcher to Python

3. **Build Extension**:
   - Configure CUDAExtension with source files
   - Specify compilation flags
   - Use BuildExtension for compilation

4. **Test Integration**:
   - Allocate CUDA tensors
   - Call kernel launcher
   - Verify against PyTorch implementation

