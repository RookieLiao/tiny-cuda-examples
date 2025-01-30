# Project Progress and Tasks

### Mandatory and Optional Tasks
| Day   | Task Description                                                                                     |
|-------|-----------------------------------------------------------------------------------------------------|
| D10   | **Mandatory FA2-Forward**: Implement forward pass for FA2 (e.g., a custom neural network layer).    |
| D20   | **Mandatory FA2-Backwards**: Implement backward pass for FA2 (e.g., gradient computation).          |

---

### Project Progress by Day
| Day   | Files & Summaries                                                                                                                                                                                                                          |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| day1  | **softmax.cu**: Softmax kernel implementation.<br>**softmax_binding.cpp**: Torch bindings for the softmax kernel.<br>**test_softmax.py**: Tests the softmax kernel against PyTorch. |


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

