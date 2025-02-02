#include <torch/extension.h>

void block_softmax_kernel_launcher(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_softmax", &block_softmax_kernel_launcher,
        "Block-based Softmax kernel (CUDA)");
}
