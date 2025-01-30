#include <torch/extension.h>

void softmax_kernel_launcher(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_kernel_launcher", &softmax_kernel_launcher,
        "Naive Safe Softmax kernel");
}
