#include <torch/extension.h>

// Updated declaration
void dtype_aware_softmax_launcher(
    torch::Tensor input,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dtype_aware_softmax", &dtype_aware_softmax_launcher, "Dtype-aware Softmax launcher");
}
