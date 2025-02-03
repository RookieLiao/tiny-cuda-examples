#include <torch/extension.h>

void vec_softmax_kernel_launcher(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vec_softmax", &vec_softmax_kernel_launcher,
        "Vectorized FP16 Softmax kernel using half4");
}
