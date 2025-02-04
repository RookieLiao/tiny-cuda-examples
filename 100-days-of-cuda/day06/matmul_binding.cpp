#include <torch/extension.h>

void naive_matmul_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_matmul", &naive_matmul_launcher,
        "Naive matrix multiplication kernel launcher");
}
