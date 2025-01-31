#include <torch/extension.h>

void warp_softmax_launcher(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("warp_softmax_launcher", &warp_softmax_launcher,
        "Warp-level Softmax kernel launcher");
}
