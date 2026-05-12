#include <torch/extension.h>

torch::Tensor vector_add_fp32_native(torch::Tensor x, torch::Tensor y);
torch::Tensor vector_add_fp32_vec4(torch::Tensor x, torch::Tensor y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add_fp32_native", torch::wrap_pybind_function(vector_add_fp32_native),
          "always scalar kernel");
    m.def("vector_add_fp32_vec4", torch::wrap_pybind_function(vector_add_fp32_vec4),
          "always float4 kernel (16B-aligned tensors required)");
}