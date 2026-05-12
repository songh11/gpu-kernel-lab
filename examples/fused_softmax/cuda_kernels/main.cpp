#include <torch/extension.h>

torch::Tensor fused_softmax_fp32_native(torch::Tensor x);
torch::Tensor fused_softmax_fp32_shm(torch::Tensor x);
torch::Tensor fused_softmax_fp32_warp(torch::Tensor x);
torch::Tensor fused_softmax_fp32_warp_vec4(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_softmax_fp32_native", torch::wrap_pybind_function(fused_softmax_fp32_native),
          "fused softmax native kernel");
    m.def("fused_softmax_fp32_shm", torch::wrap_pybind_function(fused_softmax_fp32_shm),
          "fused softmax shared memory kernel");
    m.def("fused_softmax_fp32_warp", torch::wrap_pybind_function(fused_softmax_fp32_warp),
          "fused softmax warp kernel");
    m.def("fused_softmax_fp32_warp_vec4", torch::wrap_pybind_function(fused_softmax_fp32_warp_vec4),
          "fused softmax warp kernel with float4 loads where aligned");
}