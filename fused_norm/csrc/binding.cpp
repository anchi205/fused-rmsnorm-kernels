#include <torch/extension.h>

std::vector<torch::Tensor> fused_add_rmsnorm_fwd(torch::Tensor x, torch::Tensor residual,
                                                 torch::Tensor weight, double eps, bool naive);
std::vector<torch::Tensor> fused_add_rmsnorm_bwd(torch::Tensor dy, c10::optional<torch::Tensor> ds_out,
                                                 torch::Tensor s, torch::Tensor weight,
                                                 torch::Tensor rstd);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fused_add_rmsnorm_fwd, "Fused residual add + RMSNorm forward",
        py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("eps"),
        py::arg("naive") = false);
  m.def("bwd", &fused_add_rmsnorm_bwd, "Fused residual add + RMSNorm backward",
        py::arg("dy"), py::arg("ds_out"), py::arg("s"), py::arg("weight"), py::arg("rstd"));
}
