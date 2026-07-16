/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

bool is_moe_grouped_matmul_swiglu_quant_available() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnMoeGroupedMatmulSwigluQuantGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr(
          "aclnnMoeGroupedMatmulSwigluQuant") != nullptr;
  return is_available;
}

std::tuple<at::Tensor, at::Tensor> moe_grouped_matmul_swiglu_quant(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& x_scale,
    const at::Tensor& group_list) {
  at::SmallVector<int64_t, op_infer::SIZE> y_size;
  at::SmallVector<int64_t, op_infer::SIZE> scale_size;
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    y_size.push_back(x.size(i));
    scale_size.push_back(x.size(i));
  }
  // The swiglu output width equals the MoE intermediate size I, which is a
  // property of w13 (its output channels are 2*I), NOT of the activation x.
  // Derive it from weight_scale's last dim (per-channel scale over 2*I) so the
  // output shape stays correct even when hidden_size != 2*I. Fall back to the
  // x-based estimate only if weight_scale is not the expected [.., 2*I] shape.
  int64_t swiglu_out_width = x.size(x.dim() - 1) / 2;
  if (weight_scale.dim() >= 1) {
    const int64_t scale_last = weight_scale.size(weight_scale.dim() - 1);
    if (scale_last > 0 && scale_last % 2 == 0) {
      swiglu_out_width = scale_last / 2;
    }
  }
  y_size.push_back(swiglu_out_width);

  at::Tensor y = at::empty(y_size, x.options().dtype(at::kChar));
  at::Tensor y_scale = at::empty(scale_size, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnMoeGroupedMatmulSwigluQuant,
               x, weight, weight_scale, x_scale, group_list,
               y, y_scale);

  return std::make_tuple(y, y_scale);
}

}  // namespace xllm::kernel::npu
