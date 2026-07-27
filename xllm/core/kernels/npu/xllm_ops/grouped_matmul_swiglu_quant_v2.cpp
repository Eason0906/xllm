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

#include <cstdlib>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

// Allow disabling the fused v2 path at runtime for A/B comparison against the
// separate group_gemm + dequant_swiglu_quant fallback. The fused op is ON by
// default when its symbol is deployed (this is the optimization we want).
bool fused_v2_disabled_by_env() {
  const char* value = std::getenv("XLLM_DISABLE_GMM_SWIGLU_QUANT_V2");
  if (value == nullptr) {
    return false;
  }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
         value[0] == 'y' || value[0] == 'Y';
}

}  // namespace

bool is_grouped_matmul_swiglu_quant_v2_available() {
  static const bool symbols_present =
      aclnn::detail::get_op_api_func_addr(
          "aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr(
          "aclnnGroupedMatmulSwigluQuantWeightNZ") != nullptr;
  return symbols_present && !fused_v2_disabled_by_env();
}

// Fused routed-expert grouped matmul + dequant + SwiGLU + dynamic int8 quant.
//
// Contract of aclnnGroupedMatmulSwigluQuantWeightNZ (int8 A8W8 pertoken path):
//   x            int8  (M, K)   ND
//   weight       int8  (E, K, 2N) FRACTAL_NZ  (caller must NZ-cast)
//   weight_scale fp32  (E, 2N)  per-channel dequant scale
//   x_scale      fp32  (M,)     per-token dequant scale
//   group_list   int64 (E,)     PER-EXPERT token counts (groupListType=0);
//                               the op consumes counts natively, no cumsum.
//   -> y         int8  (M, N),  N = 2N / 2
//      y_scale   fp32  (M,)     per-token quant scale
// bias / offset / y_offset are unused for this path (empty tensors).
std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_v2(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& x_scale,
    const at::Tensor& group_list) {
  const int64_t m = x.size(0);
  // Output width N = (2N) / 2, derived from weight_scale last dim (= 2N), a
  // property of the weight (not of x), so it stays correct if K != 2N.
  const int64_t two_n = weight_scale.size(weight_scale.dim() - 1);
  const int64_t n = two_n / 2;

  at::Tensor bias;
  at::Tensor offset;
  at::Tensor y_offset;
  at::Tensor y = at::empty({m, n}, x.options().dtype(at::kChar));
  at::Tensor y_scale = at::empty({m}, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantWeightNZ,
               x, weight, bias, offset, weight_scale, x_scale, group_list,
               y, y_scale, y_offset);

  return std::make_tuple(y, y_scale);
}

}  // namespace xllm::kernel::npu
