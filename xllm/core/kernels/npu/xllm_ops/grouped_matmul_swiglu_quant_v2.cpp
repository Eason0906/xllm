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
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

// Allow disabling the fused path at runtime for A/B comparison against the
// separate group_gemm + dequant_swiglu_quant fallback. ON by default when the
// op symbol is deployed (this is the optimization we want).
bool fused_v2_disabled_by_env() {
  const char* value = std::getenv("XLLM_DISABLE_GMM_SWIGLU_QUANT_V2");
  if (value == nullptr) {
    return false;
  }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
         value[0] == 'y' || value[0] == 'Y';
}

// ---------------------------------------------------------------------------
// aclnnGroupedMatmulSwigluQuantWeightNzV2 attribute encodings.
//
// V2 (unlike the type-fixed-cumulative V1 aclnnGroupedMatmulSwigluQuantWeightNZ)
// exposes an explicit groupListType, so we feed per-expert COUNTS directly and
// drop the host-side cumsum entirely. Header on device declares:
//   GetWorkspaceSize(x, weight[list], weightScale[list], weightAssistMatrix[list],
//                    bias, xScale, smoothScale, groupList,
//                    dequantMode, dequantDtype, quantMode, groupListType,
//                    tuningConfig, swigluLimit, output, outputScale, ws, exec)
//
// Enum values verified against the custom_xllm_math vendor package:
//   - op_proto/inc/grouped_matmul_swiglu_quant_v2_proto.h REG_OP defaults:
//       dequant_mode=0, dequant_dtype=0, quant_mode=0, group_list_type=0
//   - ascendc/.../grouped_matmul_swiglu_quant_v2_utils.h selects the A8W8
//       template purely from input dtypes (x=int8, weight=int8) at compile
//       time; the device kernel does NOT branch on dequantMode/dequantDtype/
//       quantMode (grep found no references), so int8 dynamic per-token quant
//       (QUANT_SCALE_INT8=127.0f) is unconditional on this path.
// => the three dequant/quant attrs stay at the op's official default 0.
// groupListType is the ONE value we deliberately override: 1 = per-expert
// COUNTS, which is exactly why we moved off the type-fixed-cumulative V1.
constexpr int64_t kV2GroupListType = 1;  // 1 = per-expert counts (no cumsum)
constexpr int64_t kV2QuantMode = 0;      // proto default; A8W8 kernel ignores
constexpr int64_t kV2DequantMode = 0;    // proto default; A8W8 kernel ignores
constexpr int64_t kV2DequantDtype = 0;   // proto default; A8W8 kernel ignores

}  // namespace

bool is_grouped_matmul_swiglu_quant_v2_available() {
  static const bool symbols_present =
      aclnn::detail::get_op_api_func_addr(
          "aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize") !=
          nullptr &&
      aclnn::detail::get_op_api_func_addr(
          "aclnnGroupedMatmulSwigluQuantWeightNzV2") != nullptr;
  return symbols_present && !fused_v2_disabled_by_env();
}

// Fused routed-expert grouped matmul + dequant + SwiGLU + dynamic int8 quant,
// via aclnnGroupedMatmulSwigluQuantWeightNzV2.
//
// Contract (int8 A8W8 per-token path):
//   x            int8  (M, K)     ND
//   weight       int8  (E, K, 2N) FRACTAL_NZ  (caller NZ-casts once)
//   weight_scale fp32  (E, 2N)    per-channel dequant scale
//   x_scale      fp32  (M,)       per-token dequant scale
//   group_list   int64 (E,)       per-expert COUNTS (groupListType=1, no cumsum)
//   swiglu_limit double            SwiGLU clamp limit
//   -> y         int8  (M, N),    N = 2N / 2
//      y_scale   fp32  (M,)       per-token quant scale
//
// weight / weight_scale are passed as single-element tensor lists (the op takes
// aclTensorList). weightAssistMatrix / bias / smoothScale / tuningConfig are
// unused on this path (empty -> nullptr).
std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_v2(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& x_scale,
    const at::Tensor& group_list,
    double swiglu_limit) {
  const int64_t m = x.size(0);
  // Output width N = (2N)/2, derived from weight_scale last dim (= 2N), a
  // property of the weight (not of x), so it stays correct if K != 2N.
  const int64_t two_n = weight_scale.size(weight_scale.dim() - 1);
  const int64_t n = two_n / 2;

  std::vector<at::Tensor> weight_vec = {weight};
  std::vector<at::Tensor> weight_scale_vec = {weight_scale};
  std::vector<at::Tensor> weight_assist_vec;  // empty: no aux matrix (int8)
  at::TensorList weight_list(weight_vec);
  at::TensorList weight_scale_list(weight_scale_vec);
  at::TensorList weight_assist_list(weight_assist_vec);

  at::Tensor bias;          // unused -> nullptr
  at::Tensor smooth_scale;  // unused -> nullptr
  c10::optional<at::IntArrayRef> tuning_config;  // nullopt -> nullptr

  at::Tensor y = at::empty({m, n}, x.options().dtype(at::kChar));
  at::Tensor y_scale = at::empty({m}, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantWeightNzV2,
               x, weight_list, weight_scale_list, weight_assist_list,
               bias, x_scale, smooth_scale, group_list,
               kV2DequantMode, kV2DequantDtype, kV2QuantMode, kV2GroupListType,
               tuning_config, swiglu_limit,
               y, y_scale);

  return std::make_tuple(y, y_scale);
}

}  // namespace xllm::kernel::npu
