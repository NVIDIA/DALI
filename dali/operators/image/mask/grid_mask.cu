// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "dali/core/static_switch.h"
#include "dali/operators/image/mask/grid_mask.h"
#include "dali/kernels/mask/grid_mask_gpu.h"

#define GRID_MASK_TYPES (float, uint8_t)

namespace dali {

bool GridMaskGpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                            const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  const auto &output = ws.Output<GPUBackend>(0);
  output_desc.resize(1);
  GetArguments(ws);
  output_desc[0] = {input.shape(), input.type()};
  kernel_manager_.Resize(1);
  TYPE_SWITCH(input.type(), type2id, Type, GRID_MASK_TYPES, (
      {
          using Kernel = kernels::GridMaskGpu<Type>;
          kernels::KernelContext ctx;
          ctx.gpu.stream = ws.stream();
          kernel_manager_.Initialize<Kernel>();
          kernel_manager_.Setup<Kernel>(0, ctx, view<const Type, 3>(input));
      }
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type()))) // NOLINT
  return true;
}

void GridMaskGpu::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  TYPE_SWITCH(input.type(), type2id, Type, GRID_MASK_TYPES, (
      {
          using Kernel = kernels::GridMaskGpu<Type>;
          kernels::KernelContext ctx;
          ctx.gpu.stream = ws.stream();
          auto in_view = view<const Type, 3>(input);
          auto out_view = view<Type, 3>(output);

          kernel_manager_.Run<Kernel>(0, ctx, out_view, in_view,
                tile_, ratio_, angle_, shift_x_, shift_y_);
      }
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type()))) // NOLINT
}

DALI_REGISTER_OPERATOR(GridMask, GridMaskGpu, GPU);

}  // namespace dali
