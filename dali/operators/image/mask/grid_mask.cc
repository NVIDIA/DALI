// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/static_switch.h"
#include "dali/operators/image/mask/grid_mask.h"
#include "dali/kernels/mask/grid_mask_cpu.h"
#include "dali/core/format.h"

#define TYPES (uint8_t, int16_t, int32_t, float)

namespace dali {

DALI_SCHEMA(GridMask)
    .DocStr(R"code(Performs the gridmask augumentation (https://arxiv.org/abs/2001.04086).

Zeroes out pixels of an image in a grid-like fashion. The grid
consists of squares repeating in x and y directions, with the same spacing in
both directions. Can be rotated around the origin.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("tile", R"code(The length of a single tile, which is equal to
width of black squares plus the spacing between them.)code",
                    100, true)
    .AddOptionalArg("ratio",
                    "The ratio between black square width and tile width.",
                    0.5f, true)
    .AddOptionalArg("angle",
                    "Angle, in radians, by which the grid is rotated.",
                    0.0f, true)
    .AddOptionalArg("shift_x",
                    "The x component of the translation vector, applied after rotation.",
                    0.0f, true)
    .AddOptionalArg("shift_y",
                    "The y component of the translation vector, applied after rotation.",
                    0.0f, true);

bool GridMaskCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                            const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  const auto &output = ws.template OutputRef<CPUBackend>(0);
  output_desc.resize(1);
  GetArguments(ws);
  output_desc[0] = {input.shape(), input.type()};
  kernel_manager_.Resize(num_threads_, max_batch_size_);
  TYPE_SWITCH(input.type().id(), type2id, Type, TYPES, (
      {
          using Kernel = kernels::GridMaskCpu<Type>;
          kernel_manager_.Initialize<Kernel>();
      }
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id()))) // NOLINT
  return true;
}


void GridMaskCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto out_shape = output.shape();
  auto& tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, Type, TYPES, (
      {
          using Kernel = kernels::GridMaskCpu<Type>;
          auto in_view = view<const Type>(input);
          auto out_view = view<Type>(output);
          for (int sid = 0; sid < input.shape().num_samples(); sid++) {
            tp.AddWork([&, sid](int tid) {
              kernels::KernelContext ctx;
              kernel_manager_.Run<Kernel>(tid, sid, ctx, out_view[sid], in_view[sid],
                tile_[sid], ratio_[sid], angle_[sid], shift_x_[sid], shift_y_[sid]);
            }, out_shape.tensor_size(sid));
          }
          tp.RunAll();
      }
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id()))) // NOLINT
}

DALI_REGISTER_OPERATOR(GridMask, GridMaskCpu, CPU);

}  // namespace dali
