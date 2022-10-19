// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/image/color/color_twist.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_gpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(Hsv, ColorTwistGpu, GPU)
DALI_REGISTER_OPERATOR(Hue, ColorTwistGpu, GPU);
DALI_REGISTER_OPERATOR(Saturation, ColorTwistGpu, GPU);
DALI_REGISTER_OPERATOR(ColorTwist, ColorTwistGpu, GPU);

template <typename OutputType, typename InputType>
void ColorTwistGpu::RunImplHelper(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto sh = input.shape();
  auto num_dims = sh.sample_dim();

  TensorListView<StorageGPU, const InputType, 3> tvin;
  TensorListView<StorageGPU, OutputType, 3> tvout;
  if (num_dims == 4) {
    auto collapsed_sh = collapse_dim(view<const InputType, 4>(input).shape, 0);
    tvin = reinterpret<const InputType, 3>(view<const InputType, 4>(input), collapsed_sh, true);
    tvout = reinterpret<OutputType, 3>(view<OutputType, 4>(output), collapsed_sh, true);
  } else {
    tvin = view<const InputType, 3>(input);
    tvout = view<OutputType, 3>(output);
  }

  using Kernel = kernels::LinearTransformationGpu<OutputType, InputType, 3, 3, 2>;
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kernel_manager_.template Resize<Kernel>(sh.num_samples());

  auto tmatrices = make_cspan(tmatrices_);
  auto toffsets = make_cspan(toffsets_);
  kernel_manager_.Setup<Kernel>(0, ctx, tvin, tmatrices, toffsets);
  kernel_manager_.Run<Kernel>(0, ctx, tvout, tvin, tmatrices, toffsets);
}

void ColorTwistGpu::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, InputType, COLOR_TWIST_SUPPORTED_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, COLOR_TWIST_SUPPORTED_TYPES, (
      {
        RunImplHelper<OutputType, InputType>(ws);
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}


}  // namespace dali
