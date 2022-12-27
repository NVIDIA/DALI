// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/color/brightness_contrast.h"
#include <vector>
#include "dali/kernels/imgproc/pointwise/multiply_add_gpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrastGpu, GPU)
DALI_REGISTER_OPERATOR(Brightness, BrightnessContrastGpu, GPU);
DALI_REGISTER_OPERATOR(Contrast, BrightnessContrastGpu, GPU);

template <typename OutputType, typename InputType>
void BrightnessContrastGpu::RunImplHelper(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto sh = input.shape();
  int num_samples = input.num_samples();
  const auto &contrast_center = GetContrastCenter<InputType>(ws, num_samples);
  auto num_dims = sh.sample_dim();

  addends_.resize(num_samples);
  multipliers_.resize(num_samples);
  for (int i = 0; i < num_samples; i++) {
    OpArgsToKernelArgs<OutputType, InputType>(addends_[i], multipliers_[i], brightness_[i],
                                              brightness_shift_[i], contrast_[i],
                                              contrast_center[i]);
  }

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

  using Kernel = kernels::MultiplyAddGpu<OutputType, InputType, 3>;
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kernel_manager_.template Resize<Kernel>(1);

  kernel_manager_.Setup<Kernel>(0, ctx, tvin, brightness_, contrast_);
  kernel_manager_.Run<Kernel>(0, ctx, tvout, tvin, addends_, multipliers_);
}

void BrightnessContrastGpu::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, InputType, BRIGHTNESS_CONTRAST_SUPPORTED_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, BRIGHTNESS_CONTRAST_SUPPORTED_TYPES, (
      {
        RunImplHelper<OutputType, InputType>(ws);
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}

}  // namespace dali
