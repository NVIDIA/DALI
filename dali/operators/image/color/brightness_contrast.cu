// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
namespace {

template <typename Out, typename In>
using TheKernel = kernels::MultiplyAddGpu<Out, In, 3>;

}  // namespace

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrastGpu, GPU)
DALI_REGISTER_OPERATOR(Brightness, BrightnessContrastGpu, GPU);
DALI_REGISTER_OPERATOR(Contrast, BrightnessContrastGpu, GPU);


bool BrightnessContrastGpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<GPUBackend> &ws) {
  KMgrResize(num_threads_, max_batch_size_);
  const auto &input = ws.template InputRef<GPUBackend>(0);
  const auto &output = ws.template OutputRef<GPUBackend>(0);
  output_desc.resize(1);
  AcquireArguments(ws);
  int N = input.ntensor();
  addends_.resize(N);
  multipliers_.resize(N);
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernel_manager_.Initialize<Kernel>();
              auto &shapes = CallSetup<Kernel, InputType>(ws, input);
              TypeInfo type;
              type.SetType<OutputType>(output_type_);
              output_desc[0] = {shapes, type};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  return true;
}


void BrightnessContrastGpu::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.template Input<GPUBackend>(0);
  auto &output = ws.template Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernels::KernelContext ctx;
              ctx.gpu.stream = ws.stream();
              auto tvin = view<const InputType, 3>(input);
              auto tvout = view<OutputType, 3>(output);
              for (int i = 0; i < tvin.num_samples(); i++) {
                OpArgsToKernelArgs<OutputType, InputType>(addends_[i], multipliers_[i],
                      brightness_[i], brightness_shift_[i], contrast_[i]);
              }
              kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, tvout, tvin,
                                          addends_, multipliers_);
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
}

}  // namespace dali
