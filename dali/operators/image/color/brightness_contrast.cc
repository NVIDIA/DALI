// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/imgproc/pointwise/multiply_add.h"

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::MultiplyAddCpu<Out, In, 3>;

}  // namespace


DALI_SCHEMA(BrightnessContrast)
    .DocStr(R"code(Adjust the brightness and contrast of the image according to the formula::

  out = brightness_shift * output_range + brightness * (grey + contrast * (in - grey))

where output_range is 1 for float outputs or the maximum positive value for integral types;
grey denotes the value of 0.5 for float, 128 for `uint8`, 16384 for `int16`, etc.

Additionally, this operator can change the type of data.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("brightness",
                    "Brightness mutliplier; 1.0 is neutral.",
                    1.0f, true)
    .AddOptionalArg("brightness_shift",
                    "Brightness shift; 0 is neutral; for signed types, 1.0 means maximum positive "
                    "value that can be represented by the type.",
                    0.0f, true)
    .AddOptionalArg("contrast",
                    "Set the contrast multiplier; 1.0 is neutral, 0.0 produces uniform grey.",
                    1.0f, true)
    .AddOptionalArg("contrast_center",
                    "Sets the instensity level that is unaffected by contrast - this is the value "
                    "which all pixels assume when contrast is zero. When not set, the half of the "
                    "input types's positive range (or 0.5 for float) is used.",
                    0.5f, false)
    .AddOptionalArg("dtype",
                    "Output data type; if not set, the input type is used.", DALI_NO_TYPE);

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrastCpu, CPU)


bool BrightnessContrastCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  const auto &output = ws.template OutputRef<CPUBackend>(0);
  output_desc.resize(1);
  AcquireArguments(ws);
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              kernel_manager_.Initialize<Kernel>();
              auto shapes = CallSetup<Kernel, InputType>(input);
              TypeInfo type;
              type.SetType<OutputType>(output_type_);
              output_desc[0] = {shapes, type};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  return true;
}


void BrightnessContrastCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  output.SetLayout(InputLayout(ws, 0));
  auto& tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              for (int sample_id = 0; sample_id < input.shape().num_samples(); sample_id++) {
                tp.DoWorkWithID([&, sample_id](int thread_id) {
                    kernels::KernelContext ctx;
                    auto tvin = view<const InputType, 3>(input[sample_id]);
                    auto tvout = view<OutputType, 3>(output[sample_id]);
                    float add, mul;
                    OpArgsToKernelArgs<OutputType, InputType>(add, mul,
                      brightness_[sample_id], brightness_shift_[sample_id], contrast_[sample_id]);
                    kernel_manager_.Run<Kernel>(thread_id, sample_id, ctx, tvout, tvin,
                                                add, mul);
                });
              }
          }
      ), DALI_FAIL("Unsupported output type"))  // NOLINT
  ), DALI_FAIL("Unsupported input type"))  // NOLINT
  tp.WaitForWork();
}

}  // namespace dali
