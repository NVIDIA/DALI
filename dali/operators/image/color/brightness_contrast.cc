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
#include "dali/kernels/imgproc/pointwise/multiply_add.h"

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::MultiplyAddCpu<Out, In, 3>;

}  // namespace


DALI_SCHEMA(Brightness)
    .DocStr(R"code(Adjusts the brightness of the images.

The brightness is adjusted based on the following formula::

    out = brightness_shift * output_range + brightness * in

Where output_range is 1 for float outputs or the maximum positive value for integral types.

This operator can also change the type of data.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("brightness",
                    "Brightness mutliplier.",
                    kDefaultBrightness, true)
    .AddOptionalArg("brightness_shift", R"code(The brightness shift.

For signed types, 1.0 represents the maximum positive value that can be represented by
the type.)code",
                    kDefaultBrightnessShift, true)
    .AddOptionalArg("dtype",
                    R"code(Output data type.

If not set, the input type is used.)code", DALI_NO_TYPE);

DALI_SCHEMA(Contrast)
    .DocStr(R"code(Adjusts the contrast of the images.

The contrast is adjusted based on the following formula::

    out = contrast_center + contrast * (in - contrast_center)

This operator can also change the type of data.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("contrast", R"code(The contrast multiplier, where 0.0 produces
the uniform grey.)code",
                    kDefaultContrast, true)
    .AddOptionalArg("contrast_center", R"code(The intensity level that is unaffected by contrast.

This is the value that all pixels assume when the contrast is zero. When not set,
the half of the input type's positive range (or 0.5 for ``float``) is used.)code",
                    brightness_contrast::HalfRange<float>(), false)
    .AddOptionalArg("dtype",
                    R"code(Output data type.

If not set, the input type is used.)code", DALI_NO_TYPE);

DALI_SCHEMA(BrightnessContrast)
    .AddParent("Brightness")
    .AddParent("Contrast")
    .DocStr(R"code(Adjusts the brightness and contrast of the images.

The brightness and contrast are adjusted based on the following formula::

  out = brightness_shift * output_range + brightness * (contrast_center + contrast * (in - contrast_center))

Where the output_range is 1 for float outputs or the maximum positive value for integral types.

This operator can also change the type of data.)code")
    .NumInput(1)
    .NumOutput(1);

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrastCpu, CPU)
DALI_REGISTER_OPERATOR(Brightness, BrightnessContrastCpu, CPU);
DALI_REGISTER_OPERATOR(Contrast, BrightnessContrastCpu, CPU);


bool BrightnessContrastCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  KMgrResize(num_threads_, max_batch_size_);
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
  output.SetLayout(input.GetLayout());
  auto out_shape = output.shape();
  auto& tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              for (int sample_id = 0; sample_id < input.shape().num_samples(); sample_id++) {
                tp.AddWork([&, sample_id](int thread_id) {
                    kernels::KernelContext ctx;
                    auto tvin = view<const InputType, 3>(input[sample_id]);
                    auto tvout = view<OutputType, 3>(output[sample_id]);
                    float add, mul;
                    OpArgsToKernelArgs<OutputType, InputType>(add, mul,
                      brightness_[sample_id], brightness_shift_[sample_id], contrast_[sample_id]);
                    kernel_manager_.Run<Kernel>(thread_id, sample_id, ctx, tvout, tvin,
                                                add, mul);
                }, out_shape.tensor_size(sample_id));
              }
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  tp.RunAll();
}

}  // namespace dali
