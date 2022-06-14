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
#include "dali/kernels/imgproc/pointwise/multiply_add.h"
#include "dali/pipeline/data/sequence_utils.h"

namespace dali {

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
                    kDefaultBrightness, true, true)
    .AddOptionalArg("brightness_shift", R"code(The brightness shift.

For signed types, 1.0 represents the maximum positive value that can be represented by
the type.)code",
                    kDefaultBrightnessShift, true, true)
    .AddOptionalTypeArg("dtype",
                    R"code(Output data type.

If not set, the input type is used.)code")
    .AllowSequences()
    .SupportVolumetric()
    .InputLayout({"FHWC", "DHWC", "HWC"});

DALI_SCHEMA(Contrast)
    .DocStr(R"code(Adjusts the contrast of the images.

The contrast is adjusted based on the following formula::

    out = contrast_center + contrast * (in - contrast_center)

This operator can also change the type of data.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("contrast", R"code(The contrast multiplier, where 0.0 produces
the uniform grey.)code",
                    kDefaultContrast, true, true)
    .AddOptionalArg("contrast_center", R"code(The intensity level that is unaffected by contrast.

This is the value that all pixels assume when the contrast is zero. When not set,
the half of the input type's positive range (or 0.5 for ``float``) is used.)code",
                    brightness_contrast::HalfRange<float>(), true, true)
    .AddOptionalTypeArg("dtype",
                    R"code(Output data type.

If not set, the input type is used.)code")
    .AllowSequences()
    .SupportVolumetric()
    .InputLayout({"FHWC", "DHWC", "HWC"});

DALI_SCHEMA(BrightnessContrast)
    .AddParent("Brightness")
    .AddParent("Contrast")
    .DocStr(R"code(Adjusts the brightness and contrast of the images.

The brightness and contrast are adjusted based on the following formula::

  out = brightness_shift * output_range + brightness * (contrast_center + contrast * (in - contrast_center))

Where the output_range is 1 for float outputs or the maximum positive value for integral types.

This operator can also change the type of data.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .InputLayout({"FHWC", "DHWC", "HWC"});

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrastCpu, CPU)
DALI_REGISTER_OPERATOR(Brightness, BrightnessContrastCpu, CPU);
DALI_REGISTER_OPERATOR(Contrast, BrightnessContrastCpu, CPU);


template <typename OutputType, typename InputType, int ndim>
void BrightnessContrastCpu::RunImplHelper(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto& tp = ws.GetThreadPool();
  int num_samples = input.num_samples();
  const auto &contrast_center = GetContrastCenter<InputType>(ws, num_samples);

  using Kernel = kernels::MultiplyAddCpu<OutputType, InputType, 3>;
  kernel_manager_.template Resize<Kernel>(1);

  auto in_view = view<const InputType, ndim>(input);
  auto out_view = view<OutputType, ndim>(output);
  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    float add, mul;
    OpArgsToKernelArgs<OutputType, InputType>(add, mul, brightness_[sample_id],
                                              brightness_shift_[sample_id], contrast_[sample_id],
                                              contrast_center[sample_id]);
    auto planes_range =
        sequence_utils::unfolded_views_range<ndim - 3>(out_view[sample_id], in_view[sample_id]);
    const auto &in_range = planes_range.template get<1>();
    for (auto &&views : planes_range) {
      tp.AddWork([&, views, add, mul](int thread_id) {
          kernels::KernelContext ctx;
          auto &[tvout, tvin] = views;
          kernel_manager_.Run<Kernel>(0, ctx, tvout, tvin, add, mul);
        }, in_range.SliceSize());
    }
  }
  tp.RunAll();
}

void BrightnessContrastCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, InputType, BRIGHTNESS_CONTRAST_SUPPORTED_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, BRIGHTNESS_CONTRAST_SUPPORTED_TYPES, (
      VALUE_SWITCH(input.sample_dim(), NDim, (3, 4), (
      {
        RunImplHelper<OutputType, InputType, NDim>(ws);
      }
      ), DALI_FAIL(make_string("Unsupported sample dimensionality: ", input.sample_dim())))  // NOLINT
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}

}  // namespace dali
