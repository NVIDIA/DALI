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

#include "dali/operators/image/color/color_twist.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::LinearTransformationCpu<Out, In, 3, 3, 3>;

}  // namespace

DALI_SCHEMA(Hsv)
    .DocStr(R"code(This operator performs an HSV manipulation.

To change the hue, the saturation, and/or the value of the image, pass the corresponding
coefficients. Remember that the ``hue`` has an additive delta argument,
while for ``saturation`` and value, the arguments are multiplicative.

This operator accepts the RGB color space as an input.

For performance reasons, the operation is approximated by a linear transform in the RGB space.
The color vector is projected along the neutral (gray) axis,
is rotated based on the hue delta, is scaled based on the value and saturation multipliers,
and is restored to the original color space.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg(color::kHue,
                    R"code(Sets the additive change of the hue.

A value of 0 denotes no-op.)code",
                    0.0f, true)
    .AddOptionalArg(color::kSaturation,
                    R"code(Sets the multiplicative change of saturation, and 1 denotes no-op)code",
                    1.0f, true)
    .AddOptionalArg(color::kValue,
                    R"code(Sets the multiplicative change of value, and 1 denotes no-op)code",
                    1.0f, true)
    .AddOptionalArg(color::kOutputType, R"code(The output data type.

If a value is not set, the input type is used.)code",
                    DALI_UINT8)
    .InputLayout(0, "HWC");

DALI_SCHEMA(ColorTransformBase)
    .DocStr(R"code(Base Schema for color transformations operators.)code")
    .AddOptionalArg("image_type",
        R"code(TThe color space of the input and the output image.)code", DALI_RGB)
    .AddOptionalArg(color::kOutputType, R"code(The output data type.

If a value is not set, the input type is used.)code",
                    DALI_UINT8);

DALI_SCHEMA(Brightness)
    .DocStr(R"code(Changes the brightness of an image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("brightness",
                    R"code(Brightness change factor.

Here is a list of the values:

* `0` - Black image,
* `1` - No change
* `2` - Increase brightness twice

.. note::
    Only values greater than 0 are accepted.)code", 1.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, "HWC");

DALI_SCHEMA(Contrast)
    .DocStr(R"code(Changes the color contrast of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("contrast",
                    R"code(Contrast change factor.

Here is a list of the values:

* `0` - Grey image.
* `1` - No change
* `2` - Increase brightness twice.

.. note::
    Values must be greater than 0.)code", 1.f, true)
    .AddParent("ColorTransformBase");

DALI_SCHEMA(Hue)
    .DocStr(R"code(Changes the hue level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue",
        R"code(Hue change in degrees.)code", 0.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, "HWC");

DALI_SCHEMA(Saturation)
    .DocStr(R"code(Changes the saturation level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("saturation",
                    R"code(Saturation change factor.

Here is a list of the values:

- `0` – Completely desaturated image
- `1` - No change to image’s saturation.

.. note::
    Values must be greater than 0.)code", 1.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, "HWC");

DALI_SCHEMA(ColorTwist)
    .DocStr(R"code(Combines the hue, the saturation, the contrast, and the brightness.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue",
        R"code(Hue change, in degrees.)code", 0.f, true)
    .AddOptionalArg("saturation",
                    R"code(Saturation change factor.

Here is a list of the values:

- `0` – Completely desaturated image
- `1` - No change to image’s saturation.

.. note::
    Values must be greater than 0.)code", 1.f, true)
    .AddOptionalArg("contrast",
                    R"code(Contrast change factor.

Here is a list of the values:

* `0` - Grey image.
* `1` - No change
* `2` - Increase brightness twice.

.. note::
    Values must be greater than 0.)code", 1.f, true)
    .AddOptionalArg("brightness",
                    R"code(Brightness change factor.

Here is a list of the values:

* `0` - Black image,
* `1` - No change
* `2` - Increase brightness twice

.. note::
    Only values greater than 0 are accepted.)code", 1.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, "HWC");


DALI_REGISTER_OPERATOR(Hsv, ColorTwistCpu, CPU)
DALI_REGISTER_OPERATOR(Brightness, ColorTwistCpu, CPU);
DALI_REGISTER_OPERATOR(Contrast, ColorTwistCpu, CPU);
DALI_REGISTER_OPERATOR(Hue, ColorTwistCpu, CPU);
DALI_REGISTER_OPERATOR(Saturation, ColorTwistCpu, CPU);
DALI_REGISTER_OPERATOR(ColorTwist, ColorTwistCpu, CPU);

bool ColorTwistCpu::SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  output_desc.resize(1);
  DetermineTransformation(ws);
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float, float16), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float, float16), (
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


void ColorTwistCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  auto out_shape = output.shape();
  output.SetLayout(input.GetLayout());
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float, float16), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float, float16), (
          {
              using Kernel = TheKernel<OutputType, InputType>;
              for (int i = 0; i < input.shape().num_samples(); i++) {
                tp.AddWork([&, i](int thread_id) {
                  kernels::KernelContext ctx;
                  auto tvin = view<const InputType, 3>(input[i]);
                  auto tvout = view<OutputType, 3>(output[i]);
                  kernel_manager_.Run<Kernel>(ws.thread_idx(), i, ctx, tvout, tvin,
                                              tmatrices_[i], toffsets_[i]);
                }, out_shape.tensor_size(i));
              }
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  tp.RunAll();
}


}  // namespace dali
