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

#include "dali/operators/image/color/color_twist.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"

namespace dali {

DALI_SCHEMA(Hsv)
    .DocStr(R"code(Adjusts hue, saturation and value (brightness) of the images.

To change the hue, the saturation, and/or the value of the image, pass the corresponding
coefficients. Remember that the ``hue`` is an additive delta argument,
while for ``saturation`` and ``value``, the arguments are multiplicative.

This operator accepts images in the RGB color space.

For performance reasons, the operation is approximated by a linear transform in the RGB space.
The color vector is projected along the neutral (gray) axis,
rotated based on the hue delta, scaled based on the value and saturation multipliers,
and restored to the original color space.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg(color::kHue,
                    R"code(Hue delta, in degrees.

The hue component can be interpreted as an angle and values outside 0-360 range wrap around, as
they would in case of rotation.)code",
                    0.0f, true)
    .AddOptionalArg(color::kSaturation,
                    R"code(The saturation multiplier.)code",
                    1.0f, true)
    .AddOptionalArg(color::kValue,
                    R"code(The value multiplier.)code",
                    1.0f, true)
    .AddOptionalArg(color::kOutputType, R"code(The output data type.

If a value is not set, the input type is used.)code",
                    DALI_UINT8)
    .InputLayout(0, {"HWC", "FHWC", "DHWC"})
    .AllowSequences();

DALI_SCHEMA(ColorTransformBase)
    .DocStr(R"code(Base Schema for color transformations operators.)code")
    .AddOptionalArg("image_type",
        R"code(The color space of the input and the output image.)code", DALI_RGB)
    .AddOptionalArg(color::kOutputType, R"code(Output data type.

If not set, the input type is used.)code",
                    DALI_UINT8)
    .AllowSequences()
    .SupportVolumetric();

DALI_SCHEMA(Hue)
    .DocStr(R"code(Changes the hue level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue",
        R"code(The hue change in degrees.)code", 0.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, {"HWC", "FHWC", "DHWC"})
    .AllowSequences()
    .SupportVolumetric();

DALI_SCHEMA(Saturation)
    .DocStr(R"code(Changes the saturation level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("saturation",
                    R"code(The saturation change factor.

Values must be non-negative.

Example values:

- `0` - Completely desaturated image.
- `1` - No change to image's saturation.
)code", 1.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, {"HWC", "FHWC", "DHWC"})
    .AllowSequences()
    .SupportVolumetric();

DALI_SCHEMA(ColorTwist)
    .DocStr(R"code(Adjusts hue, saturation, brightness and contrast of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue",
        R"code(Hue change, in degrees.)code", 0.f, true)
    .AddOptionalArg("saturation",
                    R"code(Saturation change factor.

Values must be non-negative.

Example values:

- `0` - Completely desaturated image.
- `1` - No change to image's saturation.
)code", 1.f, true)
    .AddOptionalArg("contrast",
                    R"code(Contrast change factor.

Values must be non-negative.

Example values:

* `0` - Uniform grey image.
* `1` - No change.
* `2` - Increase brightness twice.
)code", 1.f, true)
    .AddOptionalArg("brightness",
                    R"code(Brightness change factor.

Values must be non-negative.

Example values:

* `0` - Black image.
* `1` - No change.
* `2` - Increase brightness twice.
)code", 1.f, true)
    .AddParent("ColorTransformBase")
    .InputLayout(0, {"HWC", "FHWC", "DHWC"})
    .AllowSequences()
    .SupportVolumetric();


DALI_REGISTER_OPERATOR(Hsv, ColorTwistCpu, CPU)
DALI_REGISTER_OPERATOR(Hue, ColorTwistCpu, CPU);
DALI_REGISTER_OPERATOR(Saturation, ColorTwistCpu, CPU);
DALI_REGISTER_OPERATOR(ColorTwist, ColorTwistCpu, CPU);


template <typename OutputType, typename InputType>
void ColorTwistCpu::RunImplHelper(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  auto out_shape = output.shape();
  output.SetLayout(input.GetLayout());
  auto &tp = ws.GetThreadPool();
  using Kernel = kernels::LinearTransformationCpu<OutputType, InputType, 3, 3, 3>;
  kernel_manager_.Initialize<Kernel>();
  TensorListShape<> sh = input.shape();
  auto num_dims = sh.sample_dim();
  assert(num_dims == 3 || num_dims == 4);
  int num_samples = input.shape().num_samples();
  for (int i = 0; i < num_samples; i++) {
    auto sample_shape = out_shape.tensor_shape_span(i);
    auto vol = volume(sample_shape.begin() + num_dims - 3, sample_shape.end());
    if (num_dims == 4) {
      int num_frames = sample_shape[0];
      for (int frame_id = 0; frame_id < num_frames; frame_id++) {
        tp.AddWork([&, i, frame_id](int thread_id) {
          kernels::KernelContext ctx;
          auto tvin = subtensor(view<const InputType, 4>(input[i]), frame_id);
          auto tvout = subtensor(view<OutputType, 4>(output[i]), frame_id);
          kernel_manager_.Run<Kernel>(i, ctx, tvout, tvin,
                                      tmatrices_[i], toffsets_[i]);
        }, vol);
      }
    } else {
      tp.AddWork([&, i](int thread_id) {
        kernels::KernelContext ctx;
        auto tvin = view<const InputType, 3>(input[i]);
        auto tvout = view<OutputType, 3>(output[i]);
        kernel_manager_.Run<Kernel>(i, ctx, tvout, tvin,
                                    tmatrices_[i], toffsets_[i]);
      }, vol);
    }
  }
  tp.RunAll();
}

void ColorTwistCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, InputType, COLOR_TWIST_SUPPORTED_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, COLOR_TWIST_SUPPORTED_TYPES, (
      {
        RunImplHelper<OutputType, InputType>(ws);
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}


}  // namespace dali
