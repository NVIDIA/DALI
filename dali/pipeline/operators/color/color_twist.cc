// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/color/color_twist.h"
#include "dali/image/transform.h"

namespace dali {

DALI_SCHEMA(ColorTransformBase)
    .DocStr(R"code(Base Schema for color transformations operators.)code")
    .AddOptionalArg("image_type",
        R"code(The color space of input and output image)code", DALI_RGB);

DALI_SCHEMA(Brightness)
    .DocStr(R"code(Changes the brightness of an image)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("brightness",
        R"code(Brightness change factor.
Values >= 0 are accepted. For example:

* `0` - black image,
* `1` - no change
* `2` - increase brightness twice
)code", 1.f, true)
    .AddParent("ColorTransformBase");

DALI_SCHEMA(Contrast)
    .DocStr(R"code(Changes the color contrast of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("contrast",
        R"code(Contrast change factor.
Values >= 0 are accepted. For example:

* `0` - gray image,
* `1` - no change
* `2` - increase contrast twice
)code", 1.f, true)
    .AddParent("ColorTransformBase");

DALI_SCHEMA(Hue)
    .DocStr(R"code(Changes the hue level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue",
        R"code(Hue change, in degrees.)code", 0.f, true)
    .AddParent("ColorTransformBase");

DALI_SCHEMA(Saturation)
    .DocStr(R"code(Changes saturation level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("saturation",
        R"code(Saturation change factor.
Values >= 0 are supported. For example:

* `0` - completely desaturated image
* `1` - no change to image's saturation
)code", 1.f, true)
    .AddParent("ColorTransformBase");

DALI_SCHEMA(ColorTwist)
    .DocStr(R"code(Combination of hue, saturation, contrast and brightness.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue",
        R"code(Hue change, in degrees.)code", 0.f, true)
    .AddOptionalArg("saturation",
        R"code(Saturation change factor.
Values >= 0 are supported. For example:

* `0` - completely desaturated image
* `1` - no change to image's saturation
)code", 1.f, true)
    .AddOptionalArg("contrast",
        R"code(Contrast change factor.
Values >= 0 are accepted. For example:

* `0` - gray image,
* `1` - no change
* `2` - increase contrast twice
)code", 1.f, true)
    .AddOptionalArg("brightness",
        R"code(Brightness change factor.
Values >= 0 are accepted. For example:

* `0` - black image,
* `1` - no change
* `2` - increase brightness twice

)code", 1.f, true)
    .AddParent("ColorTransformBase");

template <>
void ColorTwistBase<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  const auto &input_shape = input.shape();

  CheckParam(input, "Color augmentation");

  const auto H = input_shape[0];
  const auto W = input_shape[1];
  const auto C = input_shape[2];

  output.ResizeLike(input);

  auto pImgInp = input.template data<uint8>();
  auto pImgOut = output.template mutable_data<uint8>();

  if (!augments_.empty()) {
    float matrix[nDim][nDim];
    float *m = reinterpret_cast<float *>(matrix);
    IdentityMatrix(m);
    for (size_t j = 0; j < augments_.size(); ++j) {
      augments_[j]->Prepare(ws->data_idx(), spec_, ws);
      (*augments_[j])(m);
    }

    MakeColorTransformation(pImgInp, H, W, C, m, pImgOut);
  } else {
    memcpy(pImgOut, pImgInp, H * W * C);
  }
}

DALI_REGISTER_OPERATOR(Brightness, BrightnessAdjust<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Contrast, ContrastAdjust<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Hue, HueAdjust<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Saturation, SaturationAdjust<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ColorTwist, ColorTwistAdjust<CPUBackend>, CPU);

}  // namespace dali
