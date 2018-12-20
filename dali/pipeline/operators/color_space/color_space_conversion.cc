// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/color_space/color_space_conversion.h"
#include "dali/util/ocv.h"
#include <map>

namespace dali {

DALI_SCHEMA(ColorSpaceConversion)
    .DocStr(R"code(Converts between various image color models)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("image_type",
        R"code(The color space of the input image)code", 
        DALI_IMAGE_TYPE /* TODO(janton): uncomment this? , true */ )
    .AddArg("output_type",
        R"code(The color space of the output image)code", 
        DALI_IMAGE_TYPE /* TODO(janton): uncomment this? , true */ );

template <>
void ColorSpaceConversion<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);
  const auto &input_shape = input.shape();
  auto output_shape = input_shape;

  const auto H = input_shape[0];
  const auto W = input_shape[1];
  const auto input_C = input_shape[2];

  DALI_ENFORCE( input_C == NumberOfChannels(input_type_), "Incorrect number of channels for input" );
  auto output_C = NumberOfChannels(output_type_);
  output_shape[2] = output_C;

  output->Resize(output_shape);

  auto pImgInp = input.template data<uint8>();

  const int input_channel_flag = GetOpenCvChannelType(input_C);
  const cv::Mat cv_input_img = CreateMatFromPtr(H, W, input_channel_flag, pImgInp);
  
  auto pImgOut = output->template mutable_data<uint8>();
  const int output_channel_flag = GetOpenCvChannelType(output_C);
  const cv::Mat cv_output_img = CreateMatFromPtr(H, W, output_channel_flag, pImgOut);

  cv::cvtColor(cv_input_img, cv_output_img, GetOpenCvColorConversionCode(input_type_, output_type_));
}

DALI_REGISTER_OPERATOR(ColorSpaceConversion, ColorSpaceConversion<CPUBackend>, CPU);

}  // namespace dali
