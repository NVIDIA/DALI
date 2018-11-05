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

#include <tuple>

#include "dali/pipeline/operators/crop/crop.h"
#include "dali/image/transform.h"
#include "dali/pipeline/basic/coords.h"
#include "dali/pipeline/basic/crop.h"
#include "dali/pipeline/basic/tensor.h"
#include "dali/pipeline/basic/type_switch.h"
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(Crop)
    .DocStr(R"code(Perform a random crop.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg(
        "crop_pos_x",
        R"code(Horizontal position of the crop in image coordinates (0.0 - 1.0))code",
        0.5f, true)
    .AddOptionalArg(
        "crop_pos_y",
        R"code(Vertical position of the crop in image coordinates (0.0 - 1.0))code",
        0.5f, true)
    .AddOptionalArg("image_type",
                    R"code(The color space of input and output image)code",
                    DALI_RGB, false)
    .AddOptionalArg(
        "crop",
        R"code(Size of the cropped image. If only a single value `c` is provided,
the resulting crop will be square with size `(c,c)`)code",
        std::vector<float>{0.f, 0.f})
    .EnforceInputLayout(DALI_NHWC);

std::pair<int, int> CropAttr::SetCropXY(const OpSpec &spec, const ArgumentWorkspace *ws,
    const Index dataIdx, int H, int W) {

    auto crop_x_norm = spec.GetArgument<float>("crop_pos_x", ws, dataIdx);
    auto crop_y_norm = spec.GetArgument<float>("crop_pos_y", ws, dataIdx);

    DALI_ENFORCE(crop_y_norm >= 0.f && crop_y_norm <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_x_norm >= 0.f && crop_x_norm <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");

    const int crop_y = crop_y_norm * (H - crop_height_[dataIdx]);
    const int crop_x = crop_x_norm * (W - crop_width_[dataIdx]);

    return std::make_pair(crop_y, crop_x);
}

const vector<Index> CropAttr::CheckShapes(const SampleWorkspace *ws) {
  const auto &input = ws->Input<CPUBackend>(0);

  // enforce that all shapes match
  for (int i = 1; i < ws->NumInput(); ++i) {
    DALI_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
  }

  DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");

  return input.shape();
}

template <>
Crop<CPUBackend>::Crop(const OpSpec &spec) : Operator<CPUBackend>(spec), CropAttr(spec) {
  Init(num_threads_);
}

template <>
void Crop<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto *output = ws->Output<CPUBackend>(idx);

  DALITensorLayout outLayout = output_layout_ == DALI_SAME ? input.GetLayout() : output_layout_;
  output->SetLayout(outLayout);
  // output->Resize(GetOutShape(input.GetLayout(), &outLayout));
  auto layout_id = layoutToTypeId(output->GetLayout());
  using CropOutputTypes = std::tuple<uint8_t, int16_t, int32_t, int64_t, float>;
  using CropOutputPermTypes =
      std::tuple<dali_index_sequence<0, 1, 2>, dali_index_sequence<2, 0, 1>>;

  // Check if we use u8, RGB or Greyscale
  CheckParam(input, "CropCPUBackend");

  const int threadIdx = ws->thread_idx();
  const int h_start = per_sample_crop_[threadIdx].first;
  const int w_start = per_sample_crop_[threadIdx].second;

  const int dataIdx = ws->data_idx();
  // TODO(klecki): currently float16 on CPU is probably broken as far as TypeInfo is considered
  if (output_type_ == DALI_FLOAT16) {
    if (output->GetLayout() == DALITensorLayout::DALI_NHWC) {
      basic::CropSizeHelper<half_float::half, dali_index_sequence<0, 1, 2>>::Run(
          ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);
    } else {
      basic::CropSizeHelper<half_float::half, dali_index_sequence<2, 0, 1>>::Run(
          ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);
    }
  } else {
    type_switch<basic::CropSizeHelper, CropOutputTypes, CropOutputPermTypes>::Run(
        output_type_, layout_id, ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);
  }

  if (output_type_ == DALI_FLOAT16) {
    if (output->GetLayout() == DALITensorLayout::DALI_NHWC) {
      basic::CropRunHelper<half_float::half, dali_index_sequence<0, 1, 2>>::Run(
          ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);
    } else {
      basic::CropRunHelper<half_float::half, dali_index_sequence<2, 0, 1>>::Run(
          ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);
    }
  } else {
    type_switch<basic::CropRunHelper, CropOutputTypes, CropOutputPermTypes>::Run(
        output_type_, layout_id, ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);
  }
}

template <>
void Crop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  if (output_type_ == DALI_NO_TYPE) {
    const auto &input = ws->Input<CPUBackend>(0);
    output_type_ = input.type().id();
  }

  SetupSharedSampleParams(ws, CheckShapes(ws), ws->thread_idx(),
                          ws->data_idx());
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<CPUBackend>, CPU);

}  // namespace dali
