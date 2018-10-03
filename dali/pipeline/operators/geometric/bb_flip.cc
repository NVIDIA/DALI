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

#include "dali/pipeline/operators/geometric/bb_flip.h"


namespace dali {

const std::string kCoordinatesTypeArgName = "coordinates_type";  //NOLINT
const std::string kHorizontalArgName = "horizontal";  //NOLINT
const std::string kVerticalArgName = "vertical";  //NOLINT


DALI_REGISTER_OPERATOR(BbFlip, BbFlip, CPU);


DALI_SCHEMA(BbFlip)
                .DocStr(R"code(Operator for horizontal flip (mirror) of bounding box.
                               Input: Bounding box coordinates; in either [x, y, w, h]
                               or [left, top, right, bottom] format. All coordinates are
                               in the image coordinate system (i.e. 0.0-1.0))code")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg(kCoordinatesTypeArgName,
                                R"code(True, for width-height representation.
                                False for two-point (ltrb) representation. Default: True)code",
                                true, false)
                .AddOptionalArg(kHorizontalArgName,
                                R"code(Perform flip along horizontal axis. Default: 1)code",
                                1, true)
                .AddOptionalArg(kVerticalArgName,
                                R"code(Perform flip along vertical axis. Default: 0)code",
                                0, true);


BbFlip::BbFlip(const dali::OpSpec &spec) :
        Operator<CPUBackend>(spec),
        coordinates_type_wh_(spec.GetArgument<bool>(kCoordinatesTypeArgName)) {
  vflip_is_tensor_ = spec.HasTensorArgument(kVerticalArgName);
  hflip_is_tensor_ = spec.HasTensorArgument(kHorizontalArgName);
}


void BbFlip::RunImpl(dali::SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  const auto input_data = input.data<float>();

  DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Bounding box in wrong format");

  DALI_ENFORCE([](const float *data, size_t size) -> bool {
      for (size_t i = 0; i < size; i++) {
        if (data[i] < 0 || data[i] > 1.0)
          return false;
      }
      return true;
  }(input_data, input.size()), "Not all bounding box parameters are in [0.0, 1.0]");

  DALI_ENFORCE([](const float *data, size_t size, bool coors_type_wh) -> bool {
      if (!coors_type_wh) return true;  // Assert not applicable for 2-point representation
      for (size_t i = 0; i < size; i += 4) {
        if (data[i] + data[i + 2] > 1.0 || data[i + 1] + data[i + 3] > 1.0) {
          return false;
        }
      }
      return true;
  }(input_data, input.size(), coordinates_type_wh_), "Incorrect width or height");

  DALI_ENFORCE([](const float *data, size_t size, bool coors_type_wh) -> bool {
      if (coors_type_wh) return true;  // Assert not applicable for wh representation
      for (size_t i = 0; i < size; i += 4) {
        if (data[i] > data[i + 2] || data[i + 1] > data[i + 3]) {
          return false;
        }
      }
      return true;
  }(input_data, input.size(), coordinates_type_wh_), "Incorrect first or second point");

  int vertical;
  int index = ws->data_idx();
  if (vflip_is_tensor_) {
    vertical = spec_.GetArgument<int>(kVerticalArgName, ws, index);
  } else {
    vertical = spec_.GetArgument<int>(kVerticalArgName);
  }

  int horizontal;
  if (hflip_is_tensor_) {
    horizontal = spec_.GetArgument<int>(kHorizontalArgName, ws, index);
  } else {
    horizontal = spec_.GetArgument<int>(kHorizontalArgName);
  }

  auto *output = ws->Output<CPUBackend>(idx);
  // XXX: Setting type of output (i.e. Buffer -> buffer.h)
  //      explicitly is required for further processing
  //      It can also be achieved with mutable_data<>()
  //      function.
  output->set_type(TypeInfo::Create<float>());
  output->ResizeLike(input);
  auto output_data = output->mutable_data<float>();

  for (int i = 0; i < input.size(); i += 4) {
    const auto x = input_data[i];
    const auto y = input_data[i + 1];
    const auto w = coordinates_type_wh_ ? input_data[i + 2] : input_data[i + 2] - input_data[i];
    const auto h = coordinates_type_wh_ ? input_data[i + 3] : input_data[i + 3] -
                                                              input_data[i + 1];

    output_data[i] = vertical ? (1.0f - x) - w : x;
    output_data[i + 1] = horizontal ? (1.0f - y) - h : y;
    output_data[i + 2] = coordinates_type_wh_ ? w : output_data[0] + w;
    output_data[i + 3] = coordinates_type_wh_ ? h : output_data[1] + h;
  }
}


}  // namespace dali
