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


DALI_REGISTER_OPERATOR(BbFlip, BbFlip, CPU);


DALI_SCHEMA(BbFlip)
                .DocStr(R"code(Operator for horizontal flip (mirror) of bounding box.
                               Input: Bounding box coordinates; in either [x, y, w, h]
                               or [left, top, right, bottom] format. All coordinates are
                               in the image coordinate system (i.e. 0.0-1.0))code")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg("coordinates_type",
                                R"code(True, for width-height representation.
                                False for two-point (ltrb) representation. Default: True)code",
                                true, false)
                .AddOptionalArg("flip_type",
                                R"code(True, for vertical flip (along vertical axis).
                                False for horizontal flip. Default: True)code",
                                true, false)
                .AddOptionalArg("on_off_switch",
                                R"code(Turns the operator on (True) and off (False).
                                Main purpose of this argument is to implement
                                randomness inside operation. Default value: True)code",
                                true, true);


BbFlip::BbFlip(const dali::OpSpec &spec) :
        Operator<CPUBackend>(spec),
        coordinates_type_wh_(spec.GetArgument<bool>("coordinates_type")),
        flip_type_vertical_(spec.GetArgument<bool>("flip_type")),
        on_off_switch_(spec.GetArgument<bool>("on_off_switch")) {
}


void BbFlip::RunImpl(dali::SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  const auto input_data = input.data<float>();

  DALI_ENFORCE(input.size() == kBbTypeSize, "Bounding box in wrong format");
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


  auto output = ws->Output<CPUBackend>(idx);
  // XXX: Setting type of output (i.e. Buffer -> buffer.h)
  //      explicitly is required for further processing
  //      It can also be achieved with mutable_data<>()
  //      function.
  output->set_type(TypeInfo::Create<float>());
  output->Resize({kBbTypeSize});
  auto output_data = output->mutable_data<float>();

  if (on_off_switch_) {
    const auto x = input_data[0];
    const auto y = input_data[1];
    const auto w = coordinates_type_wh_ ? input_data[2] : input_data[2] - input_data[0];
    const auto h = coordinates_type_wh_ ? input_data[3] : input_data[3] - input_data[1];

    output_data[0] = flip_type_vertical_ ? (1.0f - x) - w : x;
    output_data[1] = flip_type_vertical_ ? y : (1.0f - y) - h;
    output_data[2] = coordinates_type_wh_ ? w : output_data[0] + w;
    output_data[3] = coordinates_type_wh_ ? h : output_data[1] + h;
  } else {
    for (int i = 0; i < kBbTypeSize; i++) {
      output_data[i] = input_data[i];
    }
  }
}

}  // namespace dali
