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
#include <dali/pipeline/util/bounding_box.h>

namespace dali {

const std::string kCoordinatesTypeArgName = "ltrb";   // NOLINT
const std::string kHorizontalArgName = "horizontal";  // NOLINT
const std::string kVerticalArgName = "vertical";      // NOLINT

DALI_REGISTER_OPERATOR(BbFlip, BbFlip, CPU);

DALI_SCHEMA(BbFlip)
    .DocStr(R"code(Operator for horizontal flip (mirror) of bounding box.
Input: Bounding box coordinates; in either [x, y, w, h]
or [left, top, right, bottom] format. All coordinates are
in the image coordinate system (i.e. 0.0-1.0))code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg(kCoordinatesTypeArgName,
                    R"code(True, for two-point (ltrb).
False for for width-height representation. Default: False)code",
                    false, false)
    .AddOptionalArg(kHorizontalArgName,
                    R"code(Perform flip along horizontal axis. Default: 1)code",
                    1, true)
    .AddOptionalArg(kVerticalArgName,
                    R"code(Perform flip along vertical axis. Default: 0)code",
                    0, true);

BbFlip::BbFlip(const dali::OpSpec &spec)
    : Operator<CPUBackend>(spec),
      coordinates_type_ltrb_(spec.GetArgument<bool>(kCoordinatesTypeArgName)) {
  vflip_is_tensor_ = spec.HasTensorArgument(kVerticalArgName);
  hflip_is_tensor_ = spec.HasTensorArgument(kHorizontalArgName);
}

void BbFlip::RunImpl(dali::SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  const auto input_data = input.data<float>();

  DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Bounding box in wrong format");

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
    auto bbox =
        coordinates_type_ltrb_
            ? BoundingBox::FromLtrb(input_data[i], input_data[i + 1],
                                    input_data[i + 2], input_data[i + 3])
            : BoundingBox::FromXywh(input_data[i], input_data[i + 1],
                                    input_data[i + 2], input_data[i + 3]);

    if (horizontal) {
      bbox = bbox.HorizontalFlip();
    }
    if (vertical) {
      bbox = bbox.VerticalFlip();
    }

    auto result =
        coordinates_type_ltrb_ ? bbox.AsLtrb() : bbox.AsXywh();

    output_data[i] = result[0];
    output_data[i + 1] = result[1];
    output_data[i + 2] = result[2];
    output_data[i + 3] = result[3];
  }
}  // namespace dali

}  // namespace dali
