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

#include "dali/operators/bbox/bb_flip.h"
#include <iterator>
#include "dali/core/geom/box.h"
#include "dali/pipeline/util/bounding_box_utils.h"

namespace dali {

const std::string kCoordinatesTypeArgName = "ltrb";   // NOLINT
const std::string kHorizontalArgName = "horizontal";  // NOLINT
const std::string kVerticalArgName = "vertical";      // NOLINT

DALI_REGISTER_OPERATOR(BbFlip, BbFlip<CPUBackend>, CPU);

DALI_SCHEMA(BbFlip)
    .DocStr(R"code(Flips bounding boxes horizontaly or verticaly (mirror).

The bounding box coordinates for the  input are in the [x, y, width, height] - ``xywh`` or
[left, top, right, bottom] - ``ltrb`` format. All coordinates are in the image coordinate
system, that is 0.0-1.0)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg(kCoordinatesTypeArgName,
                    R"code(True for ``ltrb`` or False for ``xywh``.)code",
                    false, false)
    .AddOptionalArg(kHorizontalArgName,
                    R"code(Flip horizontal dimension.)code",
                    1, true)
    .AddOptionalArg(kVerticalArgName,
                    R"code(Flip vertical dimension.)code",
                    0, true);

BbFlip<CPUBackend>::BbFlip(const dali::OpSpec &spec)
    : Operator<CPUBackend>(spec),
      ltrb_(spec.GetArgument<bool>(kCoordinatesTypeArgName)),
      horz_("horizontal", spec),
      vert_("vertical", spec) {}

void BbFlip<CPUBackend>::RunImpl(dali::SampleWorkspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  const auto input_data = input.data<float>();
  DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Bounding box in wrong format");

  bool vertical = vert_.IsDefined() && vert_[ws.data_idx()].data[0];
  bool horizontal = horz_.IsDefined() && horz_[ws.data_idx()].data[0];

  auto &output = ws.Output<CPUBackend>(0);
  // XXX: Setting type of output (i.e. Buffer -> buffer.h)
  //      explicitly is required for further processing
  //      It can also be achieved with mutable_data<>()
  //      function.
  output.set_type(TypeInfo::Create<float>());
  output.ResizeLike(input);
  auto output_data = output.mutable_data<float>();

  std::vector<Box<2, float>> bboxes;
  constexpr int box_size = Box<2, float>::size;
  assert(input.size() % box_size == 0);
  int nboxes = input.size() / box_size;
  bboxes.resize(nboxes);
  TensorLayout layout = ltrb_ ? "xyXY" : "xyWH";
  ReadBoxes(make_span(bboxes), make_cspan(input_data, input.size()), layout, {});

  for (auto &bbox : bboxes) {
    if (horizontal)
      HorizontalFlip(bbox);
    if (vertical)
      VerticalFlip(bbox);
  }

  WriteBoxes(make_span(output_data, output.size()), make_cspan(bboxes), layout);
}

}  // namespace dali
