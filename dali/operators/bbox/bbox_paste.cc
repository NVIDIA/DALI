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

#include <cmath>

#include "dali/operators/bbox/bbox_paste.h"

namespace dali {

DALI_SCHEMA(BBoxPaste)
    .DocStr(
        R"code(Transforms bounding boxes so that the boxes remain in the same
place in the image after the image is pasted on a larger canvas.

Corner coordinates are transformed according to the following formula::

  (x', y') = (x/ratio + paste_x', y/ratio + paste_y')

Box sizes (if ``xywh`` is used) are transformed according to the following formula::

  (w', h') = (w/ratio, h/ratio)

Where::

  paste_x' = paste_x * (ratio - 1)/ratio
  paste_y' = paste_y * (ratio - 1)/ratio

The paste coordinates are normalized so that ``(0,0)`` aligns the image to top-left of the
canvas and ``(1,1)`` aligns it to bottom-right.
)code")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("ratio",
      R"code(Ratio of the canvas size to the input size; the value must be at least 1.)code",
      DALI_FLOAT, true)
  .AddOptionalArg("ltrb",
              R"code(True for ``ltrb`` or False for ``xywh``.)code",
              false, false)
  .AddOptionalArg("paste_x",
      R"code(Horizontal position of the paste in image coordinates (0.0 - 1.0).)code",
      0.5f, true)
  .AddOptionalArg("paste_y",
      R"code(Vertical position of the paste in image coordinates (0.0 - 1.0).)code",
      0.5f, true);

template<>
void BBoxPaste<CPUBackend>::RunImpl(Workspace<CPUBackend> &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  const auto input_data = input.data<float>();

  DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Bounding box in wrong format");
  DALI_ENFORCE(input.size() % 4 == 0, "Bounding box tensor size must be a multiple of 4."
                                      "Got: " + std::to_string(input.size()));

  auto &output = ws.Output<CPUBackend>(0);
  output.set_type(TypeInfo::Create<float>());
  output.ResizeLike(input);
  auto *output_data = output.mutable_data<float>();

  const auto data_idx = ws.data_idx();
  // pasting onto a larger canvas scales bounding boxes down by scale ratio
  float ratio = spec_.GetArgument<float>("ratio", &ws, data_idx);
  float px = spec_.GetArgument<float>("paste_x", &ws, data_idx);
  float py = spec_.GetArgument<float>("paste_y", &ws, data_idx);
  float scale = 1 / ratio;

  // offsets are scaled so that (0,0) pastes the image aligned to the top-left
  // corner and (1,1) aligns it to the (bottom, right) corner
  float ofs_mul = (ratio - 1) / ratio;
  float ofsx = px * ofs_mul;
  float ofsy = py * ofs_mul;

  // this ensures that the boxes that were in (0,1) range still are after pasting
  if (scale + ofsx > 1) {
    ofsx = 1 - scale;
    while (scale + ofsx > 1)
      ofsx = std::nextafter(ofsx, -1.0f);
  }
  if (scale + ofsy > 1) {
    ofsy = 1 - scale;
    while (scale + ofsy > 1)
      ofsy = std::nextafter(ofsy, -1.0f);
  }

  for (int j = 0; j + 4 <= input.size(); j += 4) {
    auto x0 = input_data[j];
    auto y0 = input_data[j + 1];
    auto x1w = input_data[j + 2];
    auto y1h = input_data[j + 3];
    // (x1w, y1h) contain (x1, y1) for LTRB representation and (W, H) otherwise

    x0 = x0 * scale + ofsx;
    y0 = y0 * scale + ofsy;
    if (use_ltrb_) {
      x1w = x1w * scale + ofsx;
      y1h = y1h * scale + ofsy;
    } else {
      x1w = x1w * scale;
      y1h = y1h * scale;
    }

    output_data[j] = x0;
    output_data[j+1] = y0;
    output_data[j+2] = x1w;
    output_data[j+3] = y1h;
  }
}

DALI_REGISTER_OPERATOR(BBoxPaste, BBoxPaste<CPUBackend>, CPU);

}  // namespace dali
