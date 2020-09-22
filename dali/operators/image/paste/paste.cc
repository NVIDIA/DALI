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

#include "dali/operators/image/paste/paste.h"

namespace dali {

DALI_SCHEMA(Paste)
  .DocStr(R"code(Pastes the input images on a larger canvas, where the canvas size is equal to
``input size * ratio``.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("ratio",
      R"code(Ratio of canvas size to input size. Must be >= 1.)code",
      DALI_FLOAT, true)
  .AddOptionalArg("n_channels",
      R"code(Number of channels in the image.)code", 3)
  .AddArg("fill_value",
      R"code(Tuple of the values of the color that is used to fill the canvas.

The length of the tuple must be equal to ``n_channels``.)code",
      DALI_INT_VEC)
  .AddOptionalArg("paste_x",
      R"code(Horizontal position of the paste in (0.0 - 1.0) image coordinates.)code",
      0.5f, true)
  .AddOptionalArg("paste_y",
      R"code(Vertical position of the paste in (0.0 - 1.0) image coordinates.)code",
      0.5f, true)
  .AddOptionalArg("min_canvas_size",
      R"code(Enforces the minimum paste canvas dimension after scaling the input size
by the ratio.)code",
      0.0f, true)
  .InputLayout("HWC");

}  // namespace dali
