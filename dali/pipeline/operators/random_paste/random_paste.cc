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

#include "dali/pipeline/operators/random_paste/random_paste.h"

namespace dali {

DALI_SCHEMA(RandomPaste)
  .DocStr(R"code(Randomly paste the input image on a larger canvas.
          The larges canvas size is randomly chosen
          between [input_size; input_size * `max_ratio`.
            )code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddArg("max_ratio",
      R"code(`float`
      Value of the maximum ratio, must be > 1.)code")
  .AddArg("fill_color",
      R"code(`list of int`
      RGB value of the color to fill the canvas.
      `(r, g, b)` `int tuple` expected.)code");

}  // namespace dali
