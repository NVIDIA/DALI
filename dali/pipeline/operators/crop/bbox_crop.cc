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

#include "dali/pipeline/operators/crop/bbox_crop.h"

namespace dali {

DALI_SCHEMA(BBoxCrop)
  .DocStr(R"code(Perform a prospective crop from bounding boxes.)code")
  .NumInput(2)
  .NumOutput(3)
  .AllowMultipleInputSets()
  .AddOptionalArg("overlap_thresholds",
    R"code(Minimum overlap (IoU) in order not to discard bounding boxes after crop.
    Selected at random from provided values.)code",
    std::vector<float>{0.})
  .AddOptionalArg("random_aspect_ratio",
      R"code(Range from which to choose random aspect ratio.)code",
      std::vector<float>{1., 1.})
  .AddOptionalArg("crop_scaling",
    R"code(Range for which to choose the crop size.)code",
    std::vector<float>{1.,1.})
  .EnforceInputLayout(DALI_NHWC);

}  // namespace dali
