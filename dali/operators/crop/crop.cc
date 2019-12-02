// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/crop/crop.h"
#include <tuple>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(Crop)
    .DocStr(R"code(Crops image with a given window dimensions and window position (upper left corner).)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg(
        "image_type",
        R"code(The color space of input and output image)code",
        DALI_RGB, false)
    .AddParent("CropAttr")
    .AddParent("SliceBase");

template <>
void Crop<CPUBackend>::DataDependentSetup(SampleWorkspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);

  const TensorLayout in_layout = InputLayout(ws, 0);
  DALI_ENFORCE(in_layout.ndim() == input.shape().sample_dim());
  DALI_ENFORCE(ImageLayoutInfo::HasChannel(in_layout) &&
    (ImageLayoutInfo::IsImage(in_layout) || VideoLayoutInfo::IsVideo(in_layout)),
    "Unexpected data layout");
  TensorLayout out_layout = in_layout;

  auto data_idx = ws.data_idx();
  SetupSample(data_idx, in_layout, input.shape());

  auto &output = ws.Output<CPUBackend>(0);
  output.SetLayout(out_layout);
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<CPUBackend>, CPU);

}  // namespace dali
