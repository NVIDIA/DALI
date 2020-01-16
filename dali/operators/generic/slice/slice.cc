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

#include "dali/operators/generic/slice/slice.h"

namespace dali {

DALI_SCHEMA(Slice)
    .DocStr(
        R"code(Extract a subtensor or `slice` with a given shape and anchor.
Inputs must be supplied as 3 separate tensors in a specific order: `data`, `anchor` and `shape`.
Both `anchor` and `shape` coordinates must be within the interval
[0.0, 1.0] for normalized coordinates, or within the image shape for absolute
coordinates. Both `anchor` and `shape` inputs will provide as many dimensions as specified
with arguments `axis_names` or `axes`. By default `Slice` operator uses normalized
coordinates and `WH` order for the slice arguments.)code")
    .NumInput(3)
    .NumOutput(1)
    .InputDox(0, "data", "TensorList", "Batch containing input data")
    .InputDox(1, "anchor", "1D TensorList of floats",
                 R"code(Input containing either normalized or absolute coordinates
(depending on the value of `normalized_anchor`) for the starting point of the
slice (x0, x1, x2, ...).)code")
    .InputDox(2, "shape", "1D TensorList of floats",
                 R"code(Input containing either normalized or absolute coordinates
(depending on the value of `normalized_shape`) for the dimensions of the slice
(s0, s1, s2, ...).)code")
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg("image_type",
      R"code(The color space of input and output image)code",
      DALI_RGB, false)
    .AddParent("SliceBase")
    .AddParent("SliceAttr");

template <>
void Slice<CPUBackend>::DataDependentSetup(SampleWorkspace &ws) {
  slice_attr_.ProcessArguments(ws);
  const auto &images = ws.Input<CPUBackend>(kImagesInId);
  auto data_idx = ws.data_idx();
  auto crop_window_generator = slice_attr_.GetCropWindowGenerator(data_idx);
  DALI_ENFORCE(crop_window_generator);
  auto layout = InputLayout(ws, 0);
  if (layout.empty())
    layout = GetDefaultLayout(images.shape().size());
  CropWindow win = crop_window_generator(images.shape(), layout);
  slice_shapes_[data_idx] = std::vector<int64_t>(win.shape.begin(), win.shape.end());
  slice_anchors_[data_idx] = std::vector<int64_t>(win.anchor.begin(), win.anchor.end());
}

template <>
void Slice<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  SliceBase<CPUBackend>::RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Slice, Slice<CPUBackend>, CPU);

}  // namespace dali
