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

#include "dali/pipeline/operators/crop/new_slice.h"

namespace dali {

DALI_SCHEMA(NewSlice)
    .DocStr(
        R"code(Crop a slice of a defined `size` from an `input` tensor, staring
at the location specified by `begin`. Inputs must be supplied as 3 Tensors in a
specific order: `Images` containing image data in NHWC format, `Begin` containing
the starting pixel coordinates for the `crop` in `(x,y)` format, and `Size` containing
the pixel dimensions of the `crop` in `(w,h)` format. For both `Begin` and `Size`,
coordinates must be in the interval `[0.0, 1.0]`. The resulting tensor output of
Slice operation is a cropped version of the input tensor `Images`.
**Experimental** Use `Slice` instead)code")
    .NumInput(3)
    .NumOutput(1)
    .AllowSequences(false)
    .AddOptionalArg(
      "image_type",
      R"code(The color space of input and output image)code",
      DALI_RGB, false);

template <>
void NewSlice<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, int idx) {
  const auto &images = ws->Input<CPUBackend>(0);
  const auto &anchor_tensor = ws->Input<CPUBackend>(1);
  const auto &slice_shape_tensor = ws->Input<CPUBackend>(2);
  const auto shape = images.shape();
  const float* anchor_norm = anchor_tensor.template data<float>();
  const float* slice_shape_norm = slice_shape_tensor.template data<float>();
  SetupSample(ws->data_idx(), shape, anchor_norm, slice_shape_norm);
}

template <>
void NewSlice<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  SliceBase<CPUBackend>::RunImpl(ws, idx);
}

DALI_REGISTER_OPERATOR(NewSlice, NewSlice<CPUBackend>, CPU);

}  // namespace dali
