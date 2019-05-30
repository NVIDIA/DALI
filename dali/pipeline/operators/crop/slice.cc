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

#include "dali/pipeline/operators/crop/slice.h"

namespace dali {

DALI_SCHEMA(Slice)
    .DocStr(
        R"code(Extract a subtensor or `slice` with a given shape and anchor.
 Inputs must be supplied as 3 separate tensors in a specific order: `data`
 containing input data, `anchor` containing normalize coordinates for the
 starting point of the slice (x0, x1, x2, ...), and `shape` containing the normalized
 dimensions of the slice (s0, s1, s2, ...). Both `anchor` and `shape` coordinates
 must be in the interval [0.0, 1.0] and should have as many dimensions as the input
 data. For compatibility with the previous implementation of Slice, `anchor` and
 `slice` can be specified in format (x, y) and (w, h) respectively for images.
 This way of specifying the slice arguments is deprecated and shall be removed in
 future versions of DALI.)code")
    .NumInput(3)
    .NumOutput(1)
    .AllowSequences()
    .AllowMultipleInputSets()
    .AddOptionalArg(
      "image_type",
      R"code(The color space of input and output image)code",
      DALI_RGB, false)
    .AddParent("SliceBase");

template <>
void Slice<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, int idx) {
  const auto &images = ws->Input<CPUBackend>(3 * idx);
  const auto &anchor_tensor = ws->Input<CPUBackend>(3 * idx + 1);
  const auto &slice_shape_tensor = ws->Input<CPUBackend>(3 * idx + 2);
  const auto img_shape = images.shape();
  const auto args_ndim = anchor_tensor.shape()[0];
  const float* anchor_norm = anchor_tensor.template data<float>();
  const float* slice_shape_norm = slice_shape_tensor.template data<float>();
  SetupSample(ws->data_idx(), images.GetLayout(), img_shape, args_ndim,
              anchor_norm, slice_shape_norm);
}

template <>
void Slice<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  SliceBase<CPUBackend>::RunImpl(ws, idx);
}

DALI_REGISTER_OPERATOR(Slice, Slice<CPUBackend>, CPU);

}  // namespace dali
