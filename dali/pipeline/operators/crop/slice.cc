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

#include "dali/pipeline/operators/crop/slice.h"

namespace dali {

DALI_SCHEMA(Slice)
    .DocStr(R"code(Crop as slice of a defined `size` from an `input` tensor, staring
    at the location specified by `begin`. Inputs must be supplied as 3 Tensors in a
    specific order: `Images` containing image data in NHWC format, `Begin` containing
    the starting pixel coordinates for the `crop` in `(x,y)` format, and 'Size' containing
    the pixel dimensions of the `crop` in `(w,h)` format. The resulting tensor output of
    Slice operation is a cropped version of the input tensor `Images`.)code")
    .NumInput(3)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .EnforceInputLayout(DALI_NHWC)
    .AddParent("Crop");

template <>
void Slice<CPUBackend>::DataDependentSetup(SampleWorkspace *ws) {
  // Assumes xywh. ltrb not supported atm
  const auto &input = ws->Input<CPUBackend>(0);
  const auto &begin = ws->Input<CPUBackend>(1);

  const int H = input.shape()[0];
  const int W = input.shape()[1];

  crop_y_norm_[ws->data_idx()] = static_cast<float>(begin.template data<float>()[1] / H);
  crop_x_norm_[ws->data_idx()] = static_cast<float>(begin.template data<float>()[0] / W);

  const auto &size = ws->Input<CPUBackend>(2);

  crop_width_[ws->data_idx()] = static_cast<int>(size.template data<float>()[0]);
  crop_height_[ws->data_idx()] = static_cast<int>(size.template data<float>()[1]);
}

template <>
void Slice<CPUBackend>::ThreadDependentSetup(SampleWorkspace *ws) {
  const auto &input = ws->Input<CPUBackend>(0);
  DALI_ENFORCE(input.shape().size() == 3,
              "Expects 3-dimensional image input.");

  const int H = input.shape()[0];
  const int W = input.shape()[1];

  per_sample_dimensions_[ws->thread_idx()] = std::make_pair(H, W);

  const int crop_y = crop_y_norm_[ws->data_idx()] * H;
  const int crop_x = crop_x_norm_[ws->data_idx()] * W;

  per_sample_crop_[ws->thread_idx()] = std::make_pair(crop_y, crop_x);
}

template <>
void Slice<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  // @pribalta: Order matters for DataDependent and ThreadDependent setups
  DataDependentSetup(ws);
  ThreadDependentSetup(ws);

  Crop<CPUBackend>::RunImpl(ws, idx);
}

template <>
void Slice<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  DALI_ENFORCE(ws->NumInput() == 3, "Expected 3 inputs. Received: " +
                                        std::to_string(ws->NumInput() == 3));

  if (output_type_ == DALI_NO_TYPE) {
    const auto &input = ws->Input<CPUBackend>(0);
    output_type_ = input.type().id();
  }
}

DALI_REGISTER_OPERATOR(Slice, Slice<CPUBackend>, CPU);

}  // namespace dali
