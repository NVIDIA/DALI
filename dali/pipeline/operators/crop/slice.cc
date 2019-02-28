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
    .DocStr(
        R"code(Crop a slice of a defined `size` from an `input` tensor, staring
at the location specified by `begin`. Inputs must be supplied as 3 Tensors in a
specific order: `Images` containing image data in NHWC format, `Begin` containing
the starting pixel coordinates for the `crop` in `(x,y)` format, and `Size` containing
the pixel dimensions of the `crop` in `(w,h)` format. For both `Begin` and `Size`,
coordinates must be in the interval `[0.0, 1.0]`. The resulting tensor output of
Slice operation is a cropped version of the input tensor `Images`.)code")
    .NumInput(3)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AllowSequences()
    .AddOptionalArg(
        "image_type",
        R"code(The color space of input and output image)code",
        DALI_RGB, false)
    .EnforceInputLayout(DALI_NHWC);

template <>
void Slice<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, unsigned int) {
  // Assumes xywh
  const auto &images = ws->Input<CPUBackend>(0);
  const auto &crop_begin = ws->Input<CPUBackend>(1);

  const auto H = static_cast<const int>(images.shape()[0]);
  const auto W = static_cast<const int>(images.shape()[1]);

  const auto &crop_size = ws->Input<CPUBackend>(2);

  per_sample_dimensions_[ws->thread_idx()] = std::make_pair(H, W);

  const auto crop_x = static_cast<const int>(crop_begin.template data<float>()[0] * W);
  const auto crop_y = static_cast<const int>(crop_begin.template data<float>()[1] * H);

  /*
   * To decrease floating point error, first calculate the bounding box of crop and then
   * calculate the width and height having left and top coordinates
   */
  auto crop_right_f = crop_size.template data<float>()[0] + crop_begin.template data<float>()[0];
  auto crop_bottom_f = crop_size.template data<float>()[1] + crop_begin.template data<float>()[1];
  auto crop_right = static_cast<const int>(crop_right_f * W);
  auto crop_bottom = static_cast<const int>(crop_bottom_f * H);

  crop_width_[ws->data_idx()] = crop_right - crop_x;
  crop_height_[ws->data_idx()] = crop_bottom - crop_y;

  per_sample_crop_[ws->thread_idx()] = std::make_pair(crop_y, crop_x);
}

template <>
void Slice<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  DataDependentSetup(ws);

  Crop<CPUBackend>::RunImpl(ws, idx);
}

template <>
void Slice<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  DALI_ENFORCE(ws->NumInput() == 3, "Expected 3 inputs. Received: " +
                                        std::to_string(ws->NumInput()));

  if (output_type_ == DALI_NO_TYPE) {
    const auto &input = ws->Input<CPUBackend>(0);
    output_type_ = input.type().id();
  }
}

DALI_REGISTER_OPERATOR(Slice, Slice<CPUBackend>, CPU);

}  // namespace dali
