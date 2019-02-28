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

#include <utility>

#include "dali/pipeline/operators/crop/slice.h"

namespace dali {

template <>
void Slice<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws,
                                           unsigned int idx) {
  // Assumes xywh
  const auto &images = ws->Input<GPUBackend>(ws->NumInput() * idx);
  const auto &crop_begin = ws->Input<CPUBackend>(ws->NumInput() * idx + 1);
  const auto &crop_size = ws->Input<CPUBackend>(ws->NumInput() * idx + 2);

  for (int i = 0; i < batch_size_; i++) {
    const auto H = static_cast<int>(images.tensor_shape(i)[0]);
    const auto W = static_cast<int>(images.tensor_shape(i)[1]);

    per_sample_dimensions_[i] = std::make_pair(H, W);

    const auto crop_x = static_cast<int>(crop_begin.tensor<float>(i)[0] * W);
    const auto crop_y = static_cast<int>(crop_begin.tensor<float>(i)[1] * H);
    /*
     * To decrease floating point error, first calculate the bounding box of crop and then
     * calculate the width and height having left and top coordinates
     */
    auto crop_right_f = crop_size.tensor<float>(i)[0] + crop_begin.tensor<float>(i)[0];
    auto crop_bottom_f = crop_size.tensor<float>(i)[1] + crop_begin.tensor<float>(i)[1];
    auto crop_right = static_cast<int>(crop_right_f * W);
    auto crop_bottom = static_cast<int>(crop_bottom_f * H);

    crop_width_[i] = crop_right - crop_x;
    crop_height_[i] = crop_bottom - crop_y;

    per_sample_crop_[i] = std::make_pair(crop_y, crop_x);
  }
}

template <>
void Slice<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  DataDependentSetup(ws, static_cast<unsigned int>(idx));

  Crop<GPUBackend>::RunImpl(ws, idx);
}

template <>
void Slice<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  DALI_ENFORCE(ws->NumInput() == 3, "Expected 3 inputs. Received: " +
                                        std::to_string(ws->NumInput()));

  if (output_type_ == DALI_NO_TYPE) {
    const auto &input = ws->Input<GPUBackend>(0);
    output_type_ = input.type().id();
  }
}

DALI_REGISTER_OPERATOR(Slice, Slice<GPUBackend>, GPU);

}  // namespace dali
