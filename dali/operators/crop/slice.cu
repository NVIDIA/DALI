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

#include <vector>
#include "dali/operators/crop/slice.h"

namespace dali {

template <>
void Slice<GPUBackend>::DataDependentSetup(DeviceWorkspace &ws) {
  SliceAttr::ProcessArguments(ws);
  const auto &images = ws.Input<GPUBackend>(kImagesInId);
  for (int data_idx = 0; data_idx < batch_size_; data_idx++) {
    const auto img_shape = images.tensor_shape(data_idx);
    auto crop_window_generator = SliceAttr::GetCropWindowGenerator(data_idx);
    DALI_ENFORCE(crop_window_generator);
    CropWindow win = crop_window_generator(img_shape);
    slice_shapes_[data_idx] = std::vector<int64_t>(win.shape.begin(), win.shape.end());
    slice_anchors_[data_idx] = std::vector<int64_t>(win.anchor.begin(), win.anchor.end());
  }
}

DALI_REGISTER_OPERATOR(Slice, Slice<GPUBackend>, GPU);

}  // namespace dali
