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

#include <utility>
#include "dali/pipeline/operators/crop/slice.h"

namespace dali {

template <>
void Slice<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, int idx) {
  const auto &images = ws->Input<GPUBackend>(3 * idx);
  const auto &anchor_tensor = ws->Input<CPUBackend>(3 * idx + 1);
  const auto &slice_shape_tensor = ws->Input<CPUBackend>(3 * idx + 2);
  for (int sample_idx = 0; sample_idx < batch_size_; sample_idx++) {
    const auto img_shape = images.tensor_shape(sample_idx);
    const auto args_ndims = anchor_tensor.tensor_shape(sample_idx)[0];
    const float* anchor_norm = anchor_tensor.tensor<float>(sample_idx);
    const float* slice_shape_norm = slice_shape_tensor.tensor<float>(sample_idx);
    SetupSample(sample_idx, images.GetLayout(), img_shape, args_ndims,
                anchor_norm, slice_shape_norm);
  }
}

template <>
void Slice<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  SliceBase<GPUBackend>::RunImpl(ws, idx);
}

DALI_REGISTER_OPERATOR(Slice, Slice<GPUBackend>, GPU);

}  // namespace dali
