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
#include "dali/image/transform.h"
#include "dali/kernels/slice/slice_gpu.cuh"
#include "dali/core/static_switch.h"
#include "dali/operators/image/crop/crop.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <>
void Crop<GPUBackend>::DataDependentSetup(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);

  const TensorLayout in_layout = input.GetLayout();
  DALI_ENFORCE(in_layout.ndim() == input.shape().sample_dim());
  DALI_ENFORCE(ImageLayoutInfo::HasChannel(in_layout) &&
    (ImageLayoutInfo::IsImage(in_layout) || VideoLayoutInfo::IsVideo(in_layout)),
    "Unexpected data layout");
  TensorLayout out_layout = in_layout;

  for (int i = 0; i < batch_size_; ++i) {
    SetupSample(i, in_layout, input.tensor_shape(i));
  }
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(out_layout);
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<GPUBackend>, GPU);

}  // namespace dali
