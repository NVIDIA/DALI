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


#include <vector>
#include <cmath>

#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operators/resize/random_resized_crop.h"
#include "dali/util/random_crop_generator.h"

namespace dali {

template<>
void RandomResizedCrop<GPUBackend>::BackendInit() {
  InitializeGPU(batch_size_, spec_.GetArgument<int>("minibatch_size"));
}

template<>
void RandomResizedCrop<GPUBackend>::RunImpl(DeviceWorkspace * ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");

  const int newH = size_[0];
  const int newW = size_[1];

  auto &output = ws->Output<GPUBackend>(idx);
  RunGPU(output, input, ws->stream());
}

template<>
void RandomResizedCrop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  auto &input = ws->Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");

  for (int i = 0; i < batch_size_; ++i) {
    const auto &input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
        "Expects 3-dimensional image input.");

    int H = input_shape[0];
    int W = input_shape[1];

    crops_[i] = GetCropWindowGenerator(i)(H, W);
  }
  CalcResamplingParams();
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<GPUBackend>, GPU);

}  // namespace dali
