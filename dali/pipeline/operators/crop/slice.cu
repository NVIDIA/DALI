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

template <>
void Slice<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws,
                                           unsigned int idx) {
  // Assumes xywh
  const auto &input = ws->Input<GPUBackend>(idx);

  // Need to copy to CPU as these values are required to perform other
  // calculations. Far from ideal, and should be done properly once Crop op
  // is cleaned up
  TensorList<CPUBackend> begin;
  begin.Copy(ws->Input<GPUBackend>(idx + 1), ws->stream());

  TensorList<CPUBackend> size;
  size.Copy(ws->Input<GPUBackend>(idx + 2), ws->stream());

  for (int i = 0; i < batch_size_; i++) {
    auto H = static_cast<int>(input.tensor_shape(i)[0]);
    auto W = static_cast<int>(input.tensor_shape(i)[1]);

    crop_width_[i] = static_cast<int>(size.template data<float>()[i * 2]);
    crop_height_[i] = static_cast<int>(size.template data<float>()[i * 2 + 1]);

    per_sample_dimensions_[i] = std::make_pair(H, W);

    auto crop_x = static_cast<int>(begin.template data<float>()[i * 2]);
    auto crop_y = static_cast<int>(begin.template data<float>()[i * 2 + 1]);

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
