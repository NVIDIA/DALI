// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/util/property.h"
#include "dali/pipeline/data/backend.h"

namespace dali {
namespace tensor_property {

template <>
void SourceInfo<CPUBackend>::FillOutput(Workspace &ws) {
  const auto& input = ws.Input<CPUBackend>(0);
  auto& output = ws.Output<CPUBackend>(0);
  for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto si = GetSourceInfo(input, sample_id);
    std::memcpy(output.mutable_tensor<uint8_t>(sample_id), si.c_str(), si.length());
  }
}

template <>
void Layout<CPUBackend>::FillOutput(Workspace &ws) {
  const auto& input = ws.Input<CPUBackend>(0);
  auto& output = ws.Output<CPUBackend>(0);
  for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto layout = GetLayout(input, sample_id);
    std::memcpy(output.mutable_tensor<uint8_t>(sample_id), layout.c_str(), layout.size());
  }
}

template <>
void SourceInfo<GPUBackend>::FillOutput(Workspace &ws) {
  const auto& input = ws.Input<GPUBackend>(0);
  auto& output = ws.Output<GPUBackend>(0);
  for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto si = GetSourceInfo(input, sample_id);
    auto output_ptr = output.raw_mutable_tensor(sample_id);
    cudaMemcpyAsync(output_ptr, si.c_str(), si.length(), cudaMemcpyDefault, ws.stream());
  }
}

template <>
void Layout<GPUBackend>::FillOutput(Workspace &ws) {
  const auto& input = ws.Input<GPUBackend>(0);
  auto& output = ws.Output<GPUBackend>(0);
  for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto layout = GetLayout(input, sample_id);
    auto output_ptr = output.raw_mutable_tensor(sample_id);
    cudaMemcpyAsync(output_ptr, layout.c_str(), layout.size(), cudaMemcpyDefault, ws.stream());
  }
}

}  // namespace tensor_property
}  // namespace dali
