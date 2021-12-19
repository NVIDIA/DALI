// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
void SourceInfo<CPUBackend>::FillOutput(workspace_t<CPUBackend>& ws) {
  const auto& input = ws.template Input<CPUBackend>(0);
  auto& output = ws.template Output<CPUBackend>(0);
  for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto si = GetSourceInfo(input, sample_id);
    output[sample_id].Copy(make_cspan((const uint8_t*)si.c_str(), si.length()), nullptr);
  }
}

template <>
void Layout<CPUBackend>::FillOutput(workspace_t<CPUBackend>& ws) {
  const auto& input = ws.template Input<CPUBackend>(0);
  auto& output = ws.template Output<CPUBackend>(0);
  for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto layout = GetLayout(input, sample_id);
    output[sample_id].Copy(
        make_cspan(reinterpret_cast<const uint8_t*>(layout.c_str()), layout.size()), nullptr);
  }
}

template <>
void SourceInfo<GPUBackend>::FillOutput(workspace_t<GPUBackend>& ws) {
  const auto& input = ws.template Input<GPUBackend>(0);
  auto& output = ws.template Output<GPUBackend>(0);
  for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto si = GetSourceInfo(input, sample_id);
    auto output_ptr = output.raw_mutable_tensor(static_cast<int>(sample_id));
    cudaMemcpyAsync(output_ptr, si.c_str(), si.length(), cudaMemcpyDefault, ws.stream());
  }
}

template <>
void Layout<GPUBackend>::FillOutput(workspace_t<GPUBackend>& ws) {
  const auto& input = ws.template Input<GPUBackend>(0);
  auto& output = ws.template Output<GPUBackend>(0);
  for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto layout = GetLayout(input, sample_id);
    auto output_ptr = output.raw_mutable_tensor(static_cast<int>(sample_id));
    cudaMemcpyAsync(output_ptr, layout.c_str(), layout.size(), cudaMemcpyDefault, ws.stream());
  }
}

}  // namespace tensor_property
}  // namespace dali
