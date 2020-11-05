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

#include <cstdint>

namespace dali {

namespace detail {

struct SampleDesc {
  uint64_t inner_vol, output_vol, inner_vol_classes;
  void *out = nullptr;
  const void *in = nullptr;
};

template <typename OutputType, typename InputType>
__global__ void PopulateOneHot(OutputType on_value, OutputType off_value,
                               const SampleDesc *samples) {
  uint64_t out_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto &sample = samples[blockIdx.y];
  if (out_index >= sample.output_vol) {
    return;
  }
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);
  uint64_t i = out_index / sample.inner_vol_classes;
  uint64_t j = out_index % sample.inner_vol;
  uint64_t in_index = i * sample.inner_vol + j;
  uint64_t in_val = in[in_index];
  uint64_t on_out_index = i * sample.inner_vol_classes + in_val * sample.inner_vol + j;
  out[out_index] = on_out_index == out_index ? on_value : off_value;
}

}  // namespace detail

}  // namespace dali
