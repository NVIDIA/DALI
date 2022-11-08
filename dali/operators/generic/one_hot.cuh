// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_ONE_HOT_CUH_
#define DALI_OPERATORS_GENERIC_ONE_HOT_CUH_

#include <cstdint>
#include <algorithm>
#include "dali/core/util.h"

namespace dali {
namespace one_hot {

struct SampleDesc {
  uint64_t inner_vol, output_vol, inner_vol_classes;
  void *out = nullptr;
  const void *in = nullptr;
};

template <typename OutputType, typename InputType>
__global__ void PopulateOneHot(OutputType on_value, OutputType off_value,
                               const SampleDesc *samples) {
  uint64_t out_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t grid_size = gridDim.x * blockDim.x;
  const auto &sample = samples[blockIdx.y];
  for (; out_index < sample.output_vol; out_index += grid_size) {
    auto *out = static_cast<OutputType*>(sample.out);
    auto *in = static_cast<const InputType*>(sample.in);
    uint64_t i = out_index / sample.inner_vol_classes;
    uint64_t j = out_index % sample.inner_vol;
    uint64_t in_index = i * sample.inner_vol + j;
    uint64_t in_val = in[in_index];
    uint64_t on_out_index = i * sample.inner_vol_classes + in_val * sample.inner_vol + j;
    out[out_index] = on_out_index == out_index ? on_value : off_value;
  }
}

dim3 gridHelper(uint64_t output_vol, int batch_size, int block = 256,
                uint64_t max_block_size = 65535) {
  auto block_size = std::min(div_ceil(output_vol, block), max_block_size);
  return dim3(block_size, batch_size);
}

}  // namespace one_hot
}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_ONE_HOT_CUH_
