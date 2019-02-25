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

#include "dali/aux/optical_flow/turing_of/optical_flow_turing.h"

namespace dali {
namespace optical_flow {
namespace kernel {

constexpr size_t kFractionLength = 5;
constexpr size_t kBlockSize = 256;


__host__ __device__ float decode_flow_component(int16_t value) {
  constexpr float precision = 1.0f / (1 << kFractionLength);
  return (value < 0 ? -precision : precision) * (value & 0x7fff);
}


__global__ void DecodeFlowComponentKernel(const int16_t *input, float *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  output[idx] = decode_flow_component(input[idx]);
}


void DecodeFlowComponents(const int16_t *input, float *output, size_t num_values) {
  size_t num_blocks = (num_values + kBlockSize - 1) / kBlockSize;
  DecodeFlowComponentKernel<<<num_blocks, kBlockSize>>>(input, output);
}

}  // namespace kernel
}  // namespace optical_flow
}  // namespace dali

