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

namespace {

constexpr size_t kBlockSize = 256;

}  // namespace

__global__ void prt(const int16_t* ptr) {
  for (int i=0;i<100;i++) {
    printf("%d\t%d\n",i,ptr[i]);
  }
}

void Prt(const int16_t* ptr) {
  prt<<<1,1>>>(ptr);
}


__global__ void DecodeFlowComponentKernel(const int16_t *input, float *output, size_t n) {

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;idx<n; idx += blockDim.x * gridDim.x) {
    output[idx]
  }

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  output[idx] = decode_flow_component(input[idx]);
}

/**
 *
 * @param input
 * @param output
 * @param pitch width of buffer in bytes
 * @param width
 * @param height
 */
void DecodeFlowComponents(const int16_t *input, float *output, size_t pitch, size_t width, size_t height) {
//  size_t num_blocks = (num_values + kBlockSize - 1) / kBlockSize;
//  size_t block_size = min(num_values, kBlockSize);
  size_t num_threads = width;
  size_t num_blocks = height;
  DecodeFlowComponentKernel<<<num_blocks, num_threads>>>(input, output, width*height);
}

}  // namespace kernel
}  // namespace optical_flow
}  // namespace dali

