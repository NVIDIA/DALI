// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/remap/warp_affine_params.h"
#include <cuda_runtime.h>

namespace dali {
namespace {

template <int ndims>
__global__ void InvertTransformsKernel(WarpAffineParams<ndims> *output,
                                       const WarpAffineParams<ndims> *input, int count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    output[i] = input[i].inv();
}

template <int ndims>
void InvertTransforms(WarpAffineParams<ndims> *output, const WarpAffineParams<ndims> *input,
                      int count, cudaStream_t stream) {
  int blocks = div_ceil(count, 512);
  int threads = std::min(count, 512);
  InvertTransformsKernel<ndims><<<blocks, threads, 0, stream>>>(output, input, count);
}

}  // namespace


template <>
void InvertTransformsGPU<2>(WarpAffineParams<2> *output, const WarpAffineParams<2> *input,
                            int count, cudaStream_t stream) {
  InvertTransforms<2>(output, input, count, stream);
}

template <>
void InvertTransformsGPU<3>(WarpAffineParams<3> *output, const WarpAffineParams<3> *input,
                            int count, cudaStream_t stream) {
  InvertTransforms<3>(output, input, count, stream);
}

}  // namespace dali
