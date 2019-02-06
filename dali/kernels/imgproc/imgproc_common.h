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

#ifndef DALI_KERNELS_IMGPROC_IMGPROC_COMMON_H_
#define DALI_KERNELS_IMGPROC_IMGPROC_COMMON_H_


#include <cuda_runtime.h>
#include "dali/kernels/tensor_view.h"

namespace dali {
namespace kernels {

constexpr int AnchorCenter = 0x7fffffff;

enum class BorderMode {
  Clamp, Mirror
};

struct ConvolutionFilter {
  std::vector<float> coeffs;
  int anchor = AnchorCenter;
  int size() const { return coeffs.size(); }
  float operator()(int i) const {
    return coeffs[i - (anchor ? anchor : size()/2)];
  }
};

template <int channels, typename T>
struct Pixel {
  T data[channels];
  __host__ __device__ T &operator[](int index) { return data[index]; }
  __host__ __device__ const T &operator[](int index) const { return data[index]; }
};
}  // kernels
}  // dali

#endif  // DALI_KERNELS_IMGPROC_IMGPROC_COMMON_H_
