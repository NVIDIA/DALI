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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_WINDOWS_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_WINDOWS_H_

#include <cuda_runtime.h>
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

inline __host__ __device__ float GaussianWindow(float x, float sigma) {
  x /= sigma;
  return expf(-0.5f * x * x);
}

inline __host__ __device__ float TriangularWindow(float x) {
  float t = 1 - fabsf(x);
  return t < 0 ? 0 : t;
}

inline __host__ __device__ float sinc(float x) {
  return x ? sinf(x * M_PI) / (x * M_PI) : 1;
}

inline __host__ __device__ float LanczosWindow(float x, float a) {
  if (fabsf(x) >= a)
    return 0.0f;
  return sinc(x)*sinc(x / a);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_WINDOWS_H_
