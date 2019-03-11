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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_FILTERS_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_FILTERS_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>
#ifndef __CUDACC__
#include <cmath>
#include <algorithm>
#endif

namespace dali {
namespace kernels {


struct ResamplingFilter {
  float *coeffs;
  int num_coeffs;
  float anchor;
  float scale;

  __host__ __device__ void rescale(float support) {
    float old_scale = scale;
    scale = (num_coeffs-1) / support;
    anchor = anchor * old_scale / scale;
  }

  __host__ __device__ int support() const {
    return ceilf((num_coeffs-1) / scale);
  }

  __host__ __device__ float operator()(float x) const {
    if (!(x > -1))  // negative and NaN arguments
      return 0;
    if (x >= num_coeffs)
      return 0;
#ifdef __CUDA_ARCH__
    int x0 = floorf(x);
    int x1 = x0 + 1;
    float d = x - x0;
    float f0 = x0 < 0 ? 0.0f : __ldg(coeffs + x0);
    float f1 = x1 >= num_coeffs ? 0.0f : __ldg(coeffs + x1);
#else
    int x0 = std::floor(x);
    int x1 = x0 + 1;
    float d = x - x0;
    float f0 = x0 < 0.0f ? 0 : coeffs[x0];
    float f1 = x1 >= num_coeffs ? 0.0f : coeffs[x1];
#endif
    return f0 + d * (f1 - f0);
  }
};

struct ResamplingFilters {
  std::unique_ptr<float, std::function<void(void*)>> filter_data;

  ResamplingFilter Cubic() const noexcept;
  ResamplingFilter Gaussian(float sigma) const noexcept;
  ResamplingFilter Lanczos3() const noexcept;
  ResamplingFilter Triangular(float radius) const noexcept;

  std::vector<ResamplingFilter> filters;
  ResamplingFilter &operator[](int index) noexcept {
    return filters[index];
  }

  const ResamplingFilter &operator[](int index) const noexcept {
    return filters[index];
  }
};

std::shared_ptr<ResamplingFilters> GetResamplingFilters();
std::shared_ptr<ResamplingFilters> GetResamplingFiltersCPU();

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_FILTERS_CUH_
