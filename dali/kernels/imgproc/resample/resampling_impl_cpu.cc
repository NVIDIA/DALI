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

#include <cmath>
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"

namespace dali {
namespace kernels {

void InitializeResamplingFilter(
    int32_t *out_indices, float *out_coeffs, int out_size,
    float srcx_0, float scale, const ResamplingFilter &filter) {

  srcx_0 += 0.5f * scale - 0.5f - filter.anchor;
  int support = filter.support();

  for (int x = 0; x < out_size; x++) {
    float sx0f = x * scale + srcx_0;
    int sx0 = ceilf(sx0f);  // ceiling - below sx0f we assume the filter to be zero
    out_indices[x] = sx0;
    const float f0 = sx0 - sx0f;
    float sum = 0;
    int k = 0;
    for (int k = 0; k < support; k++) {
      float c = filter((f0 + k) * filter.scale);
      out_coeffs[support * x + k] = c;
      sum += c;
    }
    if (sum) {
      for (int k = 0; k < support; k++) {
        out_coeffs[support * x + k] /= sum;
      }
    }
  }
}

}  // namespace kernels
}  // namespace dali
