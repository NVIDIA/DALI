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
#include "dali/kernels/imgproc/resample/resampling_windows.h"
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"

namespace dali {
namespace kernels {

void InitializeFilter(
    int *out_indices, float *out_coeffs, int out_width,
    float sx0, float scale, const FilterWindow &filter) {

  sx0 += 0.5f * scale - scale * filter.anchor - 0.5f;
  int support = filter.support();

  for (int x = 0; x < out_width; x++) {
    float sfx0 = sx0 + x * scale;
    int sx0 = ceil(sfx0);
    out_indices[x] = sx0;
    float f0 = sx0 - sfx0;
    for (int k = 0; k < support; k++) {
      out_coeffs[support * x + k] = filter(f0 + k);
    }
  }
}

}  // namespace kernels
}  // namespace dali
