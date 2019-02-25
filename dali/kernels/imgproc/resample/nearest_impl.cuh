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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_NEAREST_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_NEAREST_IMPL_CUH_

#include <cuda_runtime.h>
#ifndef __CUDACC__
#include <algorithm>
#endif
#include "dali/kernels/common/convert.h"

namespace dali {
namespace kernels {

template <typename Dst, typename Src>
__device__ void NNResample(
    int x0, int x1, int y0, int y1,
    float src_x0, float src_y0, float scale_x, float scale_y,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_w, int in_h, int channels) {
  src_y0 += 0.5f * scale_y;
  src_x0 += 0.5f * scale_x;
  for (int i = y0 + threadIdx.y; i < y1; i += blockDim.y) {
    int ysrc = i * scale_y + src_y0;
    ysrc = min(max(0, ysrc), in_h-1);

    Dst *out_row = &out[i * out_stride];
    const Src *in_row = &in[ysrc * in_stride];

    for (int j = x0 + threadIdx.x; j < x1; j += blockDim.x) {
      int xsrc = j * scale_x + src_x0;
      xsrc = min(max(0, xsrc), in_w-1);
      const Src *src_px = &in_row[xsrc * channels];
      for (int c = 0; c < channels; c++)
        out_row[j*channels + c] = __ldg(&src_px[c]);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_NEAREST_IMPL_CUH_
