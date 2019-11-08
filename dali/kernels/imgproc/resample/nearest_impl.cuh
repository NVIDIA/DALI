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
#include "dali/core/convert.h"
#include "dali/core/math_util.h"
#include "dali/core/geom/box.h"

namespace dali {
namespace kernels {

template <typename Dst, typename Src>
__device__ void NNResample(
    Box<2, int> out_roi,
    vec2 origin, vec2 scale,
    Dst *__restrict__ out,  vec<1, ptrdiff_t> out_stride,
    const Src *__restrict__ in, vec<1, ptrdiff_t> in_stride, ivec2 in_size, int channels) {
  origin += 0.5f * scale;
  for (int i = out_roi.lo.y + threadIdx.y; i < out_roi.hi.y; i += blockDim.y) {
    int ysrc = floor_int(i * scale.y + origin.y);
    ysrc = clamp(ysrc, 0, in_size.y-1);

    Dst *out_row = &out[i * out_stride.x];
    const Src *in_row = &in[ysrc * in_stride.x];

    for (int j = out_roi.lo.x + threadIdx.x; j < out_roi.hi.x; j += blockDim.x) {
      int xsrc = floor_int(j * scale.x + origin.x);
      xsrc = clamp(xsrc, 0, in_size.x-1);
      const Src *src_px = &in_row[xsrc * channels];
      for (int c = 0; c < channels; c++)
        out_row[j*channels + c] = __ldg(&src_px[c]);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_NEAREST_IMPL_CUH_
