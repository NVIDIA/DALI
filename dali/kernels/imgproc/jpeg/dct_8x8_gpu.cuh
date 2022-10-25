// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_JPEG_DCT_8X8_GPU_CUH_
#define DALI_KERNELS_IMGPROC_JPEG_DCT_8X8_GPU_CUH_

#include <cuda_runtime_api.h>

namespace dali {
namespace kernels {

// Implements optimized routines required for 8x8 DCT (forward and inverse)
// as described in the paper:
// https://docs.nvidia.com/cuda/samples/3_Imaging/dct8x8/doc/dct8x8.pdf


static constexpr float a = 1.387039845322148f;             // sqrt(2) * cos(    pi / 16);
static constexpr float b = 1.306562964876377f;             // sqrt(2) * cos(    pi /  8);
static constexpr float c = 1.175875602419359f;             // sqrt(2) * cos(3 * pi / 16);
static constexpr float d = 0.785694958387102f;             // sqrt(2) * cos(5 * pi / 16);
static constexpr float e = 0.541196100146197f;             // sqrt(2) * cos(3 * pi /  8);
static constexpr float f = 0.275899379282943f;             // sqrt(2) * cos(7 * pi / 16);
static constexpr float norm_factor = 0.3535533905932737f;  // 1 / sqrt(8)

template <int stride>
__inline__ __device__
void dct_fwd_8x8_1d(float* data) {
  float x0 = data[0 * stride];
  float x1 = data[1 * stride];
  float x2 = data[2 * stride];
  float x3 = data[3 * stride];
  float x4 = data[4 * stride];
  float x5 = data[5 * stride];
  float x6 = data[6 * stride];
  float x7 = data[7 * stride];

  float tmp0 = x0 + x7;
  float tmp1 = x1 + x6;
  float tmp2 = x2 + x5;
  float tmp3 = x3 + x4;

  float tmp4 = x0 - x7;
  float tmp5 = x6 - x1;
  float tmp6 = x2 - x5;
  float tmp7 = x4 - x3;

  float tmp8 = tmp0 + tmp3;
  float tmp9 = tmp0 - tmp3;
  float tmp10 = tmp1 + tmp2;
  float tmp11 = tmp1 - tmp2;

  x0 = norm_factor * (tmp8 + tmp10);
  x2 = norm_factor * (b * tmp9 + e * tmp11);
  x4 = norm_factor * (tmp8 - tmp10);
  x6 = norm_factor * (e * tmp9 - b * tmp11);

  x1 = norm_factor * (a * tmp4 - c * tmp5 + d * tmp6 - f * tmp7);
  x3 = norm_factor * (c * tmp4 + f * tmp5 - a * tmp6 + d * tmp7);
  x5 = norm_factor * (d * tmp4 + a * tmp5 + f * tmp6 - c * tmp7);
  x7 = norm_factor * (f * tmp4 + d * tmp5 + c * tmp6 + a * tmp7);

  data[0 * stride] = x0;
  data[1 * stride] = x1;
  data[2 * stride] = x2;
  data[3 * stride] = x3;
  data[4 * stride] = x4;
  data[5 * stride] = x5;
  data[6 * stride] = x6;
  data[7 * stride] = x7;
}

template <int stride>
__inline__ __device__
void dct_inv_8x8_1d(float *data) {
  float x0 = data[0 * stride];
  float x1 = data[1 * stride];
  float x2 = data[2 * stride];
  float x3 = data[3 * stride];
  float x4 = data[4 * stride];
  float x5 = data[5 * stride];
  float x6 = data[6 * stride];
  float x7 = data[7 * stride];

  float tmp0 = x0 + x4;
  float tmp1 = b * x2 + e * x6;

  float tmp2 = tmp0 + tmp1;
  float tmp3 = tmp0 - tmp1;
  float tmp4 = f * x7 + a * x1 + c * x3 + d * x5;
  float tmp5 = a * x7 - f * x1 + d * x3 - c * x5;

  float tmp6 = x0 - x4;
  float tmp7 = e * x2 - b * x6;

  float tmp8 = tmp6 + tmp7;
  float tmp9 = tmp6 - tmp7;
  float tmp10 = c * x1 - d * x7 - f * x3 - a * x5;
  float tmp11 = d * x1 + c * x7 - a * x3 + f * x5;

  x0 = norm_factor * (tmp2 + tmp4);
  x7 = norm_factor * (tmp2 - tmp4);
  x4 = norm_factor * (tmp3 + tmp5);
  x3 = norm_factor * (tmp3 - tmp5);

  x1 = norm_factor * (tmp8 + tmp10);
  x5 = norm_factor * (tmp9 - tmp11);
  x2 = norm_factor * (tmp9 + tmp11);
  x6 = norm_factor * (tmp8 - tmp10);

  data[0 * stride] = x0;
  data[1 * stride] = x1;
  data[2 * stride] = x2;
  data[3 * stride] = x3;
  data[4 * stride] = x4;
  data[5 * stride] = x5;
  data[6 * stride] = x6;
  data[7 * stride] = x7;
}

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_JPEG_DCT_8X8_GPU_CUH_
