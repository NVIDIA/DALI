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

#ifndef DALI_KERNELS_IMGPROC_JPEG_DCT_8x8_GPU_H_
#define DALI_KERNELS_IMGPROC_JPEG_DCT_8x8_GPU_H_

#include <cuda_runtime_api.h>

namespace dali {
namespace kernels {

// Implements optimized routines required for 8x8 DCT (forward and inverse)
// as described in the paper:
// https://docs.nvidia.com/cuda/samples/3_Imaging/dct8x8/doc/dct8x8.pdf


static constexpr float a = 1.387039845322148f; // sqrt(2) * cos(    pi / 16);
static constexpr float b = 1.306562964876377f; // sqrt(2) * cos(    pi /  8);
static constexpr float c = 1.175875602419359f; // sqrt(2) * cos(3 * pi / 16);
static constexpr float d = 0.785694958387102f; // sqrt(2) * cos(5 * pi / 16);
static constexpr float e = 0.541196100146197f; // sqrt(2) * cos(3 * pi /  8);
static constexpr float f = 0.275899379282943f; // sqrt(2) * cos(7 * pi / 16);
static constexpr float norm_factor = 0.3535533905932737f; // 1 / sqrt(8)

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

  float x07p = x0 + x7;
  float x16p = x1 + x6;
  float x25p = x2 + x5;
  float x34p = x3 + x4;

  float x07m = x0 - x7;
  float x61m = x6 - x1;
  float x25m = x2 - x5;
  float x43m = x4 - x3;

  float x07p34pp = x07p + x34p;
  float x07p34pm = x07p - x34p;
  float x16p25pp = x16p + x25p;
  float x16p25pm = x16p - x25p;

  x0 = norm_factor * (x07p34pp + x16p25pp);
  x2 = norm_factor * (b * x07p34pm + e * x16p25pm);
  x4 = norm_factor * (x07p34pp - x16p25pp);
  x6 = norm_factor * (e * x07p34pm - b * x16p25pm);

  x1 = norm_factor * (a * x07m - c * x61m + d * x25m - f * x43m);
  x3 = norm_factor * (c * x07m + f * x61m - a * x25m + d * x43m);
  x5 = norm_factor * (d * x07m + a * x61m + f * x25m - c * x43m);
  x7 = norm_factor * (f * x07m + d * x61m + c * x25m + a * x43m);

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

  float x04p   = x0 + x4;
  float x2b6ep = b * x2 + e * x6;

  float x04p2b6epp = x04p + x2b6ep;
  float x04p2b6epm = x04p - x2b6ep;
  float x7f1ap3c5dpp = f * x7 + a * x1 + c * x3 + d * x5;
  float x7a1fm3d5cmp = a * x7 - f * x1 + d * x3 - c * x5;

  float x04m   = x0 - x4;
  float x2e6bm = e * x2 - b * x6;

  float x04m2e6bmp = x04m + x2e6bm;
  float x04m2e6bmm = x04m - x2e6bm;
  float x1c7dm3f5apm = c * x1 - d * x7 - f * x3 - a * x5;
  float x1d7cp3a5fmm = d * x1 + c * x7 - a * x3 + f * x5;

  x0 = norm_factor * (x04p2b6epp + x7f1ap3c5dpp);
  x7 = norm_factor * (x04p2b6epp - x7f1ap3c5dpp);
  x4 = norm_factor * (x04p2b6epm + x7a1fm3d5cmp);
  x3 = norm_factor * (x04p2b6epm - x7a1fm3d5cmp);

  x1 = norm_factor * (x04m2e6bmp + x1c7dm3f5apm);
  x5 = norm_factor * (x04m2e6bmm - x1d7cp3a5fmm);
  x2 = norm_factor * (x04m2e6bmm + x1d7cp3a5fmm);
  x6 = norm_factor * (x04m2e6bmp - x1c7dm3f5apm);

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


#endif  // DALI_KERNELS_IMGPROC_JPEG_DCT_8x8_GPU_H_
