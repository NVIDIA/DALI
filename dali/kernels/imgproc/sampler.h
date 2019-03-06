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

#ifndef DALI_KERNELS_IMGPROC_SAMPLER_H_
#define DALI_KERNELS_IMGPROC_SAMPLER_H_

#include <cmath>
#include "dali/common.h"
#include "dali/kernels/common/convert.h"
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/imgproc/surface.h"

namespace dali {
namespace kernels {

template <DALIInterpType interp, typename In>
struct Sampler;

template <DALIInterpType interp, typename T>
__host__ __device__ Sampler<interp, T> make_sampler(const Surface2D<T> &surface) {
  return Sampler<interp, T>(surface);
}

template <typename In>
struct Sampler<DALI_INTERP_NN, In> {
  Sampler() = default;
  __host__ __device__ explicit Sampler(const Surface2D<const In> &surface)
  : surface(surface) {}

  Surface2D<const In> surface;

  template <typename T = In>
  __host__ __device__ T at(
      int x, int y, int c,
      const In *border_value) const {
    if (x < 0 || x >= surface.width ||
        y < 0 || y >= surface.height) {
      return clamp<T>(border_value[c]);
    } else {
      return clamp<T>(surface(x, y, c));
    }
  }

  template <typename T = In>
  __host__ __device__ T at(
      float x, float y, int c,
      const In *border_value) const {
    return at<T>(
      static_cast<int>(floorf(x)),
      static_cast<int>(floorf(y)),
      c, border_value);
  }

  template <typename T>
  __host__ __device__ void
  operator()(T *pixel, int x, int y, int c, const In *border_value) const {
    pixel[c] = at(floorf(x), floorf(y), c, border_value);
  }

  template <typename T>
  __host__ __device__ void
  operator()(T *pixel, float x, float y, int c, const In *border_value) const {
    pixel[c] = at(x, y, c, border_value);
  }

  template <typename T>
  __host__ __device__ void operator()(
      T *pixel,
      int x, int y,
      const In *border_value) const {
    if (x < 0 || x >= surface.width ||
        y < 0 || y >= surface.height) {
      for (int c = 0; c < surface.channels; c++) {
        pixel[c] = border_value[c];
      }
    } else {
      for (int c = 0; c < surface.channels; c++) {
        pixel[c] = surface(x, y, c);
      }
    }
  }

  template <typename T>
  __host__ __device__ void operator()(
      T *pixel,
      float x, float y,
      const In *border_value) const {
    (*this)(pixel, static_cast<int>(floorf(x)), static_cast<int>(floorf(y)), border_value);
  }
};


template <typename In>
struct Sampler<DALI_INTERP_LINEAR, In> {
  Sampler() = default;
  __host__ __device__ explicit Sampler(const Surface2D<const In> &surface)
  : surface(surface) {}

  Surface2D<const In> surface;

  template <typename T>
  __host__ __device__ T at(
      float x, float y, int c,
      const In *border_value) const {
    Sampler<DALI_INTERP_NN, In> NN(surface);
    x -= 0.5f;
    y -= 0.5f;
    int x0 = floorf(x);
    int y0 = floorf(y);
    In s00 = NN.at(x0,   y0,   c, border_value);
    In s01 = NN.at(x0+1, y0,   c, border_value);
    In s10 = NN.at(x0,   y0+1, c, border_value);
    In s11 = NN.at(x0+1, y0+1, c, border_value);
    float qx = x - x0;
    float px = 1 - qx;
    float qy = y - y0;

    float s0 = s00 * px + s01 * qx;
    float s1 = s10 * px + s11 * qx;
    return kernels::clamp<T>(s0 + (s1 - s0) * qy);
  }

  template <typename T>
  __host__ __device__ void operator()(
      T *out_pixel,
      float x, float y, int c,
      const In *border_value) const {
    out_pixel[c] = at<T>(x, y, c, border_value);
  }

  template <typename T>
  __host__ __device__ void operator()(
      T *out_pixel,
      float x, float y,
      const In *border_value) const {
    Sampler<DALI_INTERP_NN, In> NN(surface);
    x -= 0.5f;
    y -= 0.5f;
    int x0 = floorf(x);
    int y0 = floorf(y);
    float qx = x - x0;
    float px = 1 - qx;
    float qy = y - y0;

    for (int c = 0; c < surface.channels; c++) {
      In s00 = NN.at(x0,   y0,   c, border_value);
      In s01 = NN.at(x0+1, y0,   c, border_value);
      In s10 = NN.at(x0,   y0+1, c, border_value);
      In s11 = NN.at(x0+1, y0+1, c, border_value);
      float s0 = s00 * px + s01 * qx;
      float s1 = s10 * px + s11 * qx;
      out_pixel[c] = kernels::clamp<T>(s0 + (s1 - s0) * qy);
    }
  }

  template <typename T>
  __host__ __device__ T operator()(
      T *pixel,
      int x, int y, int c,
      const In *border_value) const {
    return Sampler<DALI_INTERP_NN, In>(surface)(pixel, x, y, c, border_value);
  }

  template <typename T>
  __host__ __device__ In operator()(
      T *pixel,
      int x, int y,
      const In *border_value) const {
    return Sampler<DALI_INTERP_NN, In>(surface)(pixel, x, y, border_value);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_SAMPLER_H_
