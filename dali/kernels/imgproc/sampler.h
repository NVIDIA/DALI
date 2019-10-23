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
#include "dali/core/common.h"
#include "dali/core/traits.h"
#include "dali/core/convert.h"
#include "dali/core/math_util.h"
#include "dali/core/geom/vec.h"
#include "dali/core/host_dev.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/imgproc/surface.h"

namespace dali {
namespace kernels {

struct BorderClamp {};

template <DALIInterpType interp, typename In>
struct Sampler;

template <DALIInterpType interp, typename T>
DALI_HOST_DEV Sampler<interp, T> make_sampler(const Surface2D<T> &surface) {
  return Sampler<interp, T>(surface);
}

template <typename T>
DALI_HOST_DEV T GetBorderChannel(const T *values, int c) {
  return values[c];
}

template <typename T>
DALI_HOST_DEV constexpr
enable_if_t<!std::is_pointer<T>::value, T> GetBorderChannel(const T &value, int) {
  return value;
}

template <typename In>
struct Sampler<DALI_INTERP_NN, In> {
  Sampler() = default;
  DALI_HOST_DEV explicit Sampler(const Surface2D<const In> &surface)
  : surface(surface) {}

  Surface2D<const In> surface;

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV T at(
      int x, int y, int c,
      BorderValue border_value) const {
    if (x < 0 || x >= surface.size.x ||
        y < 0 || y >= surface.size.y) {
      return ConvertSat<T>(GetBorderChannel(border_value, c));
    } else {
      return ConvertSat<T>(surface(x, y, c));
    }
  }

  template <typename T = In>
  DALI_HOST_DEV T at(
      int x, int y, int c,
      BorderClamp) const {
    ivec2 clamped = clamp(ivec2(x, y), ivec2(0, 0), surface.size - 1);
    return ConvertSat<T>(surface(clamped, c));
  }

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV T at(
      float x, float y, int c,
      BorderValue border_value) const {
    return at<T>(floor_int(x), floor_int(y), c, border_value);
  }

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV T at(ivec2 xy, int c, BorderValue border_value) {
    return at(xy.x, xy.y, c, border_value);
  }

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV T at(vec2 xy, int c, BorderValue border_value) {
    return at(xy.x, xy.y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, int x, int y, int c, BorderValue border_value) const {
    pixel[c] = at<T>(x, y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, float x, float y, int c, BorderValue border_value) const {
    pixel[c] = at<T>(x, y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *pixel,
      int x, int y,
      BorderValue border_value) const {
    if (x < 0 || x >= surface.size.x ||
        y < 0 || y >= surface.size.y) {
      for (int c = 0; c < surface.channels; c++) {
        pixel[c] = GetBorderChannel(border_value, c);
      }
    } else {
      for (int c = 0; c < surface.channels; c++) {
        pixel[c] = ConvertSat<T>(surface(x, y, c));
      }
    }
  }

  template <typename T>
  DALI_HOST_DEV void operator()(
      T *pixel,
      int x, int y,
      BorderClamp) const {
    ivec2 clamped = clamp(ivec2(x, y), ivec2(0, 0), surface.size - 1);
    for (int c = 0; c < surface.channels; c++) {
      pixel[c] = ConvertSat<T>(surface(clamped, c));
    }
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *pixel,
      float x, float y,
      BorderValue border_value) const {
    (*this)(pixel, floor_int(x), floor_int(y), border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, ivec2 xy, int c, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, vec2 xy, int c, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, ivec2 xy, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, vec2 xy, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, border_value);
  }
};


template <typename In>
struct Sampler<DALI_INTERP_LINEAR, In> {
  Sampler() = default;
  DALI_HOST_DEV explicit Sampler(const Surface2D<const In> &surface)
  : surface(surface) {}

  Surface2D<const In> surface;

  template <typename T, typename BorderValue>
  DALI_HOST_DEV T at(
      float x, float y, int c,
      BorderValue border_value) const {
    Sampler<DALI_INTERP_NN, In> NN(surface);
    x -= 0.5f;
    y -= 0.5f;
    int x0 = floor_int(x);
    int y0 = floor_int(y);
    In s00 = NN.at(x0,   y0,   c, border_value);
    In s01 = NN.at(x0+1, y0,   c, border_value);
    In s10 = NN.at(x0,   y0+1, c, border_value);
    In s11 = NN.at(x0+1, y0+1, c, border_value);
    float qx = x - x0;
    float px = 1 - qx;
    float qy = y - y0;

    float s0 = s00 * px + s01 * qx;
    float s1 = s10 * px + s11 * qx;
    return ConvertSat<T>(s0 + (s1 - s0) * qy);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV T at(vec2 xy, int c) {
    return at(xy.x, xy.y, c);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *out_pixel,
      float x, float y, int c,
      BorderValue border_value) const {
    out_pixel[c] = at<T>(x, y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *out_pixel,
      float x, float y,
      BorderValue border_value) const {
    Sampler<DALI_INTERP_NN, In> NN(surface);
    x -= 0.5f;
    y -= 0.5f;
    int x0 = floor_int(x);
    int y0 = floor_int(y);
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
      out_pixel[c] = ConvertSat<T>(s0 + (s1 - s0) * qy);
    }
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *pixel,
      int x, int y, int c,
      BorderValue border_value) const {
    Sampler<DALI_INTERP_NN, In> s(surface);
    s(pixel, x, y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *pixel,
      int x, int y,
      BorderValue border_value) const {
    Sampler<DALI_INTERP_NN, In> s(surface);
    s(pixel, x, y, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, ivec2 xy, int c, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, vec2 xy, int c, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, ivec2 xy, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void
  operator()(T *pixel, vec2 xy, BorderValue border_value) const {
    (*this)(pixel, xy.x, xy.y, border_value);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_SAMPLER_H_
