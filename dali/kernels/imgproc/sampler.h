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

template <DALIInterpType interp, int spatial_ndim, typename In>
struct Sampler;

template <DALIInterpType interp, typename In>
using Sampler2D = Sampler<interp, 2, In>;

template <DALIInterpType interp, typename In>
using Sampler3D = Sampler<interp, 3, In>;


template <DALIInterpType interp, int ndim, typename T>
DALI_HOST_DEV
Sampler<interp, ndim, std::remove_const_t<T>> make_sampler(const Surface<ndim, T> &surface) {
  return Sampler<interp, ndim, std::remove_const_t<T>>(surface);
}

template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE T GetBorderChannel(const T *values, int c) {
  return values[c];
}

template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE constexpr
enable_if_t<!std::is_pointer<T>::value, T> GetBorderChannel(const T &value, int) {
  return value;
}

/**
 * @brief Checks if all coordinates in `coords` are positive and below the respective
 *        coordinate in `limit`.
 *
 * @remarks `limit` is expected to have non-negative coordinates - otherwise, the result
 *          is undefined.
 */
template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr bool all_in_range(ivec<n> coords, ivec<n> limit) {
  for (int i = 0; i < n; i++) {
    // if limit is non-negative, then reintepreting to unsigned
    // checks for negative coords as well, as they wrap around and become
    // larger than any non-negative integer
    if (static_cast<unsigned>(coords[i]) >= static_cast<unsigned>(limit[i]))
      return false;
  }
  return true;
}

/**
 * @brief Checks if all coordinates in `coords` are positive and below the respective
 *        coordinate in `limit`.
 *
 * @remarks `limit` is expected to have non-negative coordinates - otherwise, the result
 *          is undefined.
 */
DALI_HOST_DEV DALI_FORCEINLINE
constexpr bool all_in_range(ivec2 coords, ivec2 limit) {
  // if limit is non-negative, then reintepreting to unsigned
  // checks for negative coords as well, as they wrap around and become
  // larger than any non-negative integer
  return static_cast<unsigned>(coords.x) < static_cast<unsigned>(limit.x) &&
         static_cast<unsigned>(coords.y) < static_cast<unsigned>(limit.y);
}

/**
 * @brief Checks if all coordinates in `coords` are positive and below the respective
 *        coordinate in `limit`.
 *
 * @remarks `limit` is expected to have non-negative coordinates - otherwise, the result
 *          is undefined.
 */
DALI_HOST_DEV DALI_FORCEINLINE
constexpr bool all_in_range(ivec3 coords, ivec3 limit) {
  // if limit is non-negative, then reintepreting to unsigned
  // checks for negative coords as well, as they wrap around and become
  // larger than any non-negative integer
  return static_cast<unsigned>(coords.x) < static_cast<unsigned>(limit.x) &&
         static_cast<unsigned>(coords.y) < static_cast<unsigned>(limit.y) &&
         static_cast<unsigned>(coords.z) < static_cast<unsigned>(limit.z);
}

template <int _spatial_ndim, typename In>
struct Sampler<DALI_INTERP_NN, _spatial_ndim, In> {
  static constexpr int spatial_ndim = _spatial_ndim;

  using icoords = dali::ivec<spatial_ndim>;
  using fcoords = dali::vec<spatial_ndim>;

  Sampler() = default;
  DALI_HOST_DEV explicit Sampler(const Surface<spatial_ndim, const In> &surface)
  : surface(surface) {}

  Surface<spatial_ndim, const In> surface;

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE T at(icoords pos, int c, BorderValue border_value) const {
    if (all_in_range(pos, surface.size)) {
      return ConvertSat<T>(surface(pos, c));
    } else {
      return ConvertSat<T>(GetBorderChannel(border_value, c));
    }
  }

  template <typename T = In>
  DALI_HOST_DEV DALI_FORCEINLINE T at(icoords pos, int c, BorderClamp) const {
    icoords clamped = clamp(pos, icoords(0), surface.size - 1);
    return ConvertSat<T>(surface(clamped, c));
  }

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE T at(fcoords pos, int c, BorderValue border_value) const {
    return at<T>(floor_int(pos), c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *pixel,
      icoords pos,
      BorderValue border_value) const {
    if (all_in_range(pos, surface.size)) {
      for (int c = 0; c < surface.channels; c++) {
        pixel[c] = ConvertSat<T>(surface(pos, c));
      }
    } else {
      for (int c = 0; c < surface.channels; c++) {
        pixel[c] = GetBorderChannel(border_value, c);
      }
    }
  }

  template <typename T>
  DALI_HOST_DEV
  void operator()(T *pixel, icoords pos, BorderClamp) const {
    icoords clamped = clamp(pos, icoords(0), surface.size - 1);
    for (int c = 0; c < surface.channels; c++) {
      pixel[c] = ConvertSat<T>(surface(clamped, c));
    }
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE
  void operator()(T *pixel, fcoords pos, BorderValue border_value) const {
    operator()(pixel, floor_int(pos), border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE
  void operator()(T *pixel, icoords pos, int c, BorderValue border_value) const {
    pixel[c] = at<T>(pos, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE
  void operator()(T *pixel, fcoords pos, int c, BorderValue border_value) const {
    pixel[c] = at<T>(pos, c, border_value);
  }
};


template <typename In>
struct Sampler<DALI_INTERP_LINEAR, 2, In> {
  static constexpr int spatial_ndim = 2;
  using icoords = ivec2;
  using fcoords = vec2;

  Sampler() = default;
  DALI_HOST_DEV explicit Sampler(const Surface2D<const In> &surface)
  : surface(surface) {}

  Surface2D<const In> surface;

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV T at(ivec2 pos, int c, BorderValue border_value) const {
    Sampler2D<DALI_INTERP_NN, In> NN(surface);
    return NN.template at<T>(pos, c, border_value);
  }

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV T at(vec2 pos, int c, BorderValue border_value) const {
    Sampler2D<DALI_INTERP_NN, In> NN(surface);
    float x = pos.x - 0.5f;
    float y = pos.y - 0.5f;
    int x0 = floor_int(x);
    int y0 = floor_int(y);
    In s00 = NN.at(ivec2(x0,   y0),   c, border_value);
    In s01 = NN.at(ivec2(x0+1, y0),   c, border_value);
    In s10 = NN.at(ivec2(x0,   y0+1), c, border_value);
    In s11 = NN.at(ivec2(x0+1, y0+1), c, border_value);
    float qx = x - x0;
    float px = 1 - qx;
    float qy = y - y0;

    float s0 = s00 * px + s01 * qx;
    float s1 = s10 * px + s11 * qx;
    return ConvertSat<T>(s0 + (s1 - s0) * qy);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV
  void operator()(
      T *out_pixel, vec2 pos,
      BorderValue border_value) const {
    Sampler2D<DALI_INTERP_NN, In> NN(surface);
    float x = pos.x - 0.5f;
    float y = pos.y - 0.5f;
    int x0 = floor_int(x);
    int y0 = floor_int(y);
    float qx = x - x0;
    float px = 1 - qx;
    float qy = y - y0;
    for (int c = 0; c < surface.channels; c++) {
      In s00 = NN.at(ivec2(x0,   y0),   c, border_value);
      In s01 = NN.at(ivec2(x0+1, y0),   c, border_value);
      In s10 = NN.at(ivec2(x0,   y0+1), c, border_value);
      In s11 = NN.at(ivec2(x0+1, y0+1), c, border_value);
      float s0 = s00 * px + s01 * qx;
      float s1 = s10 * px + s11 * qx;
      out_pixel[c] = ConvertSat<T>(s0 + (s1 - s0) * qy);
    }
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE void operator()(
    T *out_pixel, vec2 pos, int c, BorderValue border_value) const {
    out_pixel[c] = at<T>(pos, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE void operator()(
    T *out_pixel, ivec2 pos, int c, BorderValue border_value) const {
    out_pixel[c] = at<T>(pos, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE void operator()(
    T *out_pixel, ivec2 pos, BorderValue border_value) const {
    Sampler2D<DALI_INTERP_NN, In> NN(surface);
    NN(out_pixel, pos, border_value);
  }
};

template <typename In>
struct Sampler<DALI_INTERP_LINEAR, 3, In> {
  static constexpr int spatial_ndim = 3;
  using icoords = ivec3;
  using fcoords = vec3;

  Sampler() = default;
  DALI_HOST_DEV explicit Sampler(const Surface3D<const In> &surface)
  : surface(surface) {}

  Surface3D<const In> surface;

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV T at(
      vec3 pos, int c, BorderValue border_value) const {
    Sampler3D<DALI_INTERP_NN, In> NN(surface);
    float x = pos.x - 0.5f;
    float y = pos.y - 0.5f;
    float z = pos.z - 0.5f;
    int x0 = floor_int(x);
    int y0 = floor_int(y);
    int z0 = floor_int(z);
    In s000 = NN.at(ivec3(x0,   y0,   z0),   c, border_value);
    In s001 = NN.at(ivec3(x0+1, y0,   z0),   c, border_value);
    In s010 = NN.at(ivec3(x0,   y0+1, z0),   c, border_value);
    In s011 = NN.at(ivec3(x0+1, y0+1, z0),   c, border_value);
    In s100 = NN.at(ivec3(x0,   y0,   z0+1), c, border_value);
    In s101 = NN.at(ivec3(x0+1, y0,   z0+1), c, border_value);
    In s110 = NN.at(ivec3(x0,   y0+1, z0+1), c, border_value);
    In s111 = NN.at(ivec3(x0+1, y0+1, z0+1), c, border_value);
    float qx = x - x0;
    float px = 1 - qx;
    float qy = y - y0;
    float py = 1 - qy;
    float qz = z - z0;

    float s00 = s000 * px + s001 * qx;
    float s01 = s010 * px + s011 * qx;
    float s10 = s100 * px + s101 * qx;
    float s11 = s110 * px + s111 * qx;

    float s0 = s00 * py + s01 * qy;
    float s1 = s10 * py + s11 * qy;
    return ConvertSat<T>(s0 + (s1 - s0) * qz);
  }

  template <typename T = In, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE
  T at(ivec3 pos, int c, BorderValue border_value) {
    Sampler3D<DALI_INTERP_NN, In> NN(surface);
    return NN.template at<T>(pos, c, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV void operator()(
      T *out_pixel,
      vec3 pos,
      BorderValue border_value) const {
    Sampler3D<DALI_INTERP_NN, In> NN(surface);
    float x = pos.x - 0.5f;
    float y = pos.y - 0.5f;
    float z = pos.z - 0.5f;
    int x0 = floor_int(x);
    int y0 = floor_int(y);
    int z0 = floor_int(z);
    float qx = x - x0;
    float px = 1 - qx;
    float qy = y - y0;
    float py = 1 - qy;
    float qz = z - z0;

    for (int c = 0; c < surface.channels; c++) {
      In s000 = NN.at(ivec3(x0,   y0,   z0),   c, border_value);
      In s001 = NN.at(ivec3(x0+1, y0,   z0),   c, border_value);
      In s010 = NN.at(ivec3(x0,   y0+1, z0),   c, border_value);
      In s011 = NN.at(ivec3(x0+1, y0+1, z0),   c, border_value);
      In s100 = NN.at(ivec3(x0,   y0,   z0+1), c, border_value);
      In s101 = NN.at(ivec3(x0+1, y0,   z0+1), c, border_value);
      In s110 = NN.at(ivec3(x0,   y0+1, z0+1), c, border_value);
      In s111 = NN.at(ivec3(x0+1, y0+1, z0+1), c, border_value);

      float s00 = s000 * px + s001 * qx;
      float s01 = s010 * px + s011 * qx;
      float s10 = s100 * px + s101 * qx;
      float s11 = s110 * px + s111 * qx;

      float s0 = s00 * py + s01 * qy;
      float s1 = s10 * py + s11 * qy;
      out_pixel[c] = ConvertSat<T>(s0 + (s1 - s0) * qz);
    }
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE void operator()(
    T *out_pixel, ivec3 pos, BorderValue border_value) const {
    Sampler3D<DALI_INTERP_NN, In> NN(surface);
    NN(out_pixel, pos, border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE void operator()(
    T *out_pixel, int x, int y, int z, BorderValue border_value) const {
    Sampler3D<DALI_INTERP_NN, In> NN(surface);
    NN(out_pixel, ivec3(x, y, z), border_value);
  }

  template <typename T, typename BorderValue>
  DALI_HOST_DEV DALI_FORCEINLINE void operator()(
    T *out_pixel, ivec3 pos, int c, BorderValue border_value) const {
    Sampler3D<DALI_INTERP_NN, In> NN(surface);
    NN(out_pixel, pos, c, border_value);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_SAMPLER_H_
