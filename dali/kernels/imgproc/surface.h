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

#ifndef DALI_KERNELS_IMGPROC_SURFACE_H_
#define DALI_KERNELS_IMGPROC_SURFACE_H_

#include <cuda_runtime.h>
#include <type_traits>
#include "dali/core/tensor_view.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {


template <int _spatial_ndim, typename T>
struct Surface {
  static constexpr int spatial_ndim = _spatial_ndim;

  constexpr Surface() = default;

  template <int N = spatial_ndim, std::enable_if_t<N == 2, int> = 0>
  DALI_HOST_DEV
  constexpr Surface(T *data,
                    int width, int height, int channels,
                    int64_t pixel_stride, int64_t row_stride, int64_t channel_stride)
  : data(data), size(width, height), channels(channels),
    strides(pixel_stride, row_stride), channel_stride(channel_stride) {
  }

  template <int N = spatial_ndim, std::enable_if_t<N == 3, int> = 0>
  DALI_HOST_DEV
  constexpr Surface(T *data,
                    int width, int height, int depth, int channels,
                    int64_t pixel_stride, int64_t row_stride, int64_t slice_stride,
                    int64_t channel_stride)
  : data(data), size(width, height, depth), channels(channels),
    strides(pixel_stride, row_stride, slice_stride), channel_stride(channel_stride) {
  }

  DALI_HOST_DEV
  constexpr Surface(T *data,
                    ivec<spatial_ndim> size, int channels,
                    i64vec<spatial_ndim> strides, int64_t channel_stride)
  : data(data), size(size), channels(channels), strides(strides), channel_stride(channel_stride) {
  }

  T *data;
  ivec<spatial_ndim> size;
  int channels;
  i64vec<spatial_ndim> strides;
  int64_t channel_stride;

  DALI_HOST_DEV constexpr i64vec<spatial_ndim + 1> strides_ch() const {
    return cat(strides, channel_stride);
  }

  template <int N = spatial_ndim>
  DALI_HOST_DEV
  constexpr std::enable_if_t<N == 1, T &> operator()(int x, int c = 0) const {
    static_assert(N == spatial_ndim, "N mustn't be explicitly set");
    return (*this)({x, c});
  }

  template <int N = spatial_ndim>
  DALI_HOST_DEV
  constexpr std::enable_if_t<N == 2, T &> operator()(int x, int y, int c = 0) const {
    static_assert(N == spatial_ndim, "N mustn't be explicitly set");
    return (*this)({x, y, c});
  }

  template <int N = spatial_ndim>
  DALI_HOST_DEV
  constexpr std::enable_if_t<N == 3, T &> operator()(int x, int y, int z, int c = 0) const {
    static_assert(N == spatial_ndim, "N mustn't be explicitly set");
    return (*this)({x, y, z, c});
  }

  DALI_HOST_DEV constexpr T &operator()(ivec<spatial_ndim> pos, int c = 0) const {
    return (*this)(cat(pos, c));
  }

  DALI_HOST_DEV constexpr T &operator()(ivec<spatial_ndim + 1> pos_and_channel) const {
    return data[dot(pos_and_channel, this->strides_ch())];
  }

  DALI_HOST_DEV constexpr Surface<spatial_ndim - 1, T> slice(int outermost_pos = 0) const {
    return {
      &data[strides[spatial_ndim-1] * outermost_pos],
      sub<spatial_ndim - 1>(size),
      channels,
      sub<spatial_ndim - 1>(strides),
      channel_stride
    };
  }

  /**
   * @brief Provides implicit _reference_ cast to surface of type const T,
   *        if T is not already const
   *
   * @remarks The template magic is a workaround to avoid conversion to self
   *          when T is already const
   */
  template <typename U = T,
            typename V = std::enable_if_t<!std::is_const<U>::value, const U>>
  DALI_HOST_DEV operator Surface<spatial_ndim, V>&() {
    return *reinterpret_cast<Surface<spatial_ndim, V>*>(this);
  }

  /**
   * @brief Provides implicit _reference_ cast to surface of type const T,
   *        if T is not already const
   *
   * @remarks The template magic is a workaround to avoid conversion to self
   *          when T is already const
   */
  template <typename U = T,
            typename V = std::enable_if_t<!std::is_const<U>::value, const U>>
  DALI_HOST_DEV constexpr operator const Surface<spatial_ndim, V>&() const {
    return *reinterpret_cast<const Surface<spatial_ndim, V>*>(this);
  }
};

template <typename T>
using Surface2D = Surface<2, T>;

template <typename T>
using Surface3D = Surface<3, T>;

template <int channel_dim = -1, int n, typename T, typename Storage>
DALI_HOST_DEV
constexpr Surface<(channel_dim < 0 ? n : n-1), T> as_surface(const TensorView<Storage, T, n> &t) {
  const int spatial_ndim = (channel_dim < 0 ? n : n-1);
  ivec<spatial_ndim> size = shape2vec(skip_dim<channel_dim>(t.shape));
  i64vec<spatial_ndim> strides = {};
  int64_t stride = 1;
  int64_t channel_stride = 0;
  int channels = 1;
  for (int d = n - 1, i = 0; d >=0; d--) {
    if (d == channel_dim) {
      channel_stride = stride;
      channels = t.shape[d];
    } else {
      strides[i++] = stride;
    }
    stride *= t.shape[d];
  }

  return { t.data, size, channels, strides, channel_stride };
}

template <int n, typename T, typename Storage>
DALI_HOST_DEV
constexpr Surface<n-1, T> as_surface_channel_first(const TensorView<Storage, T, n> &t) {
  return as_surface<0>(t);
}

template <int n, typename T, typename Storage>
DALI_HOST_DEV
constexpr Surface<n-1, T> as_surface_channel_last(const TensorView<Storage, T, n> &t) {
  return as_surface<n-1>(t);
}

template <typename T, typename Storage>
DALI_HOST_DEV
constexpr Surface2D<T> as_surface_HWC(const TensorView<Storage, T, 3> &t) {
  return as_surface_channel_last(t);
}

template <typename T, typename Storage>
DALI_HOST_DEV
constexpr Surface2D<T> as_surface_CHW(const TensorView<Storage, T, 3> &t) {
  return as_surface_channel_first(t);
}

/**
 * Crops Surface according to given Roi
 */
template <int n, typename T>
DALI_HOST_DEV constexpr Surface<n, T> crop(const Surface<n, T> &surface, const Roi<n> &roi) {
  auto cropped = surface;
  cropped.data = &surface(roi.lo);
  cropped.size = roi.extent();
  return cropped;
}

}  // namespace kernels
}  // namespace dali

#endif   // DALI_KERNELS_IMGPROC_SURFACE_H_
