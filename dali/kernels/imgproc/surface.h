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
#include "dali/kernels/tensor_view.h"

namespace dali {
namespace kernels {

template <typename T>
struct Surface2D {
  T *data;
  int width, height, channels, pixel_stride, row_stride, channel_stride;
  __host__ __device__ constexpr T &operator()(int x, int y, int c = 0) const {
    return data[y * row_stride + x * pixel_stride + c * channel_stride];
  }

  /// @brief Provides implicit _reference_ cast to surface of type const T,
  ///        if T is not already const
  ///
  /// @remarks The template magic is a workaround to avoid conversion to self
  ///          when T is already const
  template <typename U = T,
            typename V = typename std::enable_if<!std::is_const<U>::value, const U>::type>
  __host__ __device__ operator Surface2D<V>&() {
    return *reinterpret_cast<Surface2D<V>*>(this);
  }

  /// @brief Provides implicit _reference_ cast to surface of type const T,
  ///        if T is not already const
  ///
  /// @remarks The template magic is a workaround to avoid conversion to self
  ///          when T is already const
  template <typename U = T,
            typename V = typename std::enable_if<!std::is_const<U>::value, const U>::type>
  __host__ __device__ constexpr operator const Surface2D<V>&() const {
    return *reinterpret_cast<const Surface2D<V>*>(this);
  }
};

template <typename T, typename Storage>
constexpr Surface2D<T> as_surface_HWC(const TensorView<Storage, T, 3> &t) {
  return { t.data,
    static_cast<int>(t.shape[1]),  // width
    static_cast<int>(t.shape[0]),  // height
    static_cast<int>(t.shape[2]),  // channels
    static_cast<int>(t.shape[2]),  // pixel stride
    static_cast<int>(t.shape[1]) * static_cast<int>(t.shape[2]),  // row stride
    1                              // channel stride - contiguous
  };
}

template <typename T, typename Storage>
constexpr Surface2D<T> as_surface_CHW(const TensorView<Storage, T, 3> &t) {
  return { t.data,
    static_cast<int>(t.shape[2]),  // width
    static_cast<int>(t.shape[1]),  // height
    static_cast<int>(t.shape[0]),  // channels
    1,                             // pixel stride - contiguous
    static_cast<int>(t.shape[2]),  // row stride
    static_cast<int>(t.shape[1]) * static_cast<int>(t.shape[2])  // channel stride - planes
  };
}

template <typename T, typename Storage>
constexpr Surface2D<T> as_surface(const TensorView<Storage, T, 2> &t) {
  return { t.data,
    static_cast<int>(t.shape[1]),  // width
    static_cast<int>(t.shape[0]),  // height
    1,                             // channels
    1,                             // pixel stride - contiguous
    static_cast<int>(t.shape[1]),  // row stride
    0                              // channel stride - irrelevant
  };
}

}  // namespace kernels
}  // namespace dali

#endif   // DALI_KERNELS_IMGPROC_SURFACE_H_
