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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_

#include <cuda_runtime_api.h>
#include "dali/core/geom/vec.h"
#include "dali/core/geom/mat.h"
#include "dali/core/convert.h"

namespace dali {
namespace kernels {

namespace detail {

template <int N, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE vec<N, float> norm(vec<N, Input> x) {
  vec<N, float> out;
  for (int i = 0; i < N; i++)
    out[i] = ConvertNorm<float>(x[i]);
  return out;
}

template <int N>
DALI_HOST_DEV DALI_FORCEINLINE vec<N, float> norm(vec<N, float> x) {
  return x;
}

}  // namespace detail

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_y(vec<3, Input> rgb_in) {
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  constexpr vec3 coeffs(0.299f, 0.587f, 0.114f);
  return ConvertSatNorm<Output>(dot(coeffs, rgb));
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cb(vec<3, Input> rgb_in) {
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  constexpr vec3 coeffs(-0.16873589f, -0.33126411f, 0.5f);
  return ConvertSatNorm<Output>(dot(coeffs, rgb) + 0.5f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cr(vec<3, Input> rgb_in) {
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  constexpr vec3 coeffs(0.5f, -0.41868759f, -0.08131241f);
  return ConvertSatNorm<Output>(dot(coeffs, rgb) + 0.5f);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
