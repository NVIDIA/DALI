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
namespace color {

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

// Y, Cb, Cr definition from ITU-R BT.601, with values in the range 16-235, allowing for
// footroom and headroom
namespace itu_r_bt_601 {

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_y(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs(0.257f, 0.504f, 0.098f);
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  return ConvertSatNorm<Output>(dot(coeffs, rgb) + 0.0625f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE uint8_t rgb_to_y(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.257f, 0.504f, 0.098f);
  return static_cast<uint8_t>(dot(coeffs, rgb) + 16.0f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cb(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs(-0.148f, -0.291f, 0.439f);
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  return ConvertSatNorm<Output>(dot(coeffs, rgb) + 0.5f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE uint8_t rgb_to_cb(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(-0.148f, -0.291f, 0.439f);
  if (rgb.x == rgb.y && rgb.x == rgb.z) return 128;
  return static_cast<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cr(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs(0.439f, -0.368f, -0.071f);
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  return ConvertSatNorm<Output>(dot(coeffs, rgb) + 0.5f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE uint8_t rgb_to_cr(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.439f, -0.368f, -0.071f);
  if (rgb.x == rgb.y && rgb.x == rgb.z) return 128;
  return static_cast<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr_in) {
  auto ycbcr = detail::norm(ycbcr_in);  // TODO(janton): optimize number of multiplications
  float tmp_y = 1.164f * (ycbcr.x - 0.0625f);
  float tmp_b = ycbcr.y - 0.5f;
  float tmp_r = ycbcr.z - 0.5f;
  auto r = ConvertSatNorm<Output>(tmp_y + 1.596f * tmp_r);
  auto g = ConvertSatNorm<Output>(tmp_y - 0.813f * tmp_r - 0.392f * tmp_b, 0, 255);
  auto b = ConvertSatNorm<Output>(tmp_y + 2.017f * tmp_b);
  return {r, g, b};
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, uint8_t> ycbcr_to_rgb(vec<3, uint8_t> ycbcr) {
  float tmp_y = 1.164f * (ycbcr.x - 16);
  float tmp_b = ycbcr.y - 128;
  float tmp_r = ycbcr.z - 128;
  auto r = clamp<uint8_t>(tmp_y + 1.596f * tmp_r, 0, 255);
  auto g = clamp<uint8_t>(tmp_y - 0.813f * tmp_r - 0.392f * tmp_b, 0, 255);
  auto b = clamp<uint8_t>(tmp_y + 2.017f * tmp_b, 0, 255);
  return {r, g, b};
}

}  // namespace itu_r_bt_601

// Y, Cb, Cr formulas used in JPEG, using the whole dynamic range of the type.
namespace jpeg {

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_y(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs(0.299f, 0.587f, 0.114f);
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  return ConvertSatNorm<Output>(dot(coeffs, rgb));
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE uint8_t rgb_to_y(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.299f, 0.587f, 0.114f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb));
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cb(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs(-0.16873589f, -0.33126411f, 0.5f);
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  return ConvertSatNorm<Output>(dot(coeffs, rgb) + 0.5f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE uint8_t rgb_to_cb(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(-0.16873589f, -0.33126411f, 0.5f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cr(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs(0.5f, -0.41868759f, -0.08131241f);
  auto rgb = detail::norm(rgb_in);  // TODO(janton): optimize number of multiplications
  return ConvertSatNorm<Output>(dot(coeffs, rgb) + 0.5f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE uint8_t rgb_to_cr(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.5f, -0.41868759f, -0.08131241f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr_in) {
  auto ycbcr = detail::norm(ycbcr_in);  // TODO(janton): optimize number of multiplications
  float tmp_b = ycbcr.y - 0.5f;
  float tmp_r = ycbcr.z - 0.5f;
  auto r = ConvertSatNorm<Output>(ycbcr.x + 1.402f * tmp_r);
  auto g = ConvertSatNorm<Output>(ycbcr.x - 0.34413629f * tmp_b - 0.71413629f * tmp_r);
  auto b = ConvertSatNorm<Output>(ycbcr.x + 1.772f * tmp_b);
  return {r, g, b};
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, uint8_t> ycbcr_to_rgb(vec<3, uint8_t> ycbcr) {
  float tmp_b = ycbcr.y - 128;
  float tmp_r = ycbcr.z - 128;
  auto r = clamp<uint8_t>(ycbcr.x + 1.402f * tmp_r, 0, 255);
  auto g = clamp<uint8_t>(ycbcr.x - 0.34413629f * tmp_b - 0.71413629f * tmp_r, 0, 255);
  auto b = clamp<uint8_t>(ycbcr.x + 1.772f * tmp_b, 0, 255);
  return {r, g, b};
}


}  // namespace jpeg

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_gray(vec<3, Input> rgb) {
  return jpeg::rgb_to_y<Output>(rgb);
}

}  // namespace color
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
