// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


template <typename Input>
DALI_HOST_DEV DALI_FORCEINLINE float norm(Input x) {
  return ConvertNorm<float>(x);
}

template <typename From, typename To>
constexpr DALI_HOST_DEV float scale_factor() {
  constexpr double to = is_fp_or_half<To>::value
                      ? 1.0 : static_cast<double>(max_value<To>());

  constexpr double from = is_fp_or_half<From>::value
                        ? 1.0 : static_cast<double>(max_value<From>());

  constexpr float factor = to / from;
  return factor;
}

template <typename T>
constexpr DALI_HOST_DEV std::enable_if_t<std::is_integral<T>::value, float> bias_scale() {
  // The scale is 2^positive_bits
  // Since that value always overflows, we shift by bits-1 and then multiply x2 already in
  // floating point.
  return 2.0f * static_cast<float>(T(1) << (sizeof(T) * 8 - 1 - std::is_signed<T>::value));
}

template <typename T>
constexpr DALI_HOST_DEV std::enable_if_t<!std::is_integral<T>::value, float> bias_scale() {
  return 1.0f;
}

}  // namespace detail

// Y, Cb, Cr definition from ITU-R BT.601, with values in the range 16-235, allowing for
// footroom and headroom
struct itu_r_bt_601 {
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_y(vec<3, Input> rgb_in) {
    constexpr vec3 coeffs = vec3(0.25678823529f, 0.50412941176f, 0.09790588235f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.0625f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb_in) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb_in) {
    constexpr vec3 coeffs = vec3(-0.14822289945f, -0.29099278682f, 0.43921568627f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb_in) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb_in) {
    constexpr vec3 coeffs = vec3(0.43921568627f, -0.36778831435f, -0.07142737192)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb_in) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  // Gray uses the full dynamic range of the type (e.g. 0..255)
  // while ITU-R BT.601 uses a reduced range to allow for floorroom and footroom (e.g. 16..235)
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output y_to_gray(Input y) {
    constexpr float scale = 255 * detail::scale_factor<Input, Output>() / 219;
    constexpr float bias = 0.5f * detail::bias_scale<Input>();
    return ConvertSatNorm<Output>(scale * (y - 0.0625f));
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output gray_to_y(Input y) {
    constexpr float bias = 0.0625f * detail::bias_scale<Output>();
    constexpr float scale = 219 * detail::scale_factor<Input, Output>() / 255;
    return ConvertSat<Output>(y * scale + bias);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr_in) {
    auto ycbcr = detail::norm(ycbcr_in);  // TODO(janton): optimize number of multiplications
    auto tmp_y = 1.164f * (ycbcr[0] - 0.0625f);
    auto tmp_b = ycbcr[1] - 0.5f;
    auto tmp_r = ycbcr[2] - 0.5f;
    auto r = ConvertSatNorm<Output>(tmp_y + 1.596f * tmp_r);
    auto g = ConvertSatNorm<Output>(tmp_y - 0.813f * tmp_r - 0.392f * tmp_b);
    auto b = ConvertSatNorm<Output>(tmp_y + 2.017f * tmp_b);
    return {r, g, b};
  }

};  // struct itu_r_bt_601

template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t itu_r_bt_601::rgb_to_y<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.257f, 0.504f, 0.098f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 16.0f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t itu_r_bt_601::rgb_to_cb<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(-0.148f, -0.291f, 0.439f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t itu_r_bt_601::rgb_to_cr<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.439f, -0.368f, -0.071f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t itu_r_bt_601::y_to_gray<uint8_t, uint8_t>(uint8_t gray) {
  constexpr float scale = 1 / (0.257f + 0.504f + 0.098f);
  return ConvertSat<uint8_t>(scale * (gray - 16));
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t itu_r_bt_601::gray_to_y<uint8_t, uint8_t>(uint8_t y) {
  constexpr float scale = 0.257f + 0.504f + 0.098f;
  return ConvertSat<uint8_t>(y * scale + 0.0625f);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
vec<3, uint8_t> itu_r_bt_601::ycbcr_to_rgb<uint8_t, uint8_t>(vec<3, uint8_t> ycbcr) {
  auto tmp_y = 1.164f * (ycbcr[0] - 16);
  auto tmp_b = ycbcr[1] - 128;
  auto tmp_r = ycbcr[2] - 128;
  auto r = ConvertSat<uint8_t>(tmp_y + 1.596f * tmp_r);
  auto g = ConvertSat<uint8_t>(tmp_y - 0.813f * tmp_r - 0.392f * tmp_b);
  auto b = ConvertSat<uint8_t>(tmp_y + 2.017f * tmp_b);
  return {r, g, b};
}

// Y, Cb, Cr formulas used in JPEG, using the whole dynamic range of the type.
struct jpeg {
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_y(vec<3, Input> rgb_in) {
    constexpr vec3 coeffs = vec3(0.299f, 0.587f, 0.114f) * detail::scale_factor<Input, Output>();
    float y = dot(coeffs, rgb_in);
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb_in) {
    constexpr vec3 coeffs = vec3(-0.16873589f, -0.33126411f, 0.5f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb_in) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb_in) {
    constexpr vec3 coeffs = vec3(0.5f, -0.41868759f, -0.08131241f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb_in) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr_in) {
    auto ycbcr = detail::norm(ycbcr_in);  // TODO(janton): optimize number of multiplications
    float tmp_b = ycbcr[1] - 0.5f;
    float tmp_r = ycbcr[2] - 0.5f;
    auto r = ConvertSatNorm<Output>(ycbcr[0] + 1.402f * tmp_r);
    auto g = ConvertSatNorm<Output>(ycbcr[0] - 0.34413629f * tmp_b - 0.71413629f * tmp_r);
    auto b = ConvertSatNorm<Output>(ycbcr[0] + 1.772f * tmp_b);
    return {r, g, b};
  }
};  // struct jpeg


template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t jpeg::rgb_to_y<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  vec3 coeffs(0.299f, 0.587f, 0.114f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb));
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t jpeg::rgb_to_cb<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  vec3 coeffs(-0.16873589f, -0.33126411f, 0.5f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
uint8_t jpeg::rgb_to_cr<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  vec3 coeffs(0.5f, -0.41868759f, -0.08131241f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
vec<3, uint8_t> jpeg::ycbcr_to_rgb(vec<3, uint8_t> ycbcr) {
  float tmp_b = ycbcr[1] - 128;
  float tmp_r = ycbcr[2] - 128;
  auto r = ConvertSat<uint8_t>(ycbcr[0] + 1.402f * tmp_r);
  auto g = ConvertSat<uint8_t>(ycbcr[0] - 0.34413629f * tmp_b - 0.71413629f * tmp_r);
  auto b = ConvertSat<uint8_t>(ycbcr[0] + 1.772f * tmp_b);
  return {r, g, b};
}


template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_gray(vec<3, Input> rgb) {
  return jpeg::rgb_to_y<Output>(rgb);
}

}  // namespace color
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
