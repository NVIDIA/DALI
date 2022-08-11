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

template<typename T>
constexpr DALI_HOST_DEV std::enable_if_t<!is_fp_or_half<T>::value, double> get_max_value() {
  return static_cast<double>(max_value<T>());
}

template<typename T>
constexpr DALI_HOST_DEV std::enable_if_t<is_fp_or_half<T>::value, double> get_max_value() {
  return 1.0;
}


template <typename From, typename To>
constexpr DALI_HOST_DEV float scale_factor() {
  constexpr double to = get_max_value<To>(), from = get_max_value<From>();
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
  static DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_y(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.25678823529f, 0.50412941176f, 0.09790588235f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.0625f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(-0.14822289945f, -0.29099278682f, 0.43921568627f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.43921568627f, -0.36778831435f, -0.07142737192)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> rgb_to_ycbcr(vec<3, Input> rgb) {
    return {
      rgb_to_y<Output, Input>(rgb),
      rgb_to_cb<Output, Input>(rgb),
      rgb_to_cr<Output, Input>(rgb)
    };
  }

  // Gray uses the full dynamic range of the type (e.g. 0..255)
  // while ITU-R BT.601 uses a reduced range to allow for headroom and footroom (e.g. 16..235)
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output y_to_gray(Input y) {
    constexpr float scale = 255 * detail::scale_factor<Input, Output>() / 219;
    constexpr float bias = 0.0625f * detail::bias_scale<Input>();
    return ConvertSat<Output>(scale * (y - bias));
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
  constexpr vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr) {
    constexpr float cbias = 0.5f * detail::bias_scale<Input>();
    constexpr float ybias = 0.0625f * detail::bias_scale<Input>();
    constexpr float s = detail::scale_factor<Input, Output>();
    float ys = (ycbcr[0] - ybias) * (255.0f / 219) * s;
    float tmp_b = ycbcr[1] - cbias;
    float tmp_r = ycbcr[2] - cbias;
    auto r = ConvertSat<Output>(ys + (1.5960267848f * s) * tmp_r);
    auto g = ConvertSat<Output>(ys - (0.39176228842f * s) * tmp_b - (0.81296764538f * s) * tmp_r);
    auto b = ConvertSat<Output>(ys + (2.0172321417f * s) * tmp_b);
    return {r, g, b};
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output ycbcr_to_gray(vec<3, Input> ycbcr) {
    return y_to_gray<Output>(ycbcr[0]);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> gray_to_ycbcr(Input gray) {
    auto c = ConvertNorm<Output>(0.5f);
    return {gray_to_y<Output>(gray), c, c};
  }
};  // struct itu_r_bt_601


// Y, Cb, Cr formulas used in JPEG, using the whole dynamic range of the type.
struct jpeg {
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_y(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.299f, 0.587f, 0.114f) * detail::scale_factor<Input, Output>();
    float y = dot(coeffs, rgb);
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(-0.16873589f, -0.33126411f, 0.5f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.5f, -0.41868759f, -0.08131241f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, rgb) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> rgb_to_ycbcr(vec<3, Input> rgb) {
    return {
      rgb_to_y<Output, Input>(rgb),
      rgb_to_cb<Output, Input>(rgb),
      rgb_to_cr<Output, Input>(rgb)
    };
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr) {
    constexpr float cbias = 0.5f * detail::bias_scale<Input>();
    constexpr float s = detail::scale_factor<Input, Output>();
    float tmp_b = ycbcr[1] - cbias;
    float tmp_r = ycbcr[2] - cbias;
    float ys = ycbcr[0] * s;
    auto r = ConvertSat<Output>(ys + (1.402f * s) * tmp_r);
    auto g = ConvertSat<Output>(ys - (0.344136285f * s) * tmp_b - (0.714136285f * s) * tmp_r);
    auto b = ConvertSat<Output>(ys + (1.772f * s) * tmp_b);
    return {r, g, b};
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output gray_to_y(Input gray) {
    return ConvertSatNorm<Output>(gray);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output y_to_gray(Input y) {
    return ConvertSatNorm<Output>(y);
  }
};  // struct jpeg


template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_gray(vec<3, Input> rgb) {
  return jpeg::rgb_to_y<Output>(rgb);
}

}  // namespace color
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
