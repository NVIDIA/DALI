// Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <type_traits>
#include "dali/core/geom/vec.h"
#include "dali/core/geom/mat.h"
#include "dali/core/convert.h"
#include "dali/core/math_util.h"

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

/**
 * @brief Clamps an input channel to the range used by the color conversions
 *
 * The conversions use only the non-negative part of the dynamic range: an integer value is
 * normalized by the maximum of its type (e.g. uint8_t(255) and int16_t(32767) both mean 1.0), the
 * Y and chroma biases are placed in the positive half (see `bias_scale`) and the chroma is
 * centered at 0.5. The biases make the conversions depend on the absolute value of the input, not
 * only on its scale, so an RGB or gray value outside of [0, 1] has no meaning in these color
 * spaces. Therefore, before the conversion matrix, negative values of a signed integer channel are
 * clamped to 0 and floating point values are clamped to [0, 1]. Unsigned integers are returned
 * unchanged.
 *
 * The value is only clamped: it keeps the `Input` type and scale (e.g. int16_t(-500) becomes
 * int16_t(0) and int16_t(1000) stays int16_t(1000)), and the scaling to `Output` is left to the
 * conversion formulas below.
 *
 * This is used for RGB and gray inputs, whatever the `Output` type is. For YCbCr inputs, see
 * `clamp_ycbcr_input`.
 */
template <typename Input>
constexpr DALI_HOST_DEV Input clamp_color_input(Input value) {
  static_assert(is_arithmetic_or_half<Input>::value,
                "clamp_color_input can only be used with arithmetic types");
  if constexpr (is_fp_or_half<Input>::value) {
    return clamp(value, Input(0), Input(1));
  } else if constexpr (std::is_signed<Input>::value) {  // NOLINT - linter bug, else if constexpr
    return value < 0 ? Input(0) : value;
  } else {
    return value;
  }
}

/**
 * @brief Applies the scalar `clamp_color_input` to each channel
 */
template <typename Input>
constexpr DALI_HOST_DEV vec<3, Input> clamp_color_input(vec<3, Input> v) {
  return {
    clamp_color_input(v[0]),
    clamp_color_input(v[1]),
    clamp_color_input(v[2])
  };
}

/**
 * @brief Clamps a YCbCr input (a channel or the whole pixel) as `clamp_color_input` does, but only
 *        when `Output` is an integer type; for a floating point `Output` it's returned unchanged.
 *
 * The signed YCbCr convention is not settled: here, signed integer samples are normalized by the
 * maximum of their type and the Y and chroma offsets are in the positive half (see `bias_scale`),
 * whereas a signed YCbCr pixel could also be read as Y in the positive range and CbCr centered at
 * 0, extending to both signs. Revisiting that is a part of the extended dynamic range topic and
 * would change the results for some inputs, so floating point outputs are left as they were.
 */
template <typename Output, typename Input>
constexpr DALI_HOST_DEV Input clamp_ycbcr_input(Input value) {
  if constexpr (is_fp_or_half<Output>::value) {
    return value;
  } else {
    return clamp_color_input(value);
  }
}

}  // namespace detail

/*
 * All functions below accept any combination of `Output` and `Input` types (unsigned or signed
 * integers, floating point). Each RGB or gray input channel is first clamped with
 * `detail::clamp_color_input` (negative values to 0, floating point values to [0, 1]), whatever
 * the `Output` type is. YCbCr inputs are clamped the same way only if `Output` is not a floating
 * point type (see `detail::clamp_ycbcr_input`). The results are scaled to the dynamic range of
 * `Output` and rounded/saturated as needed.
 *
 * Without the clamping, e.g. a negative value of a signed channel would contribute a negative
 * term to the conversion matrix and yield colors outside of the color space, which then saturate
 * to wrong values. Each channel is clamped separately, before the conversion matrix, as the GPU
 * decode path does; clamping only the result of the matrix would give different colors.
 *
 * For RGB and GRAY inputs, this makes the conversion functions equivalent to converting each
 * channel to `Output` first (as the GPU decode path, ConvertGPU, does) and then running the
 * conversion with `Output` as both input and output type, but without the intermediate rounding.
 * For YCbCr inputs (`ycbcr_to_rgb`, `y_to_gray`, `ycbcr_to_gray`) the Y and chroma biases are
 * in `Input` units (2^n/16 and 2^n/2), which normalize differently for each type (e.g. 64/127
 * for int8_t, 128/255 for uint8_t, 0.5 for float), so the two orders also differ by that bias
 * difference - e.g. up to ~0.85% of full scale for int8_t YCbCr to float RGB. This is negligible
 * for 16-bit and wider types.
 */

// Y, Cb, Cr definition from ITU-R BT.601, with values in the range 16-235, allowing for
// footroom and headroom
struct itu_r_bt_601 {
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_y(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.25678823529f, 0.50412941176f, 0.09790588235f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.0625f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::clamp_color_input(rgb)) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(-0.14822289945f, -0.29099278682f, 0.43921568627f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::clamp_color_input(rgb)) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.43921568627f, -0.36778831435f, -0.07142737192)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::clamp_color_input(rgb)) + bias;
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
    return ConvertSat<Output>(scale * (detail::clamp_ycbcr_input<Output>(y) - bias));
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output gray_to_y(Input y) {
    constexpr float bias = 0.0625f * detail::bias_scale<Output>();
    constexpr float scale = 219 * detail::scale_factor<Input, Output>() / 255;
    return ConvertSat<Output>(detail::clamp_color_input(y) * scale + bias);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr) {
    constexpr float cbias = 0.5f * detail::bias_scale<Input>();
    constexpr float ybias = 0.0625f * detail::bias_scale<Input>();
    constexpr float s = detail::scale_factor<Input, Output>();
    ycbcr = detail::clamp_ycbcr_input<Output>(ycbcr);
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
    float y = dot(coeffs, detail::clamp_color_input(rgb));
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(-0.16873589f, -0.33126411f, 0.5f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::clamp_color_input(rgb)) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.5f, -0.41868759f, -0.08131241f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::clamp_color_input(rgb)) + bias;
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
    ycbcr = detail::clamp_ycbcr_input<Output>(ycbcr);
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
    // Clamp first, as in the other functions - ConvertSatNorm alone would keep negative values
    // for a signed Output
    return ConvertSatNorm<Output>(detail::clamp_color_input(gray));
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output y_to_gray(Input y) {
    // Clamp first, as in the other functions - ConvertSatNorm alone would keep negative values
    // for a signed Output
    return ConvertSatNorm<Output>(detail::clamp_ycbcr_input<Output>(y));
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
