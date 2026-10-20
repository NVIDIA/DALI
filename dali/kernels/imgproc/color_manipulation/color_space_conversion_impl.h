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
 * @brief Saturates an input value to the range of normalized values representable in `Output`
 *
 * Integer values are interpreted as normalized: unsigned integers as UNORM (`v / max_value`,
 * in [0, 1]) and signed integers as SNORM (`v / max_value`, in [-1, 1]); floating point values
 * are used as they are. The value is then saturated to the normalized range of `Output`:
 *   * unsigned `Output`: [0, 1]
 *   * signed integer `Output`: [-1, 1] - `min_value<Input>()`, which is one code below
 *     `-max_value<Input>()`, is clamped to -1, as in SNORM formats of graphics APIs
 *   * floating point `Output`: no limit, the value is returned unchanged
 * Unsigned integer inputs are never modified.
 *
 * Note that the value is not converted to `Output` - it's returned in the `Input` type (and
 * units), so that the conversion formulas below can apply their usual Input -> Output scaling
 * to it.
 *
 * Unlike here, `ConvertSatNorm` to a signed integer type keeps `min_value<Input>()` one code
 * below -1 when `Output` is as wide as `Input` or wider (e.g. `ConvertSatNorm<int8_t>` of
 * int8_t(-128) is -128 and `ConvertSatNorm<int16_t>` of int8_t(-128) is -32768). Like here,
 * `ConvertSatNorm` to a floating point type doesn't clamp it - int8_t(-128) normalizes to
 * -128/127, not -1.
 */
template <typename Output, typename Input>
constexpr DALI_HOST_DEV Input saturate_to_output_range(Input value) {
  static_assert(is_arithmetic_or_half<Output>::value && is_arithmetic_or_half<Input>::value,
                "saturate_to_output_range can only be used with arithmetic types");
  if constexpr (is_fp_or_half<Output>::value) {
    return value;
  } else if constexpr (is_fp_or_half<Input>::value) {  // NOLINT - linter bug with else if constexpr
    // float16 is compared as float
    using Calc = std::conditional_t<std::is_same<Input, float16>::value, float, Input>;
    constexpr Calc lo = std::is_unsigned<Output>::value ? Calc(0) : Calc(-1);
    Calc v = static_cast<Calc>(value);
    return v < lo ? static_cast<Input>(lo) : v > Calc(1) ? static_cast<Input>(Calc(1)) : value;
  } else if constexpr (std::is_signed<Input>::value) {  // NOLINT - linter bug, else if constexpr
    constexpr Input lo = std::is_unsigned<Output>::value
                         ? Input(0)
                         : static_cast<Input>(-max_value<Input>());
    return value < lo ? lo : value;
  } else {
    return value;
  }
}

/**
 * @brief Applies the scalar `saturate_to_output_range` to each channel
 */
template <typename Output, typename Input>
constexpr DALI_HOST_DEV vec<3, Input> saturate_to_output_range(vec<3, Input> v) {
  return {
    saturate_to_output_range<Output>(v[0]),
    saturate_to_output_range<Output>(v[1]),
    saturate_to_output_range<Output>(v[2])
  };
}

}  // namespace detail

/*
 * All functions below accept any combination of `Output` and `Input` types (unsigned or signed
 * integers, floating point). The input channels are first saturated to the normalized range of
 * `Output` with `detail::saturate_to_output_range`; the results are scaled to the dynamic range
 * of `Output` and rounded/saturated as needed.
 *
 * Without the saturation, e.g. a negative value of a signed channel would contribute a negative
 * term to the conversion matrix and yield colors not representable in an unsigned output type,
 * which then saturate to wrong values.
 *
 * Each channel is clipped separately, before the conversion matrix - also for floating point
 * inputs out of the [0, 1] (or [-1, 1]) range. This is a deliberate choice that matches the GPU
 * decode path; clipping only the result of the matrix would give different colors.
 *
 * For RGB and GRAY inputs, this makes the conversion functions equivalent to converting each
 * channel to `Output` first (as the GPU decode path, ConvertGPU, does) and then running the
 * conversion with `Output` as both input and output type, but without the intermediate rounding
 * - up to one code at `min_value<Input>()` for a signed integral `Output` (see
 * `detail::saturate_to_output_range`). For YCbCr inputs (`ycbcr_to_rgb`, `y_to_gray`,
 * `ycbcr_to_gray`) the Y and chroma biases are in `Input` units (2^n/16 and 2^n/2), which
 * normalize differently for each type (e.g. 64/127 for int8_t, 128/255 for uint8_t, 0.5 for
 * float), so the two orders also differ by that bias difference - e.g. up to ~0.85% of full scale
 * for int8_t YCbCr to float RGB. This is negligible for 16-bit and wider types.
 */

// Y, Cb, Cr definition from ITU-R BT.601, with values in the range 16-235, allowing for
// footroom and headroom
struct itu_r_bt_601 {
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_y(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.25678823529f, 0.50412941176f, 0.09790588235f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.0625f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::saturate_to_output_range<Output>(rgb)) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(-0.14822289945f, -0.29099278682f, 0.43921568627f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::saturate_to_output_range<Output>(rgb)) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.43921568627f, -0.36778831435f, -0.07142737192)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::saturate_to_output_range<Output>(rgb)) + bias;
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
    return ConvertSat<Output>(scale * (detail::saturate_to_output_range<Output>(y) - bias));
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output gray_to_y(Input y) {
    constexpr float bias = 0.0625f * detail::bias_scale<Output>();
    constexpr float scale = 219 * detail::scale_factor<Input, Output>() / 255;
    return ConvertSat<Output>(detail::saturate_to_output_range<Output>(y) * scale + bias);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr) {
    constexpr float cbias = 0.5f * detail::bias_scale<Input>();
    constexpr float ybias = 0.0625f * detail::bias_scale<Input>();
    constexpr float s = detail::scale_factor<Input, Output>();
    ycbcr = detail::saturate_to_output_range<Output>(ycbcr);
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
    float y = dot(coeffs, detail::saturate_to_output_range<Output>(rgb));
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(-0.16873589f, -0.33126411f, 0.5f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::saturate_to_output_range<Output>(rgb)) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb) {
    constexpr vec3 coeffs = vec3(0.5f, -0.41868759f, -0.08131241f)
                            * detail::scale_factor<Input, Output>();
    constexpr float bias = 0.5f * detail::bias_scale<Output>();
    float y = dot(coeffs, detail::saturate_to_output_range<Output>(rgb)) + bias;
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
    ycbcr = detail::saturate_to_output_range<Output>(ycbcr);
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
    // Saturate first, so that the same saturation as in the other functions is applied
    // (ConvertSatNorm alone doesn't clamp min_value or floating point values before scaling)
    return ConvertSatNorm<Output>(detail::saturate_to_output_range<Output>(gray));
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output y_to_gray(Input y) {
    // Saturate first, so that the same saturation as in the other functions is applied
    // (ConvertSatNorm alone doesn't clamp min_value or floating point values before scaling)
    return ConvertSatNorm<Output>(detail::saturate_to_output_range<Output>(y));
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
