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
 * @brief The factor that makes the limited range spans of an integer type a fraction of 2^bits
 *
 * `limited_range::y_scale` and `c_scale` are fractions of the maximum value of the type (219/255
 * and 224/255), which is exact for 8 bits only. The ITU-R recommendations and H.273 scale the
 * limited range with 2^(bits-8): the spans are 219 * 2^(bits-8) and 224 * 2^(bits-8), that is
 * 219/256 and 224/256 of 2^bits. The factor converts one into the other (it's 1 for uint8_t).
 * Floating point types have no such definition and the factor is 1.
 *
 * The 32-bit types are limited by the precision of float: the factor is the same for all of them,
 * and the white of the limited range (235 * 2^24 for uint32_t) is converted back with an error of
 * a few float ULPs, so it can be one ULP below the maximum.
 */
template <typename T>
constexpr DALI_HOST_DEV std::enable_if_t<std::is_integral<T>::value, float> limited_span_scale() {
  return static_cast<float>(255.0 / 256 * bias_scale<T>() / get_max_value<T>());
}

template <typename T>
constexpr DALI_HOST_DEV std::enable_if_t<!std::is_integral<T>::value, float> limited_span_scale() {
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
 * For YCbCr inputs (`ycbcr_to_rgb`, `y_to_gray`, `ycbcr_to_gray`) the Y and chroma biases and
 * the spans of the limited range are in `Input` units (see below), which normalize differently
 * for each type (e.g. the chroma bias is 64/127 for int8_t, 128/255 for uint8_t, 0.5 for float;
 * the Y span is 219/256 of 2^n for integer types and 219/255 for floating point). The two orders
 * therefore differ by the difference of these interpretations, whenever `Input` and `Output` are
 * different types: up to ~2% of full scale (e.g. for int32_t or int8_t YCbCr to float RGB, or
 * int8_t to int16_t), and less for most of the other pairs. They are the same for equal types.
 */

namespace detail {

/*
 * RGB <-> YCbCr, for the luma weights Kr and Kb of red and blue (green is Kg = 1 - Kr - Kb).
 * With R, G, B in [0, 1]:
 *
 *   Y  = Kr R + Kg G + Kb B
 *   Cb = (B - Y) / (2 (1 - Kb))             Cb, Cr in [-0.5, 0.5]
 *   Cr = (R - Y) / (2 (1 - Kr))
 *
 * and the inverse
 *
 *   R = Y + 2 (1 - Kr) Cr
 *   B = Y + 2 (1 - Kb) Cb
 *   G = (Y - Kr R - Kb B) / Kg = Y - 2 Kb (1 - Kb) / Kg Cb - 2 Kr (1 - Kr) / Kg Cr
 *
 * The range then places them in the dynamic range of the type, with `y_scale` and `c_scale`
 * (the limited range is 16-235 for Y and 16-240 for Cb and Cr):
 *
 *   Y' = y_bias + y_scale Y        Cb' = 0.5 + c_scale Cb        Cr' = 0.5 + c_scale Cr
 *
 * For an integer type of n bits (the bits of the values, so 7 for int8_t), the limited range is
 * defined as in the ITU-R recommendations and H.273, with offsets and spans scaled by 2^(n-8):
 * Y' is in 16 * 2^(n-8) .. 235 * 2^(n-8) and Cb', Cr' in 16 * 2^(n-8) .. 240 * 2^(n-8), centered
 * at 2^(n-1). E.g. for 10 bits the ranges are 64..940 and 64..960, and for uint16_t 4096..60160
 * and 4096..61440. These are fractions of 2^n (16/256 = 1/16 for Y, 1/2 for the chroma, 219/256
 * and 224/256 for the spans), unlike the maximum of the type, 2^n - 1, which is used by the full
 * range. For 8 bits, the two are the same ranges, 16..235 and 16..240.
 *
 * Floating point has no such definition: the offsets are 1/16 and 1/2 and the spans are
 * `y_scale` and `c_scale` (219/255 and 224/255), so Y' is in [1/16, 1/16 + y_scale]. This is
 * within 0.6% of the values of 8-bit data normalized by 255 and of the integer definition above
 * for any bit depth, but equal to neither, because the bit depth of the source isn't known.
 *
 * The coefficients below are the matrices of these formulas with the scales folded in. The
 * limited range spans of integer types come from `limited_span_scale`.
 */

// The luma weights of the ITU-R recommendations (and of JPEG, which uses the BT.601 ones)
struct bt_601 {
  static constexpr double kr = 0.299, kb = 0.114;
};

struct bt_709 {
  static constexpr double kr = 0.2126, kb = 0.0722;
};

// Where Y (and the gray values) lie in the dynamic range. The gray functions do not depend on the
// RGB <-> YCbCr matrix, so they come with the range.

// The values in the range 16-235, allowing for footroom and headroom, as in the ITU-R
// recommendations.
struct limited_range {
  static constexpr float y_bias = 0.0625f;  // 16/256 of the dynamic range
  // Fractions of the maximum value of the type: Y is in 16..235, out of 0..255. For integer types
  // other than uint8_t, see `span_scale`
  static constexpr double y_scale = 219.0 / 255;
  static constexpr double c_scale = 224.0 / 255;  // Cb and Cr are in 16..240

  /**
   * @brief The factor of `y_scale` and `c_scale` for the type `T`
   *
   * It's 1 for floating point and uint8_t. For the other integer types, it makes the spans
   * 219 * 2^(bits-8) and 224 * 2^(bits-8), see `limited_span_scale`.
   */
  template <typename T>
  static constexpr DALI_HOST_DEV float span_scale() {
    return limited_span_scale<T>();
  }

  // Gray uses the full dynamic range of the type (e.g. 0..255) while the ITU-R recommendations use
  // a reduced range to allow for headroom and footroom (e.g. 16..235)
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output y_to_gray(Input y) {
    // 1 / (y_scale * span_scale)
    constexpr float scale = 255 * scale_factor<Input, Output>() / 219 / span_scale<Input>();
    constexpr float bias = y_bias * bias_scale<Input>();
    return ConvertSat<Output>(scale * (clamp_ycbcr_input<Output>(y) - bias));
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output gray_to_y(Input y) {
    constexpr float bias = y_bias * bias_scale<Output>();
    // y_scale * span_scale
    constexpr float scale = 219 * scale_factor<Input, Output>() / 255 * span_scale<Output>();
    return ConvertSat<Output>(clamp_color_input(y) * scale + bias);
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
};  // struct limited_range

// The whole dynamic range of the type.
struct full_range {
  static constexpr float y_bias = 0.0f;
  static constexpr double y_scale = 1.0;
  static constexpr double c_scale = 1.0;

  // The whole dynamic range is a fraction of the maximum value of the type, for any type
  template <typename T>
  static constexpr DALI_HOST_DEV float span_scale() {
    return 1.0f;
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output gray_to_y(Input gray) {
    // Clamp first, as in the other functions - ConvertSatNorm alone would keep negative values
    // for a signed Output
    return ConvertSatNorm<Output>(clamp_color_input(gray));
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output y_to_gray(Input y) {
    // Clamp first, as in the other functions - ConvertSatNorm alone would keep negative values
    // for a signed Output
    return ConvertSatNorm<Output>(clamp_ycbcr_input<Output>(y));
  }
};  // struct full_range

/**
 * @brief The coefficients of the formulas above for luma weights and a range
 */
template <typename Weights, typename Range>
struct ycbcr_coefficients {
  static constexpr double kr = Weights::kr, kb = Weights::kb, kg = 1 - kr - kb;
  static constexpr double ys = Range::y_scale, cs = Range::c_scale;

  // RGB -> Y', Cb', Cr' (before the offsets): the rows of the matrix
  static constexpr vec3 y_row = vec3(kr * ys, kg * ys, kb * ys);
  static constexpr vec3 cb_row =
      vec3(-kr / (2 * (1 - kb)) * cs, -kg / (2 * (1 - kb)) * cs, 0.5 * cs);
  static constexpr vec3 cr_row =
      vec3(0.5 * cs, -kg / (2 * (1 - kr)) * cs, -kb / (2 * (1 - kr)) * cs);

  // Y', Cb', Cr' (without the offsets) -> RGB
  static constexpr float y_gain = 1 / ys;
  static constexpr float r_cr = 2 * (1 - kr) / cs;
  static constexpr float g_cb = 2 * kb * (1 - kb) / kg / cs;
  static constexpr float g_cr = 2 * kr * (1 - kr) / kg / cs;
  static constexpr float b_cb = 2 * (1 - kb) / cs;
};

/**
 * @brief The conversions between RGB and YCbCr, and between gray and Y, for luma weights and a
 *        range
 */
template <typename Weights, typename Range>
struct ycbcr_method : Range {
 private:
  using C = ycbcr_coefficients<Weights, Range>;

  // One of Y', Cb', Cr' of an RGB pixel, from its row of the matrix (scaled for the types) and its
  // offset, which the callers compute at compile time
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_channel(const vec3 &coeffs, float bias, vec<3, Input> rgb) {
    float v = dot(coeffs, clamp_color_input(rgb)) + bias;
    return needs_clamp<Input, Output>::value ? Convert<Output>(v) : ConvertSat<Output>(v);
  }

 public:
  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_y(vec<3, Input> rgb) {
    constexpr vec3 coeffs =
        C::y_row * (scale_factor<Input, Output>() * Range::template span_scale<Output>());
    constexpr float bias = Range::y_bias * bias_scale<Output>();
    return rgb_to_channel<Output>(coeffs, bias, rgb);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cb(vec<3, Input> rgb) {
    constexpr vec3 coeffs =
        C::cb_row * (scale_factor<Input, Output>() * Range::template span_scale<Output>());
    constexpr float bias = 0.5f * bias_scale<Output>();
    return rgb_to_channel<Output>(coeffs, bias, rgb);
  }

  template <typename Output, typename Input>
  static DALI_HOST_DEV DALI_FORCEINLINE
  constexpr Output rgb_to_cr(vec<3, Input> rgb) {
    constexpr vec3 coeffs =
        C::cr_row * (scale_factor<Input, Output>() * Range::template span_scale<Output>());
    constexpr float bias = 0.5f * bias_scale<Output>();
    return rgb_to_channel<Output>(coeffs, bias, rgb);
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
    constexpr float cbias = 0.5f * bias_scale<Input>();
    constexpr float ybias = Range::y_bias * bias_scale<Input>();
    constexpr float s = scale_factor<Input, Output>();
    constexpr float k = 1.0f / Range::template span_scale<Input>();
    ycbcr = clamp_ycbcr_input<Output>(ycbcr);
    float y = (ycbcr[0] - ybias) * C::y_gain * s * k;
    float tmp_b = ycbcr[1] - cbias;
    float tmp_r = ycbcr[2] - cbias;
    auto r = ConvertSat<Output>(y + (C::r_cr * s * k) * tmp_r);
    auto g = ConvertSat<Output>(y - (C::g_cb * s * k) * tmp_b - (C::g_cr * s * k) * tmp_r);
    auto b = ConvertSat<Output>(y + (C::b_cb * s * k) * tmp_b);
    return {r, g, b};
  }
};  // struct ycbcr_method

}  // namespace detail

// Y, Cb, Cr definition from ITU-R BT.601, with values in the range 16-235, allowing for
// footroom and headroom
struct itu_r_bt_601 : detail::ycbcr_method<detail::bt_601, detail::limited_range> {};

// Y, Cb, Cr definition from ITU-R BT.709 (the usual one for HD video), with values in the range
// 16-235, allowing for footroom and headroom.
// It differs from ITU-R BT.601 only in the RGB <-> YCbCr matrix.
struct itu_r_bt_709 : detail::ycbcr_method<detail::bt_709, detail::limited_range> {};

// Y, Cb, Cr formulas used in JPEG, using the whole dynamic range of the type.
struct jpeg : detail::ycbcr_method<detail::bt_601, detail::full_range> {};

// Y, Cb, Cr with the ITU-R BT.709 matrix, using the whole dynamic range of the type (the
// counterpart of `jpeg`, which uses the ITU-R BT.601 matrix). It is what the video decoders need
// for full range video.
struct itu_r_bt_709_full_range
    : detail::ycbcr_method<detail::bt_709, detail::full_range> {};


template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_gray(vec<3, Input> rgb) {
  return jpeg::rgb_to_y<Output>(rgb);
}

namespace detail {

DALI_HOST_DEV DALI_FORCEINLINE float srgb_to_linear(float c) {
  c = clamp(c, 0.0f, 1.0f);
  return c <= 0.04045f ? c * (1.0f / 12.92f) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

DALI_HOST_DEV DALI_FORCEINLINE float linear_to_srgb(float c) {
  c = clamp(c, 0.0f, 1.0f);
  float s = c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
  return clamp(s, 0.0f, 1.0f);
}

DALI_HOST_DEV DALI_FORCEINLINE float lab_f(float t) {
  constexpr float delta = 6.0f / 29.0f;
  constexpr float delta_cubed = delta * delta * delta;            // 216.0f / 24389.0f
  constexpr float kappa_div_116 = 1.0f / (3.0f * delta * delta);  // 841.0f / 108.0f
  constexpr float offset = 4.0f / 29.0f;
  return t > delta_cubed ? cbrtf(t) : kappa_div_116 * t + offset;
}

DALI_HOST_DEV DALI_FORCEINLINE float lab_f_inv(float t) {
  constexpr float delta = 6.0f / 29.0f;
  constexpr float three_delta_sq = 3.0f * delta * delta;  // 108.0f / 841.0f
  constexpr float offset = 4.0f / 29.0f;
  return t > delta ? t * t * t : three_delta_sq * (t - offset);
}

// sRGB to CIE XYZ with D65 illuminant.
// Row sums equal the D65 white point (Xn = 0.950456, Yn = 1.0, Zn = 1.088754).
struct d65 {
  static constexpr dvec3 x_row = {0.412453, 0.357580, 0.180423};
  static constexpr dvec3 y_row = {0.212671, 0.715160, 0.072169};
  static constexpr dvec3 z_row = {0.019334, 0.119193, 0.950227};
};

// Bradford-adapted sRGB to CIE XYZ with D50 illuminant.
// Row sums equal the D50 white point (Xn = 0.96422, Yn = 1.0, Zn = 0.82521).
struct d50 {
  static constexpr dvec3 x_row = {0.4360747, 0.3850649, 0.1430804};
  static constexpr dvec3 y_row = {0.2225045, 0.7168786, 0.0606169};
  static constexpr dvec3 z_row = {0.0139322, 0.0971045, 0.7141733};
};

/**
 * @brief White-point-normalized sRGB <-> XYZ matrices computed at compile time
 *
 * Each row of the sRGB -> XYZ matrix is divided by its sum (the white point coordinate Xn, Yn,
 * Zn), so that linear white (1, 1, 1) maps to normalized XYZ (1, 1, 1) and L*a*b* (100, 0, 0).
 * The inverse matrix maps normalized XYZ back to linear sRGB.
 */
template <typename Illuminant>
struct cielab_coefficients {
  static constexpr double xn =
      Illuminant::x_row[0] + Illuminant::x_row[1] + Illuminant::x_row[2];
  static constexpr double yn =
      Illuminant::y_row[0] + Illuminant::y_row[1] + Illuminant::y_row[2];
  static constexpr double zn =
      Illuminant::z_row[0] + Illuminant::z_row[1] + Illuminant::z_row[2];

  static constexpr dvec3 m0 = Illuminant::x_row / xn;
  static constexpr dvec3 m1 = Illuminant::y_row / yn;
  static constexpr dvec3 m2 = Illuminant::z_row / zn;

  static constexpr vec3 to_xyz_x = vec3(m0[0], m0[1], m0[2]);
  static constexpr vec3 to_xyz_y = vec3(m1[0], m1[1], m1[2]);
  static constexpr vec3 to_xyz_z = vec3(m2[0], m2[1], m2[2]);

  // Cofactors and determinant of the 3x3 normalized matrix [m0; m1; m2]
  static constexpr double c00 = m1[1] * m2[2] - m1[2] * m2[1];
  static constexpr double c01 = m0[2] * m2[1] - m0[1] * m2[2];
  static constexpr double c02 = m0[1] * m1[2] - m0[2] * m1[1];

  static constexpr double c10 = m1[2] * m2[0] - m1[0] * m2[2];
  static constexpr double c11 = m0[0] * m2[2] - m0[2] * m2[0];
  static constexpr double c12 = m0[2] * m1[0] - m0[0] * m1[2];

  static constexpr double c20 = m1[0] * m2[1] - m1[1] * m2[0];
  static constexpr double c21 = m0[1] * m2[0] - m0[0] * m2[1];
  static constexpr double c22 = m0[0] * m1[1] - m0[1] * m1[0];

  static constexpr double det = m0[0] * c00 + m1[0] * c01 + m2[0] * c02;

  static constexpr vec3 from_xyz_r = vec3(c00 / det, c01 / det, c02 / det);
  static constexpr vec3 from_xyz_g = vec3(c10 / det, c11 / det, c12 / det);
  static constexpr vec3 from_xyz_b = vec3(c20 / det, c21 / det, c22 / det);
};

/**
 * @brief CIELAB conversions parameterized by reference illuminant (D65 or D50)
 *
 * For standard 8-bit images (uint8_t), canonical CIELAB coordinates
 * (L* in [0, 100], a* and b* centered at 0) are mapped to 8-bit range:
 *   L = round(L* * 255 / 100)
 *   a = round(a* + 128)
 *   b = round(b* + 128)
 */
template <typename Illuminant>
struct cielab_method {
 private:
  using C = cielab_coefficients<Illuminant>;

  static DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t pack_l(float L) {
    return ConvertSat<uint8_t>(L * (255.0f / 100.0f));
  }

  static DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t pack_ab(float ab) {
    return ConvertSat<uint8_t>(ab + 128.0f);
  }

  static DALI_HOST_DEV DALI_FORCEINLINE constexpr vec<3, uint8_t> pack_lab(vec3 lab) {
    return {
      pack_l(lab[0]),
      pack_ab(lab[1]),
      pack_ab(lab[2])
    };
  }

  static DALI_HOST_DEV DALI_FORCEINLINE constexpr float unpack_l(uint8_t l_in) {
    return static_cast<float>(l_in) * (100.0f / 255.0f);
  }

  static DALI_HOST_DEV DALI_FORCEINLINE constexpr float unpack_ab(uint8_t ab_in) {
    return static_cast<float>(ab_in) - 128.0f;
  }

  static DALI_HOST_DEV DALI_FORCEINLINE constexpr vec3 unpack_lab(vec<3, uint8_t> lab) {
    return {
      unpack_l(lab[0]),
      unpack_ab(lab[1]),
      unpack_ab(lab[2])
    };
  }

 public:
  template <typename Output = uint8_t, typename Input = uint8_t>
  static DALI_HOST_DEV DALI_FORCEINLINE vec<3, Output> rgb_to_lab(vec<3, Input> rgb) {
    static_assert(std::is_same<Output, uint8_t>::value && std::is_same<Input, uint8_t>::value,
                  "CIELAB conversion currently supports only uint8_t input and output");
    vec3 srgb = {
      ConvertNorm<float>(rgb[0]),
      ConvertNorm<float>(rgb[1]),
      ConvertNorm<float>(rgb[2])
    };
    vec3 lin = {
      srgb_to_linear(srgb[0]),
      srgb_to_linear(srgb[1]),
      srgb_to_linear(srgb[2])
    };
    constexpr vec3 to_xyz_x = C::to_xyz_x;
    constexpr vec3 to_xyz_y = C::to_xyz_y;
    constexpr vec3 to_xyz_z = C::to_xyz_z;
    float fx = lab_f(dot(to_xyz_x, lin));
    float fy = lab_f(dot(to_xyz_y, lin));
    float fz = lab_f(dot(to_xyz_z, lin));

    float L = clamp(116.0f * fy - 16.0f, 0.0f, 100.0f);
    float a = 500.0f * (fx - fy);
    float b = 200.0f * (fy - fz);
    return pack_lab({L, a, b});
  }

  template <typename Output = uint8_t, typename Input = uint8_t>
  static DALI_HOST_DEV DALI_FORCEINLINE vec<3, Output> lab_to_rgb(vec<3, Input> lab) {
    static_assert(std::is_same<Output, uint8_t>::value && std::is_same<Input, uint8_t>::value,
                  "CIELAB conversion currently supports only uint8_t input and output");
    vec3 c_lab = unpack_lab(lab);
    float fy = (c_lab[0] + 16.0f) * (1.0f / 116.0f);
    float fx = fy + c_lab[1] * (1.0f / 500.0f);
    float fz = fy - c_lab[2] * (1.0f / 200.0f);

    vec3 xyz = {
      lab_f_inv(fx),
      lab_f_inv(fy),
      lab_f_inv(fz)
    };
    constexpr vec3 from_xyz_r = C::from_xyz_r;
    constexpr vec3 from_xyz_g = C::from_xyz_g;
    constexpr vec3 from_xyz_b = C::from_xyz_b;
    float r = linear_to_srgb(dot(from_xyz_r, xyz));
    float g = linear_to_srgb(dot(from_xyz_g, xyz));
    float b = linear_to_srgb(dot(from_xyz_b, xyz));
    return {
      ConvertSatNorm<Output>(r),
      ConvertSatNorm<Output>(g),
      ConvertSatNorm<Output>(b)
    };
  }

};  // struct cielab_method

}  // namespace detail

// CIELAB with D65 reference white point (the standard sRGB illuminant)
struct cielab_d65 : detail::cielab_method<detail::d65> {};

// CIELAB with D50 reference white point (Bradford-adapted from sRGB)
struct cielab_d50 : detail::cielab_method<detail::d50> {};

}  // namespace color
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
