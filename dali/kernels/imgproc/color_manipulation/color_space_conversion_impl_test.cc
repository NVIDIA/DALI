// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/format.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"

namespace dali {
namespace kernels {
namespace color {
namespace test {

template <typename T>
constexpr int integer_bits = std::is_integral_v<T>
                           ? sizeof(T) * 8 - std::is_signed_v<T>
                           : 0;  // floating point has 0 integer bits

struct itu_ref {
  template <typename T>
  static constexpr float y_bias() {
    constexpr int bits = integer_bits<T>;
    return bits ? static_cast<float>(1 << (bits - 4)) : 1.0f / 16;
  }

  template <typename Out>
  static constexpr float gray_to_y(float g) {
    double scale = 219.0 / 255;

    if (std::is_integral_v<Out>)
      scale = max_value<Out>() * 219.0 / 255;  // floating point division - exact for any Out

    return scale * g + y_bias<Out>();
  }

  /**
   * @brief Converts a normalized floating point RGB vector to YCbCr with headroom and footroom
   *
   * The output is scaled to the dynamic range of Out, but it's kept in floating point.
   */
  template <typename Out>
  static constexpr dvec3 rgb_to_ycbcr(dvec3 rgb) {
    double r = rgb[0];
    double g = rgb[1];
    double b = rgb[2];
    double y = 0.299 * r + 0.587 * g + 0.114 * b;
    double cr = (r - y) * (0.5 / (1 - 0.299));  // scale so that R has a weight of 0.5
    double cb = (b - y) * (0.5 / (1 - 0.114));  // scale so that B has a weight of 0.5

    double ybias = 1/16.0;
    double cbias = 0.5;
    double scale = 1;
    if constexpr (std::is_integral_v<Out>) {
      int out_bits = integer_bits<Out>;
      ybias = 1_i64 << (out_bits - 4);
      cbias = 1_i64 << (out_bits - 1);
      scale = max_value<Out>();
    }

    return {
      y  * scale * 219/255 + ybias,
      cb * scale * 224/255 + cbias,
      cr * scale * 224/255 + cbias
    };
  }
};

static_assert(itu_ref::y_bias<float>() == 1.0f / 16);
static_assert(itu_ref::y_bias<uint8_t>() == 16);
static_assert(itu_ref::y_bias<uint16_t>() == 4096);
static_assert(itu_ref::y_bias<int8_t>() == 8);
static_assert(itu_ref::y_bias<int16_t>() == 2048);

struct jpeg_ref {
  template <typename T>
  static float y_bias() {
    return 0;
  }

  template <typename Out>
  static constexpr float gray_to_y(float g) {
    double scale = 1;

    if (std::is_integral_v<Out>)
      scale = max_value<Out>();

    return scale * g;
  }

  /**
   * @brief Converts a normalized floating point RGB vector to YCbCr with headroom and footroom
   *
   * The output is scaled to the dynamic range of Out, but it's kept in floating point.
   */
  template <typename Out>
  static constexpr dvec3 rgb_to_ycbcr(dvec3 rgb) {
    double r = rgb[0];
    double g = rgb[1];
    double b = rgb[2];
    double y = 0.299 * r + 0.587 * g + 0.114 * b;
    double cr = (r - y) * (0.5 / (1 - 0.299));  // scale so that R has a weight of 0.5
    double cb = (b - y) * (0.5 / (1 - 0.114));  // scale so that B has a weight of 0.5

    double cbias = 0.5f;
    double scale = 1;
    if constexpr (std::is_integral_v<Out>) {
      int out_bits = integer_bits<Out>;
      cbias = 1_i64 << (out_bits - 1);
      scale = max_value<Out>();
    }


    return {
      y  * scale,
      cb * scale + cbias,
      cr * scale + cbias
    };
  }
};

static_assert(vec3(itu_ref::rgb_to_ycbcr<uint8_t>({0, 0, 0})) == vec3(16, 128, 128));
static_assert(vec3(itu_ref::rgb_to_ycbcr<uint8_t>({1, 1, 1})) == vec3(235, 128, 128));
static_assert(itu_ref::gray_to_y<uint8_t>(0) == 16);
static_assert(itu_ref::gray_to_y<uint8_t>(1) == 235);
static_assert(itu_ref::gray_to_y<float>(0) == 0.0625f);
static_assert(itu_ref::gray_to_y<float>(1) == static_cast<float>(0.0625 + 219.0 / 255));

static_assert(vec3(jpeg_ref::rgb_to_ycbcr<uint8_t>({0, 0, 0})) == vec3(0, 128, 128));
static_assert(vec3(jpeg_ref::rgb_to_ycbcr<uint8_t>({1, 1, 1})) == vec3(255, 128, 128));
static_assert(jpeg_ref::gray_to_y<uint8_t>(0) == 0);
static_assert(jpeg_ref::gray_to_y<uint8_t>(1) == 255);

template <typename Method>
struct RefMethod;

template <>
struct RefMethod<itu_r_bt_601> {
  using type = itu_ref;
};

template <>
struct RefMethod<jpeg> {
  using type = jpeg_ref;
};

template <typename Method>
using ref_method = typename RefMethod<Method>::type;


TEST(ColorSpaceConversionTest, ITU_R_BT601_RGB2YCbCr_u8) {
  using method = itu_r_bt_601;;
  auto rgb_to_y  = [](auto rgb) { return method::rgb_to_y<uint8_t>(rgb); };
  auto rgb_to_cb = [](auto rgb) { return method::rgb_to_cb<uint8_t>(rgb); };
  auto rgb_to_cr = [](auto rgb) { return method::rgb_to_cr<uint8_t>(rgb); };
  EXPECT_EQ(rgb_to_y(u8vec3{0, 0, 0}), 16);
  EXPECT_EQ(rgb_to_y(u8vec3{255, 255, 255}), 235);
  EXPECT_EQ(rgb_to_y(u8vec3{255, 0, 0}), 81);
  EXPECT_EQ(rgb_to_y(u8vec3{0, 255, 0}), 145);
  EXPECT_EQ(rgb_to_y(u8vec3{0, 0, 255}), 41);

  EXPECT_EQ(rgb_to_cb(u8vec3{0, 0, 0}), 128);
  EXPECT_EQ(rgb_to_cr(u8vec3{0, 0, 0}), 128);
  EXPECT_EQ(rgb_to_cb(u8vec3{255, 255, 255}), 128);
  EXPECT_EQ(rgb_to_cr(u8vec3{255, 255, 255}), 128);

  EXPECT_EQ(rgb_to_cb(u8vec3{255,   0,   0}), 90);
  EXPECT_EQ(rgb_to_cr(u8vec3{255,   0,   0}), 240);
  EXPECT_EQ(rgb_to_cb(u8vec3{255, 255,   0}), 16);
  EXPECT_EQ(rgb_to_cr(u8vec3{255, 255,   0}), 146);
  EXPECT_EQ(rgb_to_cb(u8vec3{  0, 255,   0}), 54);
  EXPECT_EQ(rgb_to_cr(u8vec3{  0, 255,   0}), 34);
  EXPECT_EQ(rgb_to_cb(u8vec3{  0, 255, 255}), 166);
  EXPECT_EQ(rgb_to_cr(u8vec3{  0, 255, 255}), 16);
  EXPECT_EQ(rgb_to_cb(u8vec3{  0,   0, 255}), 240);
  EXPECT_EQ(rgb_to_cr(u8vec3{  0,   0, 255}), 110);
  EXPECT_EQ(rgb_to_cb(u8vec3{255,   0, 255}), 202);
  EXPECT_EQ(rgb_to_cr(u8vec3{255,   0, 255}), 222);
}

TEST(ColorSpaceConversionTest, JPEG_RGB2YCbCr_u8) {
  using method = jpeg;;
  auto rgb_to_y  = [](auto rgb) { return method::rgb_to_y<uint8_t>(rgb); };
  auto rgb_to_cb = [](auto rgb) { return method::rgb_to_cb<uint8_t>(rgb); };
  auto rgb_to_cr = [](auto rgb) { return method::rgb_to_cr<uint8_t>(rgb); };
  EXPECT_EQ(rgb_to_y(u8vec3{0, 0, 0}), 0);
  EXPECT_EQ(rgb_to_y(u8vec3{255, 255, 255}), 255);
  EXPECT_EQ(rgb_to_y(u8vec3{255, 0, 0}), 76);
  EXPECT_EQ(rgb_to_y(u8vec3{0, 255, 0}), 150);
  EXPECT_EQ(rgb_to_y(u8vec3{0, 0, 255}), 29);

  EXPECT_EQ(rgb_to_cb(u8vec3{0, 0, 0}), 128);
  EXPECT_EQ(rgb_to_cr(u8vec3{0, 0, 0}), 128);
  EXPECT_EQ(rgb_to_cb(u8vec3{255, 255, 255}), 128);
  EXPECT_EQ(rgb_to_cr(u8vec3{255, 255, 255}), 128);

  EXPECT_EQ(rgb_to_cb(u8vec3{255,   0,   0}), 85);
  EXPECT_EQ(rgb_to_cr(u8vec3{255,   0,   0}), 255);
  EXPECT_EQ(rgb_to_cb(u8vec3{255, 255,   0}), 1);
  EXPECT_EQ(rgb_to_cr(u8vec3{255, 255,   0}), 149);
  EXPECT_EQ(rgb_to_cb(u8vec3{  0, 255,   0}), 44);
  EXPECT_EQ(rgb_to_cr(u8vec3{  0, 255,   0}), 21);
  EXPECT_EQ(rgb_to_cb(u8vec3{  0, 255, 255}), 171);
  EXPECT_EQ(rgb_to_cr(u8vec3{  0, 255, 255}), 1);
  EXPECT_EQ(rgb_to_cb(u8vec3{  0,   0, 255}), 255);
  EXPECT_EQ(rgb_to_cr(u8vec3{  0,   0, 255}), 107);
  EXPECT_EQ(rgb_to_cb(u8vec3{255,   0, 255}), 212);
  EXPECT_EQ(rgb_to_cr(u8vec3{255,   0, 255}), 235);
}

template <typename TypeParam>
struct ColorSpaceConversionTypedTest : ::testing::Test {};

using Types = ::testing::Types<
  std::tuple<itu_r_bt_601, uint8_t, uint8_t>,
  std::tuple<itu_r_bt_601, uint8_t, uint16_t>,
  std::tuple<itu_r_bt_601, uint8_t, float>,
  std::tuple<itu_r_bt_601, uint16_t, uint8_t>,
  std::tuple<itu_r_bt_601, uint16_t, uint16_t>,
  std::tuple<itu_r_bt_601, uint16_t, float>,
  std::tuple<itu_r_bt_601, float, uint8_t>,
  std::tuple<itu_r_bt_601, float, uint16_t>,
  std::tuple<itu_r_bt_601, float, float>,
  // signed types (with non-negative values only - see ColorSpaceConversionAnyTypeTest for more)
  std::tuple<itu_r_bt_601, uint8_t, int8_t>,
  std::tuple<itu_r_bt_601, uint8_t, int16_t>,
  std::tuple<itu_r_bt_601, uint8_t, int32_t>,
  std::tuple<itu_r_bt_601, int8_t, int8_t>,
  std::tuple<itu_r_bt_601, int16_t, uint8_t>,
  std::tuple<itu_r_bt_601, int16_t, int16_t>,
  std::tuple<itu_r_bt_601, int16_t, float>,
  std::tuple<itu_r_bt_601, float, int16_t>,
  std::tuple<jpeg, uint8_t, uint8_t>,
  std::tuple<jpeg, uint8_t, uint16_t>,
  std::tuple<jpeg, uint8_t, float>,
  std::tuple<jpeg, uint16_t, uint8_t>,
  std::tuple<jpeg, uint16_t, uint16_t>,
  std::tuple<jpeg, uint16_t, float>,
  std::tuple<jpeg, float, uint8_t>,
  std::tuple<jpeg, float, uint16_t>,
  std::tuple<jpeg, float, float>,
  std::tuple<jpeg, uint8_t, int8_t>,
  std::tuple<jpeg, uint8_t, int16_t>,
  std::tuple<jpeg, uint8_t, int32_t>,
  std::tuple<jpeg, int8_t, int8_t>,
  std::tuple<jpeg, int16_t, uint8_t>,
  std::tuple<jpeg, int16_t, int16_t>,
  std::tuple<jpeg, int16_t, float>,
  std::tuple<jpeg, float, int16_t>
>;

TYPED_TEST_SUITE(ColorSpaceConversionTypedTest, Types);

TYPED_TEST(ColorSpaceConversionTypedTest, RGB_YCbCr_BothWays) {
  using Method = typename std::tuple_element_t<0, TypeParam>;
  using Out = typename std::tuple_element_t<1, TypeParam>;
  using In = typename std::tuple_element_t<2, TypeParam>;

  using ref = ref_method<Method>;

  constexpr int bits = integer_bits<Out>;

  // epsilon for forward conversion (rgb -> ycbcr)
  double eps = std::is_integral_v<Out> ? 0.52 : 1e-3;

  // epsilon for reverse conversion (rgb -> ycbcr - >rgb)
  double reverse_eps = 1e-3;
  if (std::is_integral_v<In> && std::is_integral_v<Out>) {
    // if we have two integers, we may lose precision when the input has more bits than the output
    reverse_eps = std::max(2.0, 2.0 * max_value<In>() / max_value<Out>());
  } else if (std::is_integral_v<In> || std::is_integral_v<Out>) {
    // we must accommodate for an off-by-one error when converting back and forth
    reverse_eps = 1;
  }

  auto make_rgb = [](double r, double g, double b) {
    return vec<3, In>(ConvertSatNorm<In>(r), ConvertSatNorm<In>(g), ConvertSatNorm<In>(b));
  };
  auto make_rgb_norm = [](double r, double g, double b) {
    return dvec<3>(r, g, b);
  };


  auto check = [&](double r, double g, double b) {
    auto rgb = make_rgb(r, g, b);
    // calculate the reference, following the ITU-R procedure (with dynamic range scaling or not)
    auto ref_ycbcr = ref::template rgb_to_ycbcr<Out>(make_rgb_norm(r, g, b));
    // calculate the output using current method
    auto ycbcr = Method::template rgb_to_ycbcr<Out, In>(rgb);
    EXPECT_NEAR(ycbcr[0], ref_ycbcr[0], eps) << "RGB = " << vec3(rgb);
    EXPECT_NEAR(ycbcr[1], ref_ycbcr[1], eps) << "RGB = " << vec3(rgb);
    EXPECT_NEAR(ycbcr[2], ref_ycbcr[2], eps) << "RGB = " << vec3(rgb);

    // go back to RGB - must be close to the original value
    auto reverse_rgb = Method::template ycbcr_to_rgb<In, Out>(ycbcr);
    EXPECT_NEAR(reverse_rgb[0], rgb[0], reverse_eps);
    EXPECT_NEAR(reverse_rgb[1], rgb[1], reverse_eps);
    EXPECT_NEAR(reverse_rgb[2], rgb[2], reverse_eps);
  };

  double h = 1.0 * ConvertNorm<In>(0.5) / ConvertNorm<In>(1.0);

  // black
  check(0.0, 0.0, 0.0);
  // gray
  check(h, h, h);
  // white
  check(1.0, 1.0, 1.0);

  // red
  check(1.0, 0.0, 0.0);
  // green
  check(0.0, 1.0, 0.0);
  // blue
  check(0.0, 0.0, 1.0);

  // yellow
  check(1.0, 1.0, 0.0);
  // cyan
  check(0.0, 1.0, 1.0);
  // magenta
  check(1.0, 0.0, 1.0);

  // some random colors
  std::mt19937_64 rng(12345);
  std::conditional_t<std::is_integral_v<In>,
    std::uniform_int_distribution<int64_t>,
    std::uniform_real_distribution<double>
  > dist(0, std::is_integral_v<In> ? max_value<In>() : 1);
  for (int i = 0; i < 1000; i++) {
    check(ConvertNorm<double>(In(dist(rng))),
          ConvertNorm<double>(In(dist(rng))),
          ConvertNorm<double>(In(dist(rng))));
  }
}

TYPED_TEST(ColorSpaceConversionTypedTest, Gray_Y_BothWays) {
  using Method = typename std::tuple_element_t<0, TypeParam>;
  using Out = typename std::tuple_element_t<1, TypeParam>;
  using In = typename std::tuple_element_t<2, TypeParam>;
  using ref = ref_method<Method>;

  double eps = std::is_integral_v<Out> ? 0.52 : 1e-3;

  double reverse_eps = 1e-3;
  if (std::is_integral_v<In> && std::is_integral_v<Out>) {
    // if we have two integers, we may lose precision when the input has more bits than the output
    reverse_eps = std::max(2.0, 2.0 * max_value<In>() / max_value<Out>());
  } else if (std::is_integral_v<In> || std::is_integral_v<Out>) {
    // we must accommodate for an off-by-one error when converting back and forth
    reverse_eps = 1;
  }

  // some random colors
  std::mt19937_64 rng(12345);
  std::conditional_t<std::is_integral_v<In>,
    std::uniform_int_distribution<In>,
    std::uniform_real_distribution<float>
  > dist(0, std::is_integral_v<In> ? max_value<In>() : 1);
  for (int i = 0; i < 1000; i++) {
    In gray = dist(rng);
    float gray_norm = ConvertNorm<float>(gray);
    Out y = Method::template gray_to_y<Out>(gray);
    float ref_y = ref::template gray_to_y<Out>(gray_norm);
    EXPECT_NEAR(y, ref_y, eps);
    In rev_gray = Method::template y_to_gray<In>(y);
    EXPECT_NEAR(gray, rev_gray, reverse_eps);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Arbitrary (signed/unsigned integer, floating point) Input x Output type combinations
//
// The convention verified below is the one documented in color_space_conversion_impl.h:
//  * an unsigned integer sample is UNORM:  x = v / max_value<In>(),          in [0, 1]
//  * a signed integer sample is SNORM:     x = v / max_value<In>(),          in [min/max, 1]
//    (min_value<In>() is one code below -max_value<In>(); it is clamped to -1 only when Output
//    is an integer type, matching ConvertSatNorm<float> and the GPU decode path, which leave it
//    unclamped for floating point Output)
//  * a floating point sample is used as-is
//  * before the color transform, the normalized input is saturated to the range that
//    ConvertSatNorm<Output> can represent: [0, 1] for unsigned Output, [-1, 1] for signed
//    integer Output and unbounded for floating point Output.
//    This makes the color conversion equivalent to ConvertSatNorm<Output>(input) followed by the
//    color transform in Output type (which is what the GPU image decoder path does), but without
//    the intermediate rounding.
//  * the transform itself is the ITU-R BT.601 / JPEG matrix, computed on normalized values and
//    scaled to the dynamic range of Output (see itu_ref and jpeg_ref above).
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace any_ref {

template <typename T>
constexpr double unit() {
  return std::is_integral_v<T> ? static_cast<double>(max_value<T>()) : 1.0;
}

template <typename T>
constexpr double ybias() {  // ITU-R BT.601 footroom, in the units of T
  return std::is_integral_v<T> ? static_cast<double>(1_i64 << (integer_bits<T> - 4)) : 1.0 / 16;
}

template <typename T>
constexpr double cbias() {  // chroma offset, in the units of T
  return std::is_integral_v<T> ? static_cast<double>(1_i64 << (integer_bits<T> - 1)) : 0.5;
}

/**
 * @brief Normalizes a raw sample (UNORM/SNORM/float) and saturates it to the range representable
 *        by `Out`.
 */
template <typename Out, typename In>
constexpr double norm(In v) {
  double x = std::is_integral_v<In> ? static_cast<double>(v) / max_value<In>()
                                    : static_cast<double>(v);
  // SNORM: min_value<In>() is clamped to -1, but only when Out is an integer type - matching
  // ConvertSatNorm<float>/ConvertNorm<float> and the GPU decode path, which leave it unclamped
  // (e.g. int8_t(-128) normalizes to -128/127, not -1) when Out is floating point.
  if (std::is_integral_v<In> && std::is_signed_v<In> && std::is_integral_v<Out> && x < -1)
    x = -1;
  if (std::is_integral_v<Out>) {
    double lo = std::is_unsigned_v<Out> ? 0.0 : -1.0;
    x = x < lo ? lo : x > 1 ? 1 : x;
  }
  return x;
}

/** @brief Same as norm, but scaled back to the units of In (e.g. for YCbCr inputs with biases) */
template <typename Out, typename In>
constexpr double sat(In v) {
  return norm<Out, In>(v) * unit<In>();
}

/** @brief Clamps the (unrounded) reference to the range of Out */
template <typename Out>
constexpr double finish(double v) {
  if (std::is_integral_v<Out>) {
    double lo = min_value<Out>(), hi = max_value<Out>();
    return v < lo ? lo : v > hi ? hi : v;
  }
  return v;
}

// Luma/chroma from normalized RGB
inline double luma(dvec3 rgb) { return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]; }
inline double chroma_b(dvec3 rgb) { return (rgb[2] - luma(rgb)) * (0.5 / (1 - 0.114)); }
inline double chroma_r(dvec3 rgb) { return (rgb[0] - luma(rgb)) * (0.5 / (1 - 0.299)); }

// Normalized RGB from normalized luma/chroma (chroma centered at 0)
inline dvec3 rgb_from_ycc(double y, double cb, double cr) {
  double r = y + 2 * (1 - 0.299) * cr;
  double b = y + 2 * (1 - 0.114) * cb;
  double g = (y - 0.299 * r - 0.114 * b) / 0.587;
  return {r, g, b};
}

template <typename Method>
struct method_ref;

template <>
struct method_ref<itu_r_bt_601> {
  static constexpr double yscale = 219.0 / 255;
  static constexpr double cscale = 224.0 / 255;
  template <typename T> static constexpr double ybias() { return any_ref::ybias<T>(); }
};

template <>
struct method_ref<jpeg> {
  static constexpr double yscale = 1;
  static constexpr double cscale = 1;
  template <typename T> static constexpr double ybias() { return 0; }
};

template <typename Method, typename Out, typename In>
dvec3 rgb_to_ycbcr(vec<3, In> in) {
  using M = method_ref<Method>;
  dvec3 rgb(norm<Out>(in[0]), norm<Out>(in[1]), norm<Out>(in[2]));
  return {
    finish<Out>(luma(rgb) * unit<Out>() * M::yscale + M::template ybias<Out>()),
    finish<Out>(chroma_b(rgb) * unit<Out>() * M::cscale + cbias<Out>()),
    finish<Out>(chroma_r(rgb) * unit<Out>() * M::cscale + cbias<Out>())
  };
}

template <typename Method, typename Out, typename In>
dvec3 ycbcr_to_rgb(vec<3, In> in) {
  using M = method_ref<Method>;
  double y  = (sat<Out>(in[0]) - M::template ybias<In>()) / (unit<In>() * M::yscale);
  double cb = (sat<Out>(in[1]) - cbias<In>()) / (unit<In>() * M::cscale);
  double cr = (sat<Out>(in[2]) - cbias<In>()) / (unit<In>() * M::cscale);
  dvec3 rgb = rgb_from_ycc(y, cb, cr);
  return {
    finish<Out>(rgb[0] * unit<Out>()),
    finish<Out>(rgb[1] * unit<Out>()),
    finish<Out>(rgb[2] * unit<Out>())
  };
}

template <typename Method, typename Out, typename In>
double gray_to_y(In gray) {
  using M = method_ref<Method>;
  return finish<Out>(norm<Out>(gray) * unit<Out>() * M::yscale + M::template ybias<Out>());
}

template <typename Method, typename Out, typename In>
double y_to_gray(In y) {
  using M = method_ref<Method>;
  double g = (sat<Out>(y) - M::template ybias<In>()) / (unit<In>() * M::yscale);
  return finish<Out>(g * unit<Out>());
}

}  // namespace any_ref

// Sanity checks of the reference itself
static_assert(any_ref::norm<uint8_t, int8_t>(int8_t(-128)) == 0);
static_assert(any_ref::norm<int8_t, int8_t>(int8_t(-128)) == -1);
static_assert(any_ref::norm<int8_t, int8_t>(int8_t(-127)) == -1);
static_assert(any_ref::norm<float, int16_t>(int16_t(-32768)) == -32768.0 / 32767);
static_assert(any_ref::norm<float, int16_t>(int16_t(32767)) == 1);
static_assert(any_ref::norm<uint8_t, float>(1.5f) == 1);
static_assert(any_ref::norm<uint8_t, float>(-0.5f) == 0);
static_assert(any_ref::norm<int16_t, float>(-1.5f) == -1);
static_assert(any_ref::norm<float, float>(-1.5f) == -1.5);
static_assert(any_ref::norm<float, float>(1.5f) == 1.5);
static_assert(any_ref::ybias<int32_t>() == (1 << 27));
static_assert(any_ref::cbias<uint32_t>() == 2147483648.0);

// Compile-time checks of the input saturation used by the implementation
static_assert(detail::saturate_norm<uint8_t>(int8_t(-128)) == 0);
static_assert(detail::saturate_norm<uint8_t>(int8_t(-1)) == 0);
static_assert(detail::saturate_norm<uint8_t>(int8_t(5)) == 5);
static_assert(detail::saturate_norm<uint16_t>(min_value<int32_t>()) == 0);
static_assert(detail::saturate_norm<int8_t>(int8_t(-128)) == -127);
static_assert(detail::saturate_norm<int32_t>(int8_t(-128)) == -127);
// ...but not when Output is floating point: min_value<Input>() is passed through unchanged.
static_assert(detail::saturate_norm<float>(int16_t(-32768)) == -32768);
static_assert(detail::saturate_norm<float>(int16_t(-32767)) == -32767);
static_assert(detail::saturate_norm<float>(int32_t(-5)) == -5);
static_assert(detail::saturate_norm<int8_t>(max_value<int32_t>()) == max_value<int32_t>());
// Unsigned inputs are never modified - this guarantees unchanged results for unsigned inputs
static_assert(detail::saturate_norm<uint8_t>(uint8_t(0)) == 0);
static_assert(detail::saturate_norm<uint8_t>(uint8_t(255)) == 255);
static_assert(detail::saturate_norm<int8_t>(uint32_t(0xffffffffu)) == 0xffffffffu);
static_assert(detail::saturate_norm<float>(uint16_t(0xffff)) == 0xffff);
// Floating point inputs are clamped to the range representable by integral outputs
static_assert(detail::saturate_norm<uint8_t>(1.5f) == 1.0f);
static_assert(detail::saturate_norm<uint8_t>(-0.5f) == 0.0f);
static_assert(detail::saturate_norm<uint8_t>(0.25f) == 0.25f);
static_assert(detail::saturate_norm<int16_t>(-1.5f) == -1.0f);
static_assert(detail::saturate_norm<int16_t>(-0.5f) == -0.5f);
static_assert(detail::saturate_norm<int32_t>(2.0f) == 1.0f);
// ...but not for floating point outputs
static_assert(detail::saturate_norm<float>(-1.5f) == -1.5f);
static_assert(detail::saturate_norm<float>(1.5f) == 1.5f);

namespace {

template <typename T>
const char *type_name() {
  if (std::is_same_v<T, uint8_t>) return "uint8";
  if (std::is_same_v<T, int8_t>) return "int8";
  if (std::is_same_v<T, uint16_t>) return "uint16";
  if (std::is_same_v<T, int16_t>) return "int16";
  if (std::is_same_v<T, uint32_t>) return "uint32";
  if (std::is_same_v<T, int32_t>) return "int32";
  if (std::is_same_v<T, float>) return "float";
  return "?";
}

template <typename Method>
const char *method_name() {
  return std::is_same_v<Method, itu_r_bt_601> ? "itu_r_bt_601" : "jpeg";
}

/**
 * @brief Allowed absolute difference between the implementation and the (unrounded) reference.
 *
 * Integral outputs are rounded (0.5); the computations are carried out in single precision, which
 * for 32-bit outputs introduces an error proportional to the dynamic range of the output.
 */
template <typename Out>
double tolerance(double amplification = 1) {
  if (std::is_integral_v<Out>)
    return 0.5 + 1e-3 + amplification * any_ref::unit<Out>() * std::ldexp(1.0, -20);
  else
    return amplification * 2e-6;
}

/**
 * @brief Interesting values for a type: extremes, zero, +/-1 code, and some mid-range values
 */
template <typename T>
std::vector<T> edge_values() {
  if constexpr (std::is_integral_v<T>) {
    std::vector<T> v = {min_value<T>(), T(min_value<T>() + 1), T(0), T(1),
                        T(max_value<T>() / 3), T(max_value<T>() / 2 + 1),
                        T(max_value<T>() - 1), max_value<T>()};
    if (std::is_signed_v<T>) {
      v.push_back(T(-1));
      v.push_back(T(-(max_value<T>() / 3)));
      v.push_back(T(-max_value<T>()));
    }
    return v;
  } else {
    return {-2.0f, -1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f, 1.5f};
  }
}

/** @brief Uniformly distributed values over the whole range of T ([-2, 2] for floats) */
template <typename T>
std::vector<T> random_values(int n, std::mt19937_64 &rng) {
  std::vector<T> v(n);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int64_t> dist(min_value<T>(), max_value<T>());
    for (auto &x : v) x = static_cast<T>(dist(rng));
  } else {
    std::uniform_real_distribution<double> dist(-2, 2);
    for (auto &x : v) x = static_cast<T>(dist(rng));
  }
  return v;
}

/** @brief All combinations of the edge values and some random triplets */
template <typename T>
std::vector<vec<3, T>> test_triplets() {
  std::vector<vec<3, T>> out;
  auto e = edge_values<T>();
  for (T a : e)
    for (T b : e)
      for (T c : e)
        out.push_back({a, b, c});
  std::mt19937_64 rng(4321);
  auto r = random_values<T>(3 * 300, rng);
  for (size_t i = 0; i < r.size(); i += 3)
    out.push_back({r[i], r[i + 1], r[i + 2]});
  return out;
}

template <typename T>
std::vector<T> test_scalars() {
  auto v = edge_values<T>();
  std::mt19937_64 rng(1234);
  auto r = random_values<T>(1000, rng);
  v.insert(v.end(), r.begin(), r.end());
  return v;
}

template <typename T>
dvec3 as_dvec(vec<3, T> v) {
  return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

using AllTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float>;

/** @brief Calls f(Method{}, Out{}) for all output types */
template <typename F>
void for_all_outputs(F &&f) {
  std::apply([&](auto... out) { (f(out), ...); }, AllTypes{});
}

/** @brief Calls f(Method{}, Out{}) for all methods and output types */
template <typename F>
void for_all_methods_and_outputs(F &&f) {
  for_all_outputs([&](auto out) { f(itu_r_bt_601{}, out); });
  for_all_outputs([&](auto out) { f(jpeg{}, out); });
}

}  // namespace

template <typename In>
struct ColorSpaceConversionAnyTypeTest : ::testing::Test {};

using AnyInputTypes = ::testing::Types<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
                                       float>;
TYPED_TEST_SUITE(ColorSpaceConversionAnyTypeTest, AnyInputTypes);

TYPED_TEST(ColorSpaceConversionAnyTypeTest, RGB_to_YCbCr) {
  using In = TypeParam;
  auto inputs = test_triplets<In>();
  for_all_methods_and_outputs([&](auto method, auto out_type) {
    using Method = decltype(method);
    using Out = decltype(out_type);
    SCOPED_TRACE(make_string(method_name<Method>(), " ", type_name<In>(), " -> ",
                             type_name<Out>()));
    double eps = tolerance<Out>(2);
    for (auto rgb : inputs) {
      dvec3 expected = any_ref::rgb_to_ycbcr<Method, Out>(rgb);
      auto ycbcr = Method::template rgb_to_ycbcr<Out, In>(rgb);
      auto y  = Method::template rgb_to_y<Out, In>(rgb);
      auto cb = Method::template rgb_to_cb<Out, In>(rgb);
      auto cr = Method::template rgb_to_cr<Out, In>(rgb);
      ASSERT_NEAR(ycbcr[0], expected[0], eps) << "RGB = " << as_dvec(rgb);
      ASSERT_NEAR(ycbcr[1], expected[1], eps) << "RGB = " << as_dvec(rgb);
      ASSERT_NEAR(ycbcr[2], expected[2], eps) << "RGB = " << as_dvec(rgb);
      ASSERT_EQ(y, ycbcr[0]) << "RGB = " << as_dvec(rgb);
      ASSERT_EQ(cb, ycbcr[1]) << "RGB = " << as_dvec(rgb);
      ASSERT_EQ(cr, ycbcr[2]) << "RGB = " << as_dvec(rgb);
    }
  });
}

TYPED_TEST(ColorSpaceConversionAnyTypeTest, RGB_to_Gray) {
  using In = TypeParam;
  auto inputs = test_triplets<In>();
  for_all_outputs([&](auto out_type) {
    using Out = decltype(out_type);
    SCOPED_TRACE(make_string(type_name<In>(), " -> ", type_name<Out>()));
    for (auto rgb : inputs) {
      // rgb_to_gray is the JPEG (full range) luma
      double expected = any_ref::rgb_to_ycbcr<jpeg, Out>(rgb)[0];
      ASSERT_NEAR((rgb_to_gray<Out, In>(rgb)), expected, tolerance<Out>(2))
          << "RGB = " << as_dvec(rgb);
    }
  });
}

TYPED_TEST(ColorSpaceConversionAnyTypeTest, YCbCr_to_RGB) {
  using In = TypeParam;
  auto inputs = test_triplets<In>();
  for_all_methods_and_outputs([&](auto method, auto out_type) {
    using Method = decltype(method);
    using Out = decltype(out_type);
    SCOPED_TRACE(make_string(method_name<Method>(), " ", type_name<In>(), " -> ",
                             type_name<Out>()));
    // The inverse matrix has larger coefficients (up to ~2.3 with ITU-R BT.601 scaling)
    double eps = tolerance<Out>(8);
    for (auto ycbcr : inputs) {
      dvec3 expected = any_ref::ycbcr_to_rgb<Method, Out>(ycbcr);
      auto rgb = Method::template ycbcr_to_rgb<Out, In>(ycbcr);
      ASSERT_NEAR(rgb[0], expected[0], eps) << "YCbCr = " << as_dvec(ycbcr);
      ASSERT_NEAR(rgb[1], expected[1], eps) << "YCbCr = " << as_dvec(ycbcr);
      ASSERT_NEAR(rgb[2], expected[2], eps) << "YCbCr = " << as_dvec(ycbcr);
    }
  });
}

TYPED_TEST(ColorSpaceConversionAnyTypeTest, Gray_to_Y) {
  using In = TypeParam;
  auto inputs = test_scalars<In>();
  for_all_methods_and_outputs([&](auto method, auto out_type) {
    using Method = decltype(method);
    using Out = decltype(out_type);
    SCOPED_TRACE(make_string(method_name<Method>(), " ", type_name<In>(), " -> ",
                             type_name<Out>()));
    for (In gray : inputs) {
      double expected = any_ref::gray_to_y<Method, Out>(gray);
      ASSERT_NEAR((Method::template gray_to_y<Out, In>(gray)), expected, tolerance<Out>())
          << "gray = " << static_cast<double>(gray);
    }
  });
}

TYPED_TEST(ColorSpaceConversionAnyTypeTest, Y_to_Gray) {
  using In = TypeParam;
  auto inputs = test_scalars<In>();
  for_all_methods_and_outputs([&](auto method, auto out_type) {
    using Method = decltype(method);
    using Out = decltype(out_type);
    SCOPED_TRACE(make_string(method_name<Method>(), " ", type_name<In>(), " -> ",
                             type_name<Out>()));
    for (In y : inputs) {
      double expected = any_ref::y_to_gray<Method, Out>(y);
      ASSERT_NEAR((Method::template y_to_gray<Out, In>(y)), expected, tolerance<Out>(2))
          << "Y = " << static_cast<double>(y);
    }
  });
}

TYPED_TEST(ColorSpaceConversionAnyTypeTest, Gray_to_YCbCr) {
  using In = TypeParam;
  auto inputs = test_scalars<In>();
  for_all_outputs([&](auto out_type) {
    using Out = decltype(out_type);
    SCOPED_TRACE(make_string(type_name<In>(), " -> ", type_name<Out>()));
    for (In gray : inputs) {
      auto ycbcr = itu_r_bt_601::gray_to_ycbcr<Out, In>(gray);
      double expected_y = any_ref::gray_to_y<itu_r_bt_601, Out>(gray);
      ASSERT_NEAR(ycbcr[0], expected_y, tolerance<Out>()) << "gray = " << double(gray);
      // neutral chroma
      ASSERT_EQ(ycbcr[1], any_ref::cbias<Out>()) << "gray = " << double(gray);
      ASSERT_EQ(ycbcr[2], any_ref::cbias<Out>()) << "gray = " << double(gray);
    }
  });
}

TYPED_TEST(ColorSpaceConversionAnyTypeTest, YCbCr_to_Gray) {
  using In = TypeParam;
  auto inputs = test_triplets<In>();
  for_all_outputs([&](auto out_type) {
    using Out = decltype(out_type);
    SCOPED_TRACE(make_string(type_name<In>(), " -> ", type_name<Out>()));
    for (auto ycbcr : inputs) {
      // Only luma matters
      double expected = any_ref::y_to_gray<itu_r_bt_601, Out>(ycbcr[0]);
      ASSERT_NEAR((itu_r_bt_601::ycbcr_to_gray<Out, In>(ycbcr)), expected, tolerance<Out>(2))
          << "YCbCr = " << as_dvec(ycbcr);
    }
  });
}

/**
 * @brief Converting directly from `In` must be equivalent (up to the intermediate rounding) to
 *        converting to `Out` first with ConvertSatNorm and then running the color conversion
 *        with `Out` as both input and output type.
 *
 * The two-step procedure is what the GPU image decoder does (normalize to the output type first,
 * then convert the color space), so this guarantees consistent CPU and GPU results.
 * It also means that decoding to YCbCr gives the same results as decoding to RGB and then
 * converting the result to YCbCr.
 *
 * This holds for floating point outputs too: ConvertSatNorm<float> does not clamp the SNORM
 * min_value<In>() to -1 either, matching saturate_norm's behavior for a floating point Output.
 */
TYPED_TEST(ColorSpaceConversionAnyTypeTest, EquivalentToConvertSatNormFirst) {
  using In = TypeParam;
  auto triplets = test_triplets<In>();
  auto scalars = test_scalars<In>();
  for_all_methods_and_outputs([&](auto method, auto out_type) {
    using Method = decltype(method);
    using Out = decltype(out_type);
    {
      SCOPED_TRACE(make_string(method_name<Method>(), " ", type_name<In>(), " -> ",
                               type_name<Out>()));
      auto to_out = [](vec<3, In> v) {
        return vec<3, Out>(ConvertSatNorm<Out>(v[0]), ConvertSatNorm<Out>(v[1]),
                           ConvertSatNorm<Out>(v[2]));
      };
      // The intermediate rounding (up to 0.5) is amplified by the conversion matrix
      double slack = any_ref::unit<Out>() * std::ldexp(1.0, -20);
      double fwd_eps = 1.0 + 1e-3 + 4 * slack;
      // For YCbCr inputs, the luma footroom and chroma offsets are powers of 2 (e.g. 16 and 128
      // for 8 bits, 4096 and 32768 for 16 bits), so their normalized values differ slightly
      // between types, e.g. 128/255 vs 32768/65535. The inverse conversion interprets them in the
      // type of the data (In), while the two-step procedure does so in Out - account for that.
      double dy = std::abs(any_ref::ybias<In>() / any_ref::unit<In>() -
                           any_ref::ybias<Out>() / any_ref::unit<Out>());
      double dc = std::abs(any_ref::cbias<In>() / any_ref::unit<In>() -
                           any_ref::cbias<Out>() / any_ref::unit<Out>());
      double inv_eps = 2.5 + 1e-3 + 16 * slack +
                       (255.0 / 219 * dy + 2.02 * 255 / 224 * dc) * any_ref::unit<Out>();
      double y_to_gray_eps = fwd_eps + 0.5 + 255.0 / 219 * dy * any_ref::unit<Out>();
      for (auto v : triplets) {
        auto direct = Method::template rgb_to_ycbcr<Out, In>(v);
        auto two_step = Method::template rgb_to_ycbcr<Out, Out>(to_out(v));
        for (int c = 0; c < 3; c++)
          ASSERT_NEAR(direct[c], two_step[c], fwd_eps) << "RGB = " << as_dvec(v) << " c=" << c;

        auto direct_rgb = Method::template ycbcr_to_rgb<Out, In>(v);
        auto two_step_rgb = Method::template ycbcr_to_rgb<Out, Out>(to_out(v));
        for (int c = 0; c < 3; c++)
          ASSERT_NEAR(direct_rgb[c], two_step_rgb[c], inv_eps)
              << "YCbCr = " << as_dvec(v) << " c=" << c;
      }
      for (In v : scalars) {
        ASSERT_NEAR((Method::template gray_to_y<Out, In>(v)),
                    (Method::template gray_to_y<Out, Out>(ConvertSatNorm<Out>(v))), fwd_eps)
            << "gray = " << static_cast<double>(v);
        ASSERT_NEAR((Method::template y_to_gray<Out, In>(v)),
                    (Method::template y_to_gray<Out, Out>(ConvertSatNorm<Out>(v))),
                    y_to_gray_eps)
            << "Y = " << static_cast<double>(v);
      }
    }
  });
}

/**
 * @brief Hand-computed reference values for signed and cross-type conversions
 */
TEST(ColorSpaceConversionSignedTest, ITU_R_BT601_RGB2YCbCr_Pinned) {
  using method = itu_r_bt_601;
  // A pixel from a signed int32 TIFF, where the CPU and GPU decoders used to disagree.
  // Normalized: R = 0.98824, G = -0.97255 -> 0 (clamped: not representable in uint8),
  // B = 0.99608; Y = 16 + 219 * (0.299 R + 0.114 B) = 105.58, Cb = 202.21, Cr = 220.54
  i32vec3 px = {2122218880, -2088533504, 2139061888};
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int32_t>(px)), u8vec3(106, 202, 221));

  // Zero is black (not mid-gray) - signed samples are SNORM, not offset-binary
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int8_t>({0, 0, 0})), u8vec3(16, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int16_t>({0, 0, 0})), u8vec3(16, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int32_t>({0, 0, 0})), u8vec3(16, 128, 128));
  // Negative values are below black and saturate to black in an unsigned output
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int8_t>({-128, -128, -128})), u8vec3(16, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int16_t>({-1, -1, -1})), u8vec3(16, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int32_t>({min_value<int32_t>(), -1, -12345})),
            u8vec3(16, 128, 128));
  // max_value is white
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int8_t>({127, 127, 127})), u8vec3(235, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int32_t>(i32vec3(max_value<int32_t>()))),
            u8vec3(235, 128, 128));
  // Pure red, with negative (i.e. clamped to 0) green and blue
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int16_t>({32767, -32768, -100})),
            u8vec3(81, 90, 240));

  // Signed output: min_value is clamped to -1 (SNORM), i.e. the same as -max_value.
  // Y = 8 - 127 * 219/255 = -101.07; Cb = Cr = 64
  EXPECT_EQ((method::rgb_to_ycbcr<int8_t, int8_t>({-128, -128, -128})), i8vec3(-101, 64, 64));
  EXPECT_EQ((method::rgb_to_ycbcr<int8_t, int8_t>({-127, -127, -127})), i8vec3(-101, 64, 64));
  // Y = 2048 - 32767 * 219/255 = -26093.07
  EXPECT_EQ((method::rgb_to_ycbcr<int16_t, int8_t>({-128, -128, -128})),
            i16vec3(-26093, 16384, 16384));
  // Floating point output: min_value<int16_t>() (-32768) is NOT clamped to -1 here (unlike for
  // an integer Output): it normalizes to -32768/32767, slightly past -max_value's -1, matching
  // ConvertSatNorm<float> and the GPU decode path. Y = 1/16 - (219/255) * (-32768/32767).
  EXPECT_NEAR((method::rgb_to_y<float, int16_t>({-32768, -32768, -32768})),
              0.0625 - 219.0 / 255 * (32768.0 / 32767), 1e-6);
  // -max_value<int16_t>() (-32767) normalizes to exactly -1: Y = 1/16 - 219/255.
  EXPECT_NEAR((method::rgb_to_y<float, int16_t>({-32767, -32767, -32767})),
              0.0625 - 219.0 / 255, 1e-6);
}

TEST(ColorSpaceConversionSignedTest, JPEG_Pinned) {
  using method = jpeg;
  i32vec3 px = {2122218880, -2088533504, 2139061888};
  // Y = 255 * (0.299 * 0.98824 + 0.114 * 0.99608) = 104.30
  // Cb = 128 + 255 * (-0.16874 * 0.98824 + 0.5 * 0.99608) = 212.48
  // Cr = 128 + 255 * (0.5 * 0.98824 - 0.08131 * 0.99608) = 233.35
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int32_t>(px)), u8vec3(104, 212, 233));
  EXPECT_EQ((rgb_to_gray<uint8_t, int32_t>(px)), 104);
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int16_t>({0, 0, 0})), u8vec3(0, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int16_t>({-32768, -1, -32767})), u8vec3(0, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, int16_t>(i16vec3(32767))), u8vec3(255, 128, 128));

  EXPECT_EQ((method::gray_to_y<uint8_t, int16_t>(-1)), 0);
  EXPECT_EQ((method::gray_to_y<uint8_t, int16_t>(32767)), 255);
  EXPECT_EQ((method::y_to_gray<uint8_t, int32_t>(min_value<int32_t>())), 0);
  // SNORM: -128 is clamped to -1 (-32767 in int16), rather than -1.0079 -> -32768
  EXPECT_EQ((method::gray_to_y<int16_t, int8_t>(-128)), -32767);
  EXPECT_EQ((method::y_to_gray<int16_t, int8_t>(-128)), -32767);
}

TEST(ColorSpaceConversionSignedTest, ITU_R_BT601_Inverse_Pinned) {
  using method = itu_r_bt_601;
  // Negative Y is below the footroom of an unsigned output -> black
  EXPECT_EQ((method::ycbcr_to_rgb<uint8_t, int16_t>({-5000, 16384, 16384})), u8vec3(0, 0, 0));
  // Y at footroom (2048 = 16/256 of the positive range) with neutral chroma is black...
  EXPECT_EQ((method::ycbcr_to_rgb<uint8_t, int16_t>({2048, 16384, 16384})), u8vec3(0, 0, 0));
  // ...and max_value saturates to white
  EXPECT_EQ((method::ycbcr_to_rgb<uint8_t, int16_t>({32767, 16384, 16384})),
            u8vec3(255, 255, 255));
  // Negative chroma is clamped to 0 (the most negative chroma representable in an unsigned
  // output) before the conversion: Cb = Cr = -1 behaves as Cb = Cr = 0
  EXPECT_EQ((method::ycbcr_to_rgb<uint8_t, int16_t>({16384, -1, -1})),
            (method::ycbcr_to_rgb<uint8_t, int16_t>({16384, 0, 0})));
  EXPECT_EQ((method::y_to_gray<uint8_t, int16_t>(-1)), 0);
  EXPECT_EQ((method::y_to_gray<uint8_t, int32_t>(min_value<int32_t>())), 0);
  // gray_to_y: 0 is black (footroom), negative is still black for unsigned output
  EXPECT_EQ((method::gray_to_y<uint8_t, int32_t>(0)), 16);
  EXPECT_EQ((method::gray_to_y<uint8_t, int32_t>(-1000)), 16);
  EXPECT_EQ((method::gray_to_y<uint8_t, int32_t>(max_value<int32_t>())), 235);
  EXPECT_EQ((method::gray_to_ycbcr<uint8_t, int8_t>(-128)), u8vec3(16, 128, 128));
  EXPECT_EQ((method::ycbcr_to_gray<uint8_t, int8_t>({-128, 5, 5})), 0);
}

TEST(ColorSpaceConversionSignedTest, FloatInputSaturation) {
  // Out-of-range floating point input is clamped to the range representable by an integral
  // output (like ConvertSatNorm), before the color conversion...
  using method = itu_r_bt_601;
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, float>({1.5f, 1.5f, 1.5f})), u8vec3(235, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, float>({-0.5f, -0.5f, -0.5f})), u8vec3(16, 128, 128));
  EXPECT_EQ((method::rgb_to_ycbcr<uint8_t, float>({1.5f, -0.5f, -0.5f})), u8vec3(81, 90, 240));
  // ...but it's kept as-is for floating point output
  EXPECT_NEAR((method::rgb_to_y<float, float>({1.5f, 1.5f, 1.5f})), 0.0625 + 1.5 * 219 / 255,
              1e-6);
  EXPECT_NEAR((method::rgb_to_y<float, float>({-0.5f, -0.5f, -0.5f})),
              0.0625 - 0.5 * 219 / 255, 1e-6);
}


}  // namespace test
}  // namespace color
}  // namespace kernels
}  // namespace dali
