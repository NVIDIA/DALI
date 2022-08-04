// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <limits>
#include <random>
#include <tuple>
#include <utility>
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
      scale = max_value<Out>() * 219 / 255;

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
  std::tuple<jpeg, uint8_t, uint8_t>,
  std::tuple<jpeg, uint8_t, uint16_t>,
  std::tuple<jpeg, uint8_t, float>,
  std::tuple<jpeg, uint16_t, uint8_t>,
  std::tuple<jpeg, uint16_t, uint16_t>,
  std::tuple<jpeg, uint16_t, float>,
  std::tuple<jpeg, float, uint8_t>,
  std::tuple<jpeg, float, uint16_t>,
  std::tuple<jpeg, float, float>
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


}  // namespace test
}  // namespace color
}  // namespace kernels
}  // namespace dali
