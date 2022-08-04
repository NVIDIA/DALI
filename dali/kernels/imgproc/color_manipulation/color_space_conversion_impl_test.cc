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

namespace {
  template <typename T>
  constexpr double to_norm_fp(T x) {
    return is_fp_or_half<T>::value ? 1.0 : static_cast<double>(std::numeric_limits<T>::max());
  }

  template <typename T>
  constexpr T from_norm_fp(T x) {
    return is_fp_or_half<T>::value ? 1.0 : static_cast<double>(std::numeric_limits<T>::max() * x);
  }

  template <typename T>
  constexpr float chroma_bias() {
    return 0.5f * detail::bias_scale<T>();
  }

  static_assert(chroma_bias<float>()    == 0.5f);
  static_assert(chroma_bias<uint8_t>()  == 128.0f);
  static_assert(chroma_bias<int8_t>()   == 64.0f);
  static_assert(chroma_bias<uint16_t>() == 32768.0f);
  static_assert(chroma_bias<int16_t>()  == 16384.0f);
  static_assert(chroma_bias<uint32_t>() == static_cast<float>(0x80000000u));
  static_assert(chroma_bias<int32_t>()  == static_cast<float>(0x40000000));

}  // namespace

namespace test {

struct itu_ref {
  template <int out_bits = 0>
  static constexpr vec3 rgb_to_ycbcr(vec3 rgb) {
    double r = rgb[0];
    double g = rgb[1];
    double b = rgb[2];
    double y = 0.299 * r + 0.587 * g + 0.114 * b;
    double cr = (r - y) * (0.5 / (1 - 0.299));  // scale so that R has a weight of 0.5
    double cb = (b - y) * (0.5 / (1 - 0.114));  // scale so that B has a weight of 0.5

    double ybias = 1/16.0;
    double cbias = 0.5f;
    if (out_bits > 0) {
      ybias = 1 << (out_bits - 4);
      cbias = 1 << (out_bits - 1);
    }

    return {
      y  * 219/255 + ybias,
      cb * 224/255 + cbias,
      cr * 224/255 + cbias
    };
  }
};

struct jpeg_ref {
  template <int out_bits = 0>
  static constexpr vec3 rgb_to_ycbcr(vec3 rgb) {
    double r = rgb[0];
    double g = rgb[1];
    double b = rgb[2];
    double y = 0.299 * r + 0.587 * g + 0.114 * b;
    double cr = (r - y) * (0.5 / (1 - 0.299));  // scale so that R has a weight of 0.5
    double cb = (b - y) * (0.5 / (1 - 0.114));  // scale so that B has a weight of 0.5

    double cbias = 0.5f;
    if (out_bits > 0) {
      cbias = 1 << (out_bits - 1);
    }

    return {
      y,
      cb + cbias,
      cr + cbias
    };
  }
};

static_assert(itu_ref::rgb_to_ycbcr<8>({0, 0, 0}) == vec3(16, 128, 128));
static_assert(itu_ref::rgb_to_ycbcr<8>({255, 255, 255}) == vec3(235, 128, 128));

static_assert(jpeg_ref::rgb_to_ycbcr<8>({0, 0, 0}) == vec3(0, 128, 128));
static_assert(jpeg_ref::rgb_to_ycbcr<8>({255, 255, 255}) == vec3(255, 128, 128));

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

TYPED_TEST(ColorSpaceConversionTypedTest, rgb_to_ycbcr) {
  using Method = typename std::tuple_element_t<0, TypeParam>;
  using Out = typename std::tuple_element_t<1, TypeParam>;
  using In = typename std::tuple_element_t<2, TypeParam>;

  using ref = ref_method<Method>;

  constexpr int bits = std::is_integral<Out>::value
    ? sizeof(Out) * 8 - std::is_signed<Out>::value : 0;

  // epsilon for forward conversion (rgb -> ycbcr)
  double eps = std::is_integral<Out>::value ? 0.52 : 1e-2;

  // epsilon for reverse conversion (rgb -> ycbcr - >rgb)
  double reverse_eps = 1e-3;
  if (std::is_integral<In>::value && std::is_integral<Out>::value) {
    // if we have two integers, we may lose precision when the input has more bits than the output
    reverse_eps = std::max(2.0, 2.0 * max_value<In>() / max_value<Out>());
  } else if (std::is_integral<In>::value || std::is_integral<Out>::value) {
    // we must accommodate for an off-by-one error when converting back and forth
    reverse_eps = 1;
  }

  auto make_rgb = [](float r, float g, float b) {
    return vec<3, In>(ConvertSatNorm<In>(r), ConvertSatNorm<In>(g), ConvertSatNorm<In>(b));
  };
  auto make_rgb_ref = [](float r, float g, float b) {
    return vec<3>(r, g, b) * detail::scale_factor<float, Out>();
  };


  auto check = [&](float r, float g, float b) {
    auto rgb = make_rgb(r, g, b);
    // calculate the reference, following the ITU-R procedure (with dynamic range scaling or not)
    auto ref_ycbcr = ref::template rgb_to_ycbcr<bits>(make_rgb_ref(r, g, b));
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

  float h = 1.0f * ConvertNorm<In>(0.5) / ConvertNorm<In>(1.0);

  // black
  check(0.0f, 0.0f, 0.0f);
  // gray
  check(h, h, h);
  // white
  check(1.0f, 1.0f, 1.0f);

  // red
  check(1.0f, 0.0f, 0.0f);
  // green
  check(0.0f, 1.0f, 0.0f);
  // blue
  check(0.0f, 0.0f, 1.0f);

  // yellow
  check(1.0f, 1.0f, 0.0f);
  // cyan
  check(0.0f, 1.0f, 1.0f);
  // magenta
  check(1.0f, 0.0f, 1.0f);

  // some random colors
  std::mt19937_64 rng(12345);
  std::conditional_t<std::is_integral<In>::value,
    std::uniform_int_distribution<In>,
    std::uniform_real_distribution<float>
  > dist(0, std::is_integral<In>::value ? max_value<In>() : 1);
  for (int i = 0; i < 1000; i++) {
    constexpr float s = std::is_integral<In>::value ? 1.0f / max_value<In>() : 1.0f;
    check(dist(rng) * s, dist(rng) * s, dist(rng) * s);
  }
}

}  // namespace test
}  // namespace color
}  // namespace kernels
}  // namespace dali
