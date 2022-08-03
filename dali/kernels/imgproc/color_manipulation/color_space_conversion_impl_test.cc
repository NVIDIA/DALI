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

    return vec3(
      y  * 219/255 + ybias,
      cb * 224/255 + cbias,
      cr * 224/255 + cbias
    );
  }

};

static_assert(itu_ref::rgb_to_ycbcr<8>({0, 0, 0}) == vec3(16, 128, 128));
static_assert(itu_ref::rgb_to_ycbcr<8>({255, 255, 255}) == vec3(235, 128, 128));


TEST(ColorSpaceConversion_ITU_R_BT601_Test, rgb_to_ycbcr_u8) {
  auto rgb_to_y  = [](auto rgb) { return itu_r_bt_601::rgb_to_y<uint8_t>(rgb); };
  auto rgb_to_cb = [](auto rgb) { return itu_r_bt_601::rgb_to_cb<uint8_t>(rgb); };
  auto rgb_to_cr = [](auto rgb) { return itu_r_bt_601::rgb_to_cr<uint8_t>(rgb); };
  EXPECT_EQ(rgb_to_y(u8vec3(0, 0, 0)), 16);
  EXPECT_EQ(rgb_to_y(u8vec3(255, 255, 255)), 235);
  EXPECT_EQ(rgb_to_y(u8vec3(255, 0, 0)), 82);
  EXPECT_EQ(rgb_to_y(u8vec3(0, 255, 0)), 145);
  EXPECT_EQ(rgb_to_y(u8vec3(0, 0, 255)), 41);

  EXPECT_EQ(rgb_to_cb(u8vec3(0, 0, 0)), 128);
  EXPECT_EQ(rgb_to_cr(u8vec3(0, 0, 0)), 128);
  EXPECT_EQ(rgb_to_cb(u8vec3(255, 255, 255)), 128);
  EXPECT_EQ(rgb_to_cr(u8vec3(255, 255, 255)), 128);

  EXPECT_EQ(rgb_to_cb(u8vec3(255,   0,   0)), 90);
  EXPECT_EQ(rgb_to_cr(u8vec3(255,   0,   0)), 240);
  EXPECT_EQ(rgb_to_cb(u8vec3(255, 255,   0)), 16);
  EXPECT_EQ(rgb_to_cr(u8vec3(255, 255,   0)), 146);
  EXPECT_EQ(rgb_to_cb(u8vec3(  0, 255,   0)), 54);
  EXPECT_EQ(rgb_to_cr(u8vec3(  0, 255,   0)), 34);
  EXPECT_EQ(rgb_to_cb(u8vec3(  0, 255, 255)), 166);
  EXPECT_EQ(rgb_to_cr(u8vec3(  0, 255, 255)), 16);
  EXPECT_EQ(rgb_to_cb(u8vec3(  0,   0, 255)), 240);
  EXPECT_EQ(rgb_to_cr(u8vec3(  0,   0, 255)), 110);
  EXPECT_EQ(rgb_to_cb(u8vec3(255,   0, 255)), 202);
  EXPECT_EQ(rgb_to_cr(u8vec3(255,   0, 255)), 222);
}

template <typename TypeParam>
struct ColorSpaceConversionTypedTest;

template <typename Out, typename In>
struct ColorSpaceConversionTypedTest<std::pair<Out, In>> : ::testing::Test {};

using Backends = ::testing::Types<
  std::pair<uint8_t, uint8_t>,
  std::pair<uint8_t, uint16_t>,
  std::pair<uint8_t, float>,
  std::pair<uint16_t, uint8_t>,
  std::pair<uint16_t, uint16_t>,
  std::pair<uint16_t, float>,
  std::pair<float, uint8_t>,
  std::pair<float, uint16_t>,
  std::pair<float, float>>;

TYPED_TEST_SUITE(ColorSpaceConversionTypedTest, Backends);

TYPED_TEST(ColorSpaceConversionTypedTest, rgb_to_ycbcr) {
  using Out = typename TypeParam::first_type;
  using In = typename TypeParam::second_type;

  using ref = itu_ref;
  using method = itu_r_bt_601;

  constexpr int bits = std::is_integral<Out>::value
    ? sizeof(Out) * 8 - std::is_signed<Out>::value : 0;

  double eps = std::is_integral<Out>::value ? 0.51 : 1e-3;

  auto make_rgb = [](float r, float g, float b) {
    return vec<3, In>(vec<3>(r, g, b) * detail::scale_factor<float, In>());
  };
  auto make_rgb_ref = [](float r, float g, float b) {
    return vec<3>(r, g, b) * detail::scale_factor<float, Out>();
  };


  auto check = [&](float r, float g, float b) {
    auto rgb = make_rgb(r, g ,b);
    auto ycbcr = ref::rgb_to_ycbcr<bits>(make_rgb_ref(r, g, b));
    EXPECT_NEAR(method::rgb_to_y<Out>(rgb) , ycbcr[0], eps) << "RGB = " << vec3(rgb);
    EXPECT_NEAR(method::rgb_to_cb<Out>(rgb), ycbcr[1], eps) << "RGB = " << vec3(rgb);
    EXPECT_NEAR(method::rgb_to_cr<Out>(rgb), ycbcr[2], eps) << "RGB = " << vec3(rgb);
  };

  check(0, 0, 0);
  check(1, 1, 1);
}

}  // namespace test
}  // namespace color
}  // namespace kernels
}  // namespace dali
