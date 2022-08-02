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
namespace test {

TEST(ColorSpaceConversionTest, rgb_to_ycbcr_u8) {
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(u8vec3(0, 0, 0)), 16);
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(u8vec3(255, 255, 255)), 235);
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(u8vec3(255, 0, 0)), 82);
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(u8vec3(0, 255, 0)), 145);
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(u8vec3(0, 0, 255)), 41);

  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(0, 0, 0)), 128);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(0, 0, 0)), 128);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(255, 255, 255)), 128);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(255, 255, 255)), 128);

  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(255,   0,   0)), 90);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(255,   0,   0)), 240);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(255, 255,   0)), 16);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(255, 255,   0)), 146);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(  0, 255,   0)), 54);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(  0, 255,   0)), 34);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(  0, 255, 255)), 166);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(  0, 255, 255)), 16);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(  0,   0, 255)), 240);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(  0,   0, 255)), 110);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(u8vec3(255,   0, 255)), 202);
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(u8vec3(255,   0, 255)), 222);
}

template <typename T>
constexpr double to_norm_fp(T x) {
  return is_fp_or_half<T>::value ? 1.0 : static_cast<double>(std::numeric_limits<T>::max());
}

template <typename T>
constexpr T from_norm_fp(T x) {
  return is_fp_or_half<T>::value ? 1.0 : static_cast<double>(std::numeric_limits<T>::max() * x);
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


  auto rgb = [](double r, double g, double b) {
    return vec
  };

  EXPECT_EQ(itu_r_bt_601::rgb_to_y(rgb(0, 0, 0)), scale(16));
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(rgb(1, 1, 1)), scale(235));
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(rgb(1, 0, 0)), scale(82));
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(rgb(0, 1, 0)), scale(145));
  EXPECT_EQ(itu_r_bt_601::rgb_to_y(rgb(0, 0, 1)), scale(41));

  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(0, 0, 0)), scale(128));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(0, 0, 0)), scale(128));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(1, 1, 1)), scale(128));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(1, 1, 1)), scale(128));

  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(1, 0, 0)), scale(90));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(1, 0, 0)), scale(240));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(1, 1, 0)), scale(16));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(1, 1, 0)), scale(146));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(0, 1, 0)), scale(54));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(0, 1, 0)), scale(34));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(0, 1, 1)), scale(166));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(0, 1, 1)), scale(16));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(0, 0, 1)), scale(240));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(0, 0, 1)), scale(110));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cb(rgb(1, 0, 1)), scale(202));
  EXPECT_EQ(itu_r_bt_601::rgb_to_cr(rgb(1, 0, 1)), scale(222));
}

}  // namespace test
}  // namespace color
}  // namespace kernels
}  // namespace dali
