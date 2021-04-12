// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/math_util.h"

namespace dali {

TEST(MathUtil, rsqrt) {
  constexpr float rel_err = 5e-6f;
  EXPECT_NEAR(rsqrt(0.25f), 2.0f, 2.0f * rel_err);
  EXPECT_NEAR(rsqrt(0.16f), 2.5f, 2.5f * rel_err);
  EXPECT_NEAR(rsqrt(16.0f), 0.25f, 0.25f * rel_err);
}

TEST(MathUtil, fast_rsqrt) {
  constexpr float rel_err = 2e-3f;
  EXPECT_NEAR(fast_rsqrt(0.25f), 2.0f, 2.0f * rel_err);
  EXPECT_NEAR(fast_rsqrt(0.16f), 2.5f, 2.5f * rel_err);
  EXPECT_NEAR(fast_rsqrt(16.0f), 0.25f, 0.25f * rel_err);
}

TEST(MathUtil, rsqrtd) {
  constexpr double rel_err = 1e-11;
  EXPECT_NEAR(rsqrt(0.25), 2.0, 2.0 * rel_err);
  EXPECT_NEAR(rsqrt(0.16), 2.5, 2.0 * rel_err);
  EXPECT_NEAR(rsqrt(16.0), 0.25, 0.25 * rel_err);
}

TEST(MathUtil, fast_rsqrtd) {
  constexpr double rel_err = 1e-10;
  EXPECT_NEAR(fast_rsqrt(0.25), 2.0, 2.0 * rel_err);
  EXPECT_NEAR(fast_rsqrt(0.16), 2.5, 2.0 * rel_err);
  EXPECT_NEAR(fast_rsqrt(16.0), 0.25, 0.25 * rel_err);
}

TEST(MathUtil, IpowAnyZero) {
  EXPECT_EQ(ipow(100, 0), 1);
  EXPECT_EQ(ipow(1, 0), 1);
  EXPECT_EQ(ipow(0, 0), 1);
  EXPECT_EQ(ipow(-1, 0), 1);
  EXPECT_EQ(ipow(-100, 0), 1);
}

TEST(MathUtil, IpowAnyNeg) {
  EXPECT_EQ(ipow(100, -1), 0);
  EXPECT_EQ(ipow(1, -1), 0);
  EXPECT_EQ(ipow(0, -1), 0);
  EXPECT_EQ(ipow(-1, -1), 0);
  EXPECT_EQ(ipow(-100, -1), 0);
}

TEST(MathUtil, IpowNeg) {
  EXPECT_EQ(ipow(-1, -1), 0);
  EXPECT_EQ(ipow(-1, 0), 1);
  EXPECT_EQ(ipow(-1, 1), -1);
  EXPECT_EQ(ipow(-1, 2), 1);
  EXPECT_EQ(ipow(-1, 3), -1);
  EXPECT_EQ(ipow(-1, 4), 1);
}

TEST(MathUtil, Ipow) {
  EXPECT_EQ(ipow(2, 1), 2);
  EXPECT_EQ(ipow(2, 2), 4);
  EXPECT_EQ(ipow(2, 3), 8);
  EXPECT_EQ(ipow(2, 4), 16);
  EXPECT_EQ(ipow(2, 5), 32);
  EXPECT_EQ(ipow(2, 6), 64);
  EXPECT_EQ(ipow(2, 7), 128);
  EXPECT_EQ(ipow(2, 8), 256);
  EXPECT_EQ(ipow(2, 9), 512);
  EXPECT_EQ(ipow(2, 10), 1024);
  EXPECT_EQ(ipow(-2, 1), -2);
  EXPECT_EQ(ipow(-2, 2), 4);
  EXPECT_EQ(ipow(-2, 3), -8);
  EXPECT_EQ(ipow(-2, 4), 16);
  EXPECT_EQ(ipow(-2, 5), -32);
  EXPECT_EQ(ipow(-2, 6), 64);
  EXPECT_EQ(ipow(-2, 7), -128);
  EXPECT_EQ(ipow(-2, 8), 256);
  EXPECT_EQ(ipow(-2, 9), -512);
  EXPECT_EQ(ipow(-2, 10), 1024);
}

TEST(MathUtil, IpowUnsigned) {
  EXPECT_EQ(ipow(2, 0u), 1u);
  EXPECT_EQ(ipow(2u, 0), 1u);
  EXPECT_EQ(ipow(2u, 2), 4u);
  EXPECT_EQ(ipow(2, 2u), 4u);
  EXPECT_EQ(ipow(2u, 2u), 4u);
}

}  // namespace dali
