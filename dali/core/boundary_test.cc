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
#include <random>
#include "dali/core/boundary.h"

namespace dali {

TEST(Boundary, reflect_101) {
  using boundary::idx_reflect_101;
  EXPECT_EQ(idx_reflect_101(0, 0, 1), 0);
  EXPECT_EQ(idx_reflect_101(-1, 0, 1), 0);
  EXPECT_EQ(idx_reflect_101(1, 0, 1), 0);

  EXPECT_EQ(idx_reflect_101(1, 1, 2), 1);
  EXPECT_EQ(idx_reflect_101(-1, 1, 2), 1);
  EXPECT_EQ(idx_reflect_101(2, 1, 2), 1);

  EXPECT_EQ(idx_reflect_101(1, 1, 2), 1);
  EXPECT_EQ(idx_reflect_101(1, 2, 6), 3);
  EXPECT_EQ(idx_reflect_101(0, 2, 6), 4);
  EXPECT_EQ(idx_reflect_101(4, 4, 10), 4);
  EXPECT_EQ(idx_reflect_101(7, 4, 10), 7);
  EXPECT_EQ(idx_reflect_101(9, 4, 10), 9);
  EXPECT_EQ(idx_reflect_101(10, 4, 10), 8);
  EXPECT_EQ(idx_reflect_101(11, 4, 10), 7);
}

TEST(Boundary, reflect_1001) {
  using boundary::idx_reflect_1001;
  EXPECT_EQ(idx_reflect_1001(0, 0, 1), 0);
  EXPECT_EQ(idx_reflect_1001(-1, 0, 1), 0);
  EXPECT_EQ(idx_reflect_1001(1, 0, 1), 0);

  EXPECT_EQ(idx_reflect_1001(1, 1, 2), 1);
  EXPECT_EQ(idx_reflect_1001(-1, 1, 2), 1);
  EXPECT_EQ(idx_reflect_1001(2, 1, 2), 1);

  EXPECT_EQ(idx_reflect_1001(1, 1, 2), 1);
  EXPECT_EQ(idx_reflect_1001(1, 2, 6), 2);
  EXPECT_EQ(idx_reflect_1001(0, 2, 6), 3);
  EXPECT_EQ(idx_reflect_1001(4, 4, 10), 4);
  EXPECT_EQ(idx_reflect_1001(7, 4, 10), 7);
  EXPECT_EQ(idx_reflect_1001(9, 4, 10), 9);
  EXPECT_EQ(idx_reflect_1001(10, 4, 10), 9);
  EXPECT_EQ(idx_reflect_1001(11, 4, 10), 8);
}

TEST(Boundary, clamp) {
  using boundary::idx_clamp;
  EXPECT_EQ(idx_clamp(0, 1, 4), 1);
  EXPECT_EQ(idx_clamp(1, 1, 4), 1);
  EXPECT_EQ(idx_clamp(2, 1, 4), 2);
  EXPECT_EQ(idx_clamp(3, 1, 4), 3);
  EXPECT_EQ(idx_clamp(4, 1, 4), 3);
}

TEST(Boundary, wrap) {
  using boundary::idx_wrap;
  EXPECT_EQ(idx_wrap(-4, 4), 0);
  EXPECT_EQ(idx_wrap(-3, 4), 1);
  EXPECT_EQ(idx_wrap(-2, 4), 2);
  EXPECT_EQ(idx_wrap(-1, 4), 3);
  EXPECT_EQ(idx_wrap(0, 4), 0);
  EXPECT_EQ(idx_wrap(1, 4), 1);
  EXPECT_EQ(idx_wrap(2, 4), 2);
  EXPECT_EQ(idx_wrap(3, 4), 3);
  EXPECT_EQ(idx_wrap(4, 4), 0);
}

#define CHECK3(func, x, lo, hi)                                                \
  {                                                                            \
    auto _vresult = func(x, lo, hi);                                           \
    for (int _d = 0; _d < _vresult.size(); _d++) {                             \
      auto _sresult = func(x[_d], lo[_d], hi[_d]);                             \
      ASSERT_EQ(_vresult[_d], _sresult)                                        \
          << #func " with vector vs scalar arguments yields different result"; \
    }                                                                          \
  }

#define CHECK2(func, x, hi)                                                    \
  {                                                                            \
    auto _vresult = func(x, hi);                                               \
    for (int _d = 0; _d < _vresult.size(); _d++) {                             \
      auto _sresult = func(x[_d], hi[_d]);                                     \
      ASSERT_EQ(_vresult[_d], _sresult)                                        \
          << #func " with vector vs scalar arguments yields different result"; \
    }                                                                          \
  }

TEST(Boundary, vec_vs_scalar) {
  std::mt19937_64 rng(1234);

  int N = 10000;
  std::uniform_int_distribution<int> dist(-100, 100);
  std::bernoulli_distribution reverse(100.0/N);

  for (int i = 0; i < N; i++) {
    ivec3 v, lo, hi;
    for (int j = 0; j < 3; j++) {
      v[j] = dist(rng);
      lo[j] = dist(rng);
      hi[j] = dist(rng);
      bool should_swap = lo[j] > hi[j];
      if (reverse(rng))
        should_swap = !should_swap;
      if (should_swap)
        std::swap(lo[j], hi[j]);
    }
    CHECK3(boundary::idx_reflect_101, v, lo, hi);
    CHECK3(boundary::idx_reflect_1001, v, lo, hi);
    CHECK3(boundary::idx_clamp, v, lo, hi);

    v += lo;
    hi -= lo;

    CHECK2(boundary::idx_reflect_101, v, hi);
    CHECK2(boundary::idx_reflect_1001, v, hi);
    CHECK2(boundary::idx_clamp, v, hi);

    // wrap crashes for zero size (division by 0)
    for (int j = 0; j < 3; j++)
      if (hi[j] <= 0) hi[j] = 1;
    CHECK2(boundary::idx_wrap, v, hi);
  }
}

}  // namespace dali
