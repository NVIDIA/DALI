// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/kernel.h"
#include "dali/kernels/type_tag.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {

namespace {
const int A = 1;
const float B = 1.5f;
const char C = 'c';
const double D = 1.0;
const int E = 3;
const float F = 5.5f;

}  // namespace

inline double fun(int &cnt, int a, float b, char c, double d, int e, float f) {
  cnt++;
  EXPECT_EQ(a, A);
  EXPECT_EQ(b, B);
  EXPECT_EQ(c, C);
  EXPECT_EQ(d, D);
  EXPECT_EQ(e, E);
  EXPECT_EQ(f, F);
  return a+b+c+d+e+f;
}

TEST(Tuple, Apply) {
  int cnt = 0;
  double sum = A+B+C+D+E+F;
  EXPECT_EQ(apply(fun, std::tie(cnt, A, B, C, D, E, F)), sum);
  EXPECT_EQ(cnt, 1);
}


TEST(Tuple, ApplyAll) {
  int cnt = 0;
  double sum = A+B+C+D+E+F;
  EXPECT_EQ(apply_all(fun, std::tie(cnt, A, B, C, D, E, F)), sum);
  EXPECT_EQ(apply_all(fun, cnt, A, B, C, D, E, F), sum);
  EXPECT_EQ(apply_all(fun, std::tie(cnt, A, B), C, D, std::make_tuple(E, F)), sum);
  EXPECT_EQ(apply_all(fun, cnt, A, std::make_tuple(B, C), D, std::make_tuple(E, F)), sum);
  EXPECT_EQ(apply_all(fun, cnt, A, std::make_tuple(B, C), D, std::make_tuple(E), F), sum);
  EXPECT_EQ(cnt, 5);
}

inline void fun_void(int &cnt, int a, float b, char c, double d, int e, float f) {
  fun(cnt, a, b, c, d, e, f);
}

TEST(Tuple, ApplyAllVoidResult) {
  int cnt = 0;
  double sum = A+B+C+D+E+F;
  apply_all(fun_void, std::tie(cnt, A, B, C, D, E, F));
  EXPECT_EQ(cnt, 1);
}

constexpr int TheAnswer() { return 42; }

TEST(Tuple, ApplyAllNoParams) {
  EXPECT_EQ(apply_all(TheAnswer), 42);
}

template <typename T, typename U>
constexpr auto Add(T a, U b)->decltype(a+b) { return a+b; }

#if !defined(__AARCH64_QNX__)

static_assert(apply_all(Add<int, char>, 1, 'a') == 'b', "Add(1, 'a') should yield 'b'");

#endif

}  // namespace kernels
}  // namespace dali
