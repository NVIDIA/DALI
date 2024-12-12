// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/compare.h"
#include <gtest/gtest.h>
#include <array>
#include <list>
#include <vector>
#include "dali/core/int_literals.h"

using namespace std::literals;  // NOLINT

namespace dali {

TEST(CompareTest, Primitive) {
  EXPECT_EQ(compare(1, 1_u8), 0);
  EXPECT_LT(compare(-2, 1_u8), 0);
  EXPECT_GT(compare(2, 1_i64), 0);
  EXPECT_GT(compare(0x110000007f000000_i64, 0x100000007fffffff_i64), 0);
  EXPECT_LT(compare(0x100000007f000000_i64, 0x100000007fffffff_i64), 0);
  EXPECT_GT(compare(float16(2), 1.0f), 0);
  EXPECT_EQ(compare(2.0f, float16(2)), 0);
}

TEST(CompareTest, Enum) {
  enum E {
    A = 1,
    B = 2,
    C = 3
  };
  EXPECT_EQ(compare(A, 1.0f), 0);
  EXPECT_LT(compare(A, B), 0);
  EXPECT_EQ(compare(C, A + B), 0);
}

TEST(CompareTest, String) {
  EXPECT_LT(compare("abc", "abcd"), 0);
  EXPECT_GT(compare(std::string("abcde"), std::string("abcd")), 0);
  EXPECT_LT(compare(std::string_view("abcd"), std::string_view("abdd")), 0);
}

TEST(CompareTest, CArray) {
  int shorter[] = { 1, 2, 3 };
  int longer[] = { 1, 2, 3, 4 };
  EXPECT_LT(compare(shorter, longer), 0);
  EXPECT_GT(compare(longer, shorter), 0);
  EXPECT_EQ(compare(shorter, shorter), 0);

  int ints[] = { 1, 2, 3 };
  int16_t shorts[] = { 1, 2, 3 };
  EXPECT_EQ(compare(ints, shorts), 0);

  int16_t different[] = { 1, 3, 3 };
  EXPECT_LT(compare(ints, different), 0);
}

TEST(CompareTest, MixedCollections) {
  std::vector<int> shorter = { 1, 2, 3 };
  int longer[] = { 1, 2, 3, 4 };
  EXPECT_LT(compare(shorter, longer), 0);
  EXPECT_GT(compare(longer, shorter), 0);
  EXPECT_EQ(compare(shorter, shorter), 0);

  int ints[] = { 1, 2, 3 };
  std::array<int16_t, 3> shorts = {{ 1, 2, 3 }};
  EXPECT_EQ(compare(ints, shorts), 0);

  std::list<int16_t> different = { 1, 3, 3 };
  EXPECT_LT(compare(ints, different), 0);
}

TEST(CompareTest, Tuple) {
  EXPECT_EQ(compare(std::make_tuple(1, 2, 3), std::make_tuple(1_u8, 2.0f, 3.0)), 0);
  EXPECT_GT(compare(std::make_tuple(1, 2, 3), std::make_tuple(1_u8, 2.0f)), 0);
  EXPECT_LT(compare(std::make_tuple(1, 2), std::make_tuple(1_u8, 2.0f, 3.0)), 0);

  EXPECT_LT(compare(std::make_tuple(1, "Former"s, 3), std::make_tuple(1_u8, "Jesse"s, 3.0)), 0);
  EXPECT_GT(compare(std::make_tuple(1.000001f, "a"sv, 3), std::make_tuple(1_u8, "b"sv, 3.0)), 0);
}

TEST(CompareTest, Pair) {
  EXPECT_EQ(compare(std::make_pair(1.0f, 42), std::make_pair(1, 42.0f)), 0);
  EXPECT_GT(compare(std::make_pair(1.1f, 42), std::make_pair(1, 42.0f)), 0);
  EXPECT_LT(compare(std::make_pair(1.0f, 42), std::make_pair(1, 42.1f)), 0);
}

}  // namespace dali
