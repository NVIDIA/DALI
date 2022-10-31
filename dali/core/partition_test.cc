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

#include "dali/core/partition.h"
#include <gtest/gtest.h>
#include <vector>

namespace dali {

TEST(MultiPartitionTest, CArray) {
  const int N = 9;
  int input[N] = { 5, 6, 3, 5, 10, 0, 3, 4, 5 };
  int ref[N] =   { 5, 5, 5, 3, 0, 3, 4, 6, 10 };
  // partition points      ^           ^  ^
  auto [less_than_5, less_than_7, rest] = multi_partition(
      input,
      [](int x) { return x == 5; },
      [](int x) { return x < 5; },
      [](int x) { return x < 7; });  // note that reordering these would yield a different result!
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(input[i], ref[i]);
  }
  EXPECT_EQ(less_than_5, input + 3);
  EXPECT_EQ(less_than_7, input + 7);
  EXPECT_EQ(rest, input + 8);
}

TEST(MultiPartitionTest, Iter) {
  const int N = 9;
  std::vector<int> input = { 6, 7, 3, 10, 0, 8, 3, 4, 1 };
  int ref[]              = { 3, 0, 3, 4, 1, 6, 7, 10, 8 };
  // partition points       ^              ^           ^
  auto [less_than_five, more_than_five, ten, rest] = multi_partition(
      input.begin(), input.end(),
      [](int x) { return x == 5; },  // no elements satisfy this
      [](int x) { return x < 5; },
      [](int x) { return x > 5; },
      [](int x) { return x == 10; });  // overlaps with previous predicate, should be empty
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(input[i], ref[i]);
  }
  EXPECT_EQ(less_than_five, input.begin());
  EXPECT_EQ(more_than_five, input.begin() + 5);
  EXPECT_EQ(ten, input.end());
  EXPECT_EQ(rest, input.end());
}


TEST(MultiPartitionTest, Vector) {
  const int N = 9;
  std::vector<int> input = { 6, 7, 3, 10, 0, 8, 3, 4, 1 };
  int ref[]              = { 3, 0, 3, 4, 1, 10, 6, 7, 8 };
  // partition points       ^              ^   ^     ^
  auto [less_than_5, ten, less_than_8, rest] = multi_partition(
      input,
      [](int x) { return x == 5; },  // no elements satisfy this
      [](int x) { return x < 5; },
      [](int x) { return x == 10; },
      [](int x) { return x < 8; });
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(input[i], ref[i]);
  }
  EXPECT_EQ(less_than_5, input.begin());
  EXPECT_EQ(ten,         input.begin() + 5);
  EXPECT_EQ(less_than_8, input.begin() + 6);
  EXPECT_EQ(rest,        input.begin() + 8);
}

}  // namespace dali
