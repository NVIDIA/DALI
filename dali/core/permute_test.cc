// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/permute.h"  // NOLINT
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include "dali/core/span.h"

namespace dali {

TEST(PermuteTest, PopulateCArray) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 2, 3, 1, 0 };
  int out[]  = { 0, 0, 0, 0 };
  int ref[]  = { 6, 7, 5, 4 };
  permute(out, data, perm);
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(out[i], ref[i]);
}

TEST(PermuteTest, PopulateCArraySubset) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 3, 1 };
  int out[]  = { 0, 0 };
  int ref[]  = { 7, 5 };
  permute(out, data, perm);
  for (int i = 0; i < 2; i++)
    EXPECT_EQ(out[i], ref[i]);
}

TEST(PermuteTest, PopulateCArrayRepeat) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 3, 1, 2, 0, 2 };
  int out[]  = { 0, 0, 0, 0, 0 };
  int ref[]  = { 7, 5, 6, 4, 6 };
  permute(out, data, perm);
  for (int i = 0; i < 5; i++)
    EXPECT_EQ(out[i], ref[i]);
}

TEST(PermuteTest, PopulateSpan) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 2, 3, 1, 0 };
  int out[]  = { 0, 0, 0, 0 };
  int ref[]  = { 6, 7, 5, 4 };
  permute(make_span(out), make_span(data), perm);
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(out[i], ref[i]);
}

TEST(PermuteTest, PopulateVector) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 2, 3, 1, 0 };
  std::vector<int> out;
  std::vector<int> ref  = { 6, 7, 5, 4 };
  permute(out, data, perm);
  EXPECT_EQ(out, ref);
}

TEST(PermuteTest, PopulateVectorSubset) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 2, 1, 0 };
  std::vector<int> out;
  std::vector<int> ref  = { 6, 5, 4 };
  permute(out, data, perm);
  EXPECT_EQ(out, ref);
}

TEST(PermuteTest, ReturnSameType) {
  std::array<int, 4> data = { 4, 5, 6, 7 };
  int perm[] = { 3, 0, 1, 2 };
  std::array<int, 4> ref  = { 7, 4, 5, 6 };
  std::array<int, 4> out = permute(data, make_span(perm));
  EXPECT_EQ(out, ref);
}

TEST(PermuteTest, ReturnVectorExplicit) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 2, 3, 1, 0 };
  std::vector<int> ref  = { 6, 7, 5, 4 };
  std::vector<int> out = permute<std::vector<int>>(data, perm);
  EXPECT_EQ(out, ref);
}

TEST(PermuteTest, ReturnVectorSubset) {
  int data[] = { 4, 5, 6, 7 };
  int perm[] = { 2, 3, 0 };
  std::vector<int> ref  = { 6, 7, 4 };
  std::vector<int> out = permute<std::vector<int>>(data, perm);
  EXPECT_EQ(out, ref);
}


TEST(InversePermuteTest, PopulateCArray) {
  int perm[] = { 2, 3, 1, 0 };
  std::vector<int> ref  = { 3, 2, 0, 1 };
  std::vector<int> out = inverse_permutation<std::vector<int>>(perm);
  EXPECT_EQ(out, ref);
}

TEST(InversePermuteTest, ReturnSameType) {
  std::array<int, 4> perm = { 2, 3, 1, 0 };
  std::array<int, 4> ref  = { 3, 2, 0, 1 };
  std::array<int, 4> out = inverse_permutation(perm);
  EXPECT_EQ(out, ref);
}

TEST(InversePermuteTest, ReturnVectorExplicit) {
  int perm[] = { 2, 3, 1, 0 };
  std::vector<int> ref  = { 3, 2, 0, 1 };
  std::vector<int> out = inverse_permutation<std::vector<int>>(perm);
  EXPECT_EQ(out, ref);
}

TEST(PermuteInPlaceTest, AllPermutationsUpTo7) {
  const int maxN = 7;
  std::vector<int> perm, inout;
  for (int n = 1; n < maxN; n++) {
    perm.resize(n);
    inout.resize(n);
    std::iota(perm.begin(), perm.end(), 0);
    do {
      // fill the input with a sequence 100, 101, ... , 100+n
      std::iota(inout.begin(), inout.end(), 100);
      permute_in_place(inout, perm);
      // the element in a permuted sequence should be equal to the source index + 100
      for (int i = 0; i < n; i++) {
        ASSERT_EQ(inout[i], perm[i] + 100);
      }
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
}

}  // namespace dali
