// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>

#include "dali/core/span.h"
#include "dali/kernels/common/transpose.h"

namespace dali {

using compact_result = SmallVector<std::pair<int, int>, 6>;

TEST(TransposeTest, PermutationBlocksEmpty) {
  auto perm = std::vector<int>{};
  auto result = kernels::transpose_impl::PermutationBlocks(make_cspan(perm));
  auto expected = compact_result{};
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, PermutationBlocksOneElem) {
  auto perm = std::vector<int>{0};
  auto result = kernels::transpose_impl::PermutationBlocks(make_cspan(perm));
  auto expected = compact_result{{0, 0}};
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, PermutationBlocksTwoElems) {
  auto perm_0 = std::vector<int>{0, 1};
  auto result_0 = kernels::transpose_impl::PermutationBlocks(make_cspan(perm_0));
  auto expected_0 = compact_result{{0, 1}};
  EXPECT_EQ(expected_0, result_0);

  auto perm_1 = std::vector<int>{1, 0};
  auto result_1 = kernels::transpose_impl::PermutationBlocks(make_cspan(perm_1));
  auto expected_1 = compact_result{{0, 0}, {1, 1}};
  EXPECT_EQ(expected_1, result_1);
}

TEST(TransposeTest, PermutationBlocks) {
  auto perm_0 = std::vector<int>{0, 1, 2};
  auto result_0 = kernels::transpose_impl::PermutationBlocks(make_cspan(perm_0));
  auto expected_0 = compact_result{{0, 2}};
  EXPECT_EQ(expected_0, result_0);

  auto perm_1 = std::vector<int>{2, 0, 1};
  auto result_1 = kernels::transpose_impl::PermutationBlocks(make_cspan(perm_1));
  auto expected_1 = compact_result{{0, 0}, {1, 2}};
  EXPECT_EQ(expected_1, result_1);

  auto perm_2 = std::vector<int>{3, 4, 5, 2, 0, 1};
  auto result_2 = kernels::transpose_impl::PermutationBlocks(make_cspan(perm_2));
  auto expected_2 = compact_result{{0, 2}, {3, 3}, {4, 5}};
  EXPECT_EQ(expected_2, result_2);

  auto perm_3 = std::vector<int>{3, 0, 1, 2};
  auto result_3 = kernels::transpose_impl::PermutationBlocks(make_cspan(perm_3));
  auto expected_3 = compact_result{{0, 0}, {1, 3}};
  EXPECT_EQ(expected_3, result_3);
}

TEST(TransposeTest, CollapsePermutationEmpty) {
  auto perm = std::vector<int>{};
  auto groups = compact_result{};
  auto result = kernels::transpose_impl::CollapsePermutation(make_cspan(perm), make_cspan(groups));
  auto expected = std::vector<int>{};
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, CollapsePermutationOneElem) {
  auto perm = std::vector<int>{0};
  auto groups = compact_result{{0, 0}};
  auto result = kernels::transpose_impl::CollapsePermutation(make_cspan(perm), make_cspan(groups));
  auto expected = std::vector<int>{0};
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, CollapsePermutationTwoElems) {
  auto perm_0 = std::vector<int>{0, 1};
  auto groups_0 = compact_result{{0, 1}};
  auto result_0 =
      kernels::transpose_impl::CollapsePermutation(make_cspan(perm_0), make_cspan(groups_0));
  auto expected_0 = std::vector<int>{0};
  EXPECT_EQ(expected_0, result_0);

  auto perm_1 = std::vector<int>{1, 0};
  auto groups_1 = compact_result{{0, 0}, {1, 1}};
  auto result_1 =
      kernels::transpose_impl::CollapsePermutation(make_cspan(perm_1), make_cspan(groups_1));
  auto expected_1 = std::vector<int>{1, 0};
  EXPECT_EQ(expected_1, result_1);
}

TEST(TransposeTest, CollapsePermutation) {
  auto perm_0 = std::vector<int>{0, 1, 2};
  auto groups_0 = compact_result{{0, 2}};
  auto result_0 =
      kernels::transpose_impl::CollapsePermutation(make_cspan(perm_0), make_cspan(groups_0));
  auto expected_0 = std::vector<int>{0};
  EXPECT_EQ(expected_0, result_0);

  auto perm_1 = std::vector<int>{2, 0, 1};
  auto groups_1 = compact_result{{0, 0}, {1, 2}};
  auto result_1 =
      kernels::transpose_impl::CollapsePermutation(make_cspan(perm_1), make_cspan(groups_1));
  auto expected_1 = std::vector<int>{1, 0};
  EXPECT_EQ(expected_1, result_1);

  auto perm_2 = std::vector<int>{3, 4, 5, 2, 0, 1};
  auto groups_2 = compact_result{{0, 2}, {3, 3}, {4, 5}};
  auto result_2 =
      kernels::transpose_impl::CollapsePermutation(make_cspan(perm_2), make_cspan(groups_2));
  auto expected_2 = std::vector<int>{2, 1, 0};
  EXPECT_EQ(expected_2, result_2);

  auto perm_3 = std::vector<int>{3, 0, 1, 2};
  auto groups_3 = compact_result{{0, 0}, {1, 3}};
  auto result_3 =
      kernels::transpose_impl::CollapsePermutation(make_cspan(perm_3), make_cspan(groups_3));
  auto expected_3 = std::vector<int>{1, 0};
  EXPECT_EQ(expected_3, result_3);
}

}  // namespace dali
