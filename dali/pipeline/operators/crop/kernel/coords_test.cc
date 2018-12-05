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
#include <random>

#include "dali/pipeline/operators/crop/kernel/coords.h"

namespace dali {
namespace detail {

class GetOffsetTest : public ::testing::Test {
 protected:
  GetOffsetTest()
      : shape_{20, 4, 1080, 1920, 3},
        plane_offsets_{shape_[1] * shape_[2] * shape_[3] * shape_[4],
                       shape_[2] * shape_[3] * shape_[4], shape_[3] * shape_[4], shape_[4], 1},
        gen_(rd_()) {}

  template <size_t N>
  std::array<int64_t, N> GetRandomArray(std::array<int64_t, N> limits) {
    std::array<int64_t, N> result;
    for (size_t i = 0; i < N; i++) {
      std::uniform_int_distribution<> dis(0, limits[i] - 1);
      result[i] = dis(gen_);
    }
    return result;
  }

  std::array<int64_t, 5> shape_;
  std::array<int64_t, 5> plane_offsets_;

 private:
  std::random_device rd_;
  std::mt19937 gen_;
};

TEST_F(GetOffsetTest, OrderSupport) {
  // This is really weird thing to test, but check if both calls compile
  dali_index_sequence<0, 1, 2> non_permuted;
  std::array<int64_t, 3> dummy_shape = {42, 42, 42};
  std::array<int64_t, 3> dummy_coords = {10, 10, 10};
  auto off_0 = getOffset<0, 1, 2>(dummy_shape, dummy_coords);
  auto off_1 = getOffset(dummy_shape, dummy_coords, non_permuted);
  // Can't put template function call in macro because it breaks on commas
  ASSERT_EQ(off_0, off_1);
}

TEST_F(GetOffsetTest, NonPermutatedPlanesOnly) {
  dali_index_sequence<0, 1, 2, 3, 4> non_perm;
  // Check individual planes
  for (int64_t i = 0; i < shape_[0]; i++) {
    ASSERT_EQ(getOffset(shape_, {i, 0, 0, 0, 0}, non_perm), i * plane_offsets_[0]);
  }
  for (int64_t i = 0; i < shape_[1]; i++) {
    ASSERT_EQ(getOffset(shape_, {0, i, 0, 0, 0}, non_perm), i * plane_offsets_[1]);
  }
  for (int64_t i = 0; i < shape_[2]; i++) {
    ASSERT_EQ(getOffset(shape_, {0, 0, i, 0, 0}, non_perm), i * plane_offsets_[2]);
  }
  for (int64_t i = 0; i < shape_[3]; i++) {
    ASSERT_EQ(getOffset(shape_, {0, 0, 0, i, 0}, non_perm), i * plane_offsets_[3]);
  }
  for (int64_t i = 0; i < shape_[4]; i++) {
    ASSERT_EQ(getOffset(shape_, {0, 0, 0, 0, i}, non_perm), i * plane_offsets_[4]);
  }
}

TEST_F(GetOffsetTest, NonPermutatedRandomCoords) {
  dali_index_sequence<0, 1, 2, 3, 4> non_perm;
  // Check individual planes
  for (size_t i = 0; i < 1000; i++) {
    auto coords = GetRandomArray(shape_);
    int64_t result = 0;
    for (size_t j = 0; j < 5; j++) {
      result += coords[j] * plane_offsets_[j];
    }
    ASSERT_EQ(getOffset(shape_, coords, non_perm), result);
  }
}

TEST_F(GetOffsetTest, ReversedRandomCoords) {
  dali_index_sequence<4, 3, 2, 1, 0> rev_perm;
  dali_index_sequence<0, 1, 2, 3, 4> non_perm;
  std::array<int64_t, 5> rev_shape = {shape_[4], shape_[3], shape_[2], shape_[1], shape_[0]};
  // Check individual planes
  for (size_t i = 0; i < 1000; i++) {
    auto coords = GetRandomArray(shape_);
    decltype(coords) rev_coords = {coords[4], coords[3], coords[2], coords[1], coords[0]};
    ASSERT_EQ(getOffset(rev_shape, coords, rev_perm), getOffset(rev_shape, rev_coords, non_perm));
  }
}

TEST_F(GetOffsetTest, PermutatedRandomCoords) {
  dali_index_sequence<4, 2, 0, 3, 1> perm;
  dali_index_sequence<0, 1, 2, 3, 4> non_perm;
  std::array<int64_t, 5> perm_shape = {shape_[4], shape_[2], shape_[0], shape_[1], shape_[3]};
  // Check individual planes
  for (size_t i = 0; i < 1000; i++) {
    auto coords = GetRandomArray(shape_);
    decltype(coords) perm_coords = {coords[4], coords[2], coords[0], coords[3], coords[1]};
    ASSERT_EQ(getOffset(perm_shape, coords, perm), getOffset(perm_shape, perm_coords, non_perm));
  }
}

}  // namespace detail
}  // namespace dali
