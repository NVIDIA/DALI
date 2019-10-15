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

#include "dali/pipeline/data/tensor_vector.h"
#include "dali/core/tensor_shape.h"

namespace dali {

template <typename T>
class TensorVectorSuite : public ::testing::Test {
 protected:
  void validate(const TensorVector<T> &tv) {
    ASSERT_EQ(tv.size(), 2);
    EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
    EXPECT_EQ(tv[0]->shape(), TensorShape<>(2, 4));
    EXPECT_EQ(tv[1]->shape(), TensorShape<>(4, 2));
    EXPECT_EQ(tv[0]->nbytes(), 4 * 2 * sizeof(int32_t));
    EXPECT_EQ(tv[1]->nbytes(), 4 * 2 * sizeof(int32_t));
    EXPECT_EQ(tv.is_pinned(), false);
    EXPECT_EQ(tv[0]->is_pinned(), false);
    EXPECT_EQ(tv[1]->is_pinned(), false);
  }
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;

TYPED_TEST_SUITE(TensorVectorSuite, Backends);

// Check if inerleaving any of
// * Resize
// * set_type
// * resreve
// * set_pinned
// behaves as it is supposed to.

TYPED_TEST(TensorVectorSuite, PinnedAfterReserveThrows) {
  TensorVector<TypeParam> tv_0, tv_1;
  tv_0.reserve(100);
  EXPECT_THROW(tv_0.set_pinned(false), std::runtime_error);
  tv_1.reserve(100, 2);
  EXPECT_THROW(tv_1.set_pinned(false), std::runtime_error);
  TensorVector<TypeParam> tv_2(2), tv_3(2);
  tv_2.reserve(100);
  EXPECT_THROW(tv_2.set_pinned(false), std::runtime_error);
  tv_3.reserve(100, 2);
  EXPECT_THROW(tv_3.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, PinnedAfterResizeThrows) {
  TensorVector<TypeParam> tv;
  tv.reserve(100);
  tv.Resize({{2, 4}, {4, 2}});
  tv.set_type(TypeInfo::Create<int32_t>());
  ASSERT_EQ(tv.size(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv[0].nbytes(), 4 * 2 * sizeof(int32_t));
  EXPECT_EQ(tv[1].nbytes(), 4 * 2 * sizeof(int32_t));
  EXPECT_EQ(tv[0].capacity(), 4 * 2 * sizeof(int32_t));
  EXPECT_EQ(tv[1].capacity(), 4 * 2 * sizeof(int32_t));
  ASSERT_THROW(tv.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, PinnedBeforeResizeContiguous) {
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.reserve(100);
  tv.Resize({{2, 4}, {4, 2}});
  tv.set_type(TypeInfo::Create<int32_t>());
  ASSERT_EQ(tv.size(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  for (auto &t : tv) {
    EXPECT_EQ(t->nbytes(), 4 * 2 * sizeof(int32_t));
    EXPECT_EQ(t->capacity(), 4 * 2 * sizeof(int32_t));
    EXPECT_EQ(t->is_pinned(), false);
  }
}

TYPED_TEST(TensorVectorSuite, PinnedBeforeResizeNoncontiguous) {
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.reserve(50, 2);
  tv.Resize({{2, 4}, {4, 2}});
  tv.set_type(TypeInfo::Create<int32_t>());
  ASSERT_EQ(tv.size(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  for (auto &t : tv) {
    EXPECT_EQ(t->nbytes(), 4 * 2 * sizeof(int32_t));
    EXPECT_EQ(t->capacity(), 50);
    EXPECT_EQ(t->is_pinned(), false);
  }
}

TYPED_TEST(TensorVectorSuite, BatchResize) {
  TensorVector<TypeParam> tv(5);
  ASSERT_EQ(tv.size(), 5);
  tv.reserve(100);
  tv.reserve(200);
  ASSERT_THROW(tv.Resize({{1}}), std::runtime_error);
  ASSERT_THROW(tv.reserve(20, 4), std::runtime_error);
  tv.Resize(uniform_list_shape(5, {10, 20}));
  tv.set_type(TypeInfo::Create<int32_t>());
  for (auto &t : tv) {
    EXPECT_TRUE(t->shares_data());
  }
}

}  // namespace dali
