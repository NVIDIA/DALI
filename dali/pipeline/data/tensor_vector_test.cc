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
#include "dali/kernels/tensor_shape.h"

namespace dali {

using kernels::TensorShape;
using kernels::TensorListShape;

template <typename T>
class TensorVectorSuite : public ::testing::Test {
 protected:
  void validate(const TensorVector<T> &tv) {
    ASSERT_EQ(tv.size(), 2);
    EXPECT_EQ(tv.shape(), kernels::TensorListShape<>({{2, 4}, {4, 2}}));
    EXPECT_EQ(tv[0]->shape(), kernels::TensorShape<>(2, 4));
    EXPECT_EQ(tv[1]->shape(), kernels::TensorShape<>(4, 2));
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
  TensorVector<TypeParam> tv;
  tv.reserve(100);
  tv.reserve(100, 2);
  ASSERT_THROW(tv.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, PinnedAfterResizeThrows) {
  TensorVector<TypeParam> tv;
  tv.reserve(100);
  tv.Resize({{2, 4}, {4, 2}});
  tv.set_type(TypeInfo::Create<int32_t>());
  ASSERT_EQ(tv.size(), 2);
  EXPECT_EQ(tv.shape(), kernels::TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0]->shape(), kernels::TensorShape<>(2, 4));
  EXPECT_EQ(tv[1]->shape(), kernels::TensorShape<>(4, 2));
  EXPECT_EQ(tv[0]->nbytes(), 4 * 2 * sizeof(int32_t));
  EXPECT_EQ(tv[1]->nbytes(), 4 * 2 * sizeof(int32_t));
  EXPECT_EQ(tv[0]->capacity(), 50);
  EXPECT_EQ(tv[1]->capacity(), 50);
  ASSERT_THROW(tv.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, PinnedBeforeResize) {
  TensorVector<TypeParam> tv;
  tv.reserve(100);
  tv.set_pinned(false);
  tv.Resize({{2, 4}, {4, 2}});
  tv.set_type(TypeInfo::Create<int32_t>());
  ASSERT_EQ(tv.size(), 2);
  EXPECT_EQ(tv.shape(), kernels::TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0]->shape(), kernels::TensorShape<>(2, 4));
  EXPECT_EQ(tv[1]->shape(), kernels::TensorShape<>(4, 2));
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
  for (const auto &t : tv) {
    EXPECT_EQ(t->capacity(), 20);
  }
  tv.reserve(200);
  for (const auto &t : tv) {
    EXPECT_EQ(t->capacity(), 40);
  }
  ASSERT_THROW(tv.Resize({{1}}), std::runtime_error);
  ASSERT_THROW(tv.reserve(20, 4), std::runtime_error);
}

}  // namespace dali
