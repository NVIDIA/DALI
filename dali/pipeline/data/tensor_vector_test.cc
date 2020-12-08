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
#include <string>
#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/tensor_vector.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace test {

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

typedef ::testing::Types<CPUBackend, GPUBackend> Backends;

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
  tv.Resize(uniform_list_shape(5, {10, 20}));
  tv.set_type(TypeInfo::Create<int32_t>());
  for (auto &t : tv) {
    EXPECT_TRUE(t->shares_data());
  }
}

TYPED_TEST(TensorVectorSuite, VariableBatchResizeDown) {
  TensorVector<TypeParam> tv(32);
  ASSERT_EQ(tv.size(), 32);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size);
  ASSERT_EQ(tv.size(), new_size.num_samples());
}

TYPED_TEST(TensorVectorSuite, VariableBatchResizeUp) {
  TensorVector<TypeParam> tv(2);
  ASSERT_EQ(tv.size(), 2);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size);
  ASSERT_EQ(tv.size(), new_size.num_samples());
}

namespace {

/**
 * GTest predicate formatter. Compares a batch of data contained in TensorList or TensorVector
 * @tparam T TensorList<CPUBackend> or TensorVector<CPUBackend>
 * @tparam U TensorList<CPUBackend> or TensorVector<CPUBackend>
 */
template <typename T, typename U>
::testing::AssertionResult Compare(const char *rhs_expr, const char *lhs_expr, const T &rhs,
                                   const U &lhs) {
  static_assert(std::is_same<T, TensorList<CPUBackend>>::value ||
                    std::is_same<T, TensorVector<CPUBackend>>::value,
                "T must be either TensorList<CPUBackend> or TensorVector<CPUBackend>");
  static_assert(std::is_same<U, TensorList<CPUBackend>>::value ||
                    std::is_same<U, TensorVector<CPUBackend>>::value,
                "U must be either TensorList<CPUBackend> or TensorVector<CPUBackend>");
  std::string testing_values = make_string(rhs_expr, ", ", lhs_expr);
  if (rhs.ntensor() != lhs.ntensor()) {
    ::testing::AssertionFailure() << make_string("[Testing: ", testing_values,
                                                 "] Inconsistent number of tensors");
  }
  for (size_t tensor_idx = 0; tensor_idx < rhs.ntensor(); tensor_idx++) {
    if (rhs.tensor_shape(tensor_idx) != lhs.tensor_shape(tensor_idx)) {
      ::testing::AssertionFailure()
          << make_string("[Testing: ", testing_values, "] Inconsistent shapes");
    }
    auto vol = volume(rhs.tensor_shape(tensor_idx));
    auto lptr = reinterpret_cast<const char*>(lhs.raw_tensor(tensor_idx));
    auto rptr = reinterpret_cast<const char*>(rhs.raw_tensor(tensor_idx));
    for (int i = 0; i < vol; i++) {
      if (rptr[i] != lptr[i]) {
        return ::testing::AssertionFailure()
               << make_string("[Testing: ", testing_values, "] Values at index [", i,
                              "] don't match: ", static_cast<int>(rptr[i]), " vs ",
                              static_cast<int>(lptr[i]), "");
      }
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace

class TensorVectorVariableBatchSizeTest : public ::testing::Test {
 protected:
  void SetUp() final {
    GenerateTestTv();
    GenerateTestTl();
  }

  void GenerateTestTv() {
    test_tv_.Resize(shape_);
    test_tv_.set_type(TypeInfo::Create<float>());
    for (int i = 0; i < shape_.num_samples(); i++) {
      UniformRandomFill(view_as_tensor<float>(test_tv_[i]), rng_, 0.f, 1.f);
    }
  }

  void GenerateTestTl() {
    test_tl_.Resize(shape_);
    test_tl_.set_type(TypeInfo::Create<float>());
    for (int i = 0; i < shape_.num_samples(); i++) {
      UniformRandomFill(view<float>(test_tl_), rng_, 0.f, 1.f);
    }
  }

  std::mt19937 rng_;
  TensorListShape<> shape_ = {{7, 2, 3}, {6, 1, 6}, {9, 3, 6}, {2, 9, 4},
                              {2, 3, 7}, {5, 3, 5}, {1, 1, 1}, {8, 3, 5}};
  TensorList<CPUBackend> test_tl_;
  TensorVector<CPUBackend> test_tv_;
};

TEST_F(TensorVectorVariableBatchSizeTest, SelfTest) {
  for (int i = 0; i < shape_.num_samples(); i++) {
    EXPECT_EQ(test_tv_.tensor_shape(i), shape_[i]);
    EXPECT_EQ(test_tl_.tensor_shape(i), shape_[i]);
  }
  EXPECT_PRED_FORMAT2(Compare, test_tv_, test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, test_tl_);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvShareWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.ShareData(&this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvShareWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.ShareData(&this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlShareWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.ShareData(&this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlShareWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.ShareData(&this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvCopyWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.Copy(this->test_tv_, nullptr);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvCopyWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.Copy(this->test_tv_, nullptr);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlCopyWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.Copy(this->test_tl_, nullptr);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlCopyWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.Copy(this->test_tl_, nullptr);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

}  // namespace test
}  // namespace dali
