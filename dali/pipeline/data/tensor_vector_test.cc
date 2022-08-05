// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdexcept>
#include <string>

#include "dali/core/access_order.h"
#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_vector.h"
#include "dali/pipeline/data/types.h"
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

constexpr cudaStream_t cuda_stream = 0;

TYPED_TEST_SUITE(TensorVectorSuite, Backends);

// Check if interleaving any of
// * set_pinned
// * set_type
// * Resize(shape)
// * Resize(shape, type)
// * reserve
// behaves as it is supposed to - that is set_type always first, set_type before Reshape,
// reserve can be without type.

// TODO(klecki): reverse pinned and capacity tests

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
  EXPECT_THROW(tv.Resize({{2, 4}, {4, 2}}), std::runtime_error);
  tv.Resize({{2, 4}, {4, 2}}, DALI_INT32);
  ASSERT_EQ(tv.num_samples(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv[0].type(), DALI_INT32);
  EXPECT_EQ(tv[1].type(), DALI_INT32);
  ASSERT_THROW(tv.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, PinnedBeforeResizeContiguous) {
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.reserve(100);
  EXPECT_THROW(tv.Resize({{2, 4}, {4, 2}}), std::runtime_error);
  tv.template set_type<int32_t>();
  tv.Resize({{2, 4}, {4, 2}});
  ASSERT_EQ(tv.num_samples(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv.is_pinned(), false);
  for (int i = 0; i < tv.num_samples(); i++) {
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
}

TYPED_TEST(TensorVectorSuite, PinnedBeforeResizeNoncontiguous) {
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.reserve(50, 2);
  tv.template set_type<int32_t>();
  tv.Resize({{2, 4}, {4, 2}});
  ASSERT_EQ(tv.num_samples(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv.is_pinned(), false);
  for (int i = 0; i < tv.num_samples(); i++) {
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
}

namespace {

const uint8_t kMagicNumber = 42;

void FillWithMagicNumber(TensorVector<CPUBackend> &tv) {
  for (int i = 0; i < tv.shape().num_elements(); i++) {
    // We utilize the contiguity of the TensorVector
    tv.mutable_tensor<uint8_t>(0)[i] = kMagicNumber;
  }
}

void FillWithMagicNumber(TensorVector<GPUBackend> &tv) {
  // We utilize the contiguity of the TensorVector
  cudaMemset(tv.mutable_tensor<uint8_t>(0), kMagicNumber, tv.shape().num_elements());
}


void CompareWithMagicNumber(Tensor<CPUBackend> &t) {
  for (int i = 0; i < t.shape().num_elements(); i++) {
    EXPECT_EQ(t.data<uint8_t>()[i], kMagicNumber);
  }
}

void CompareWithMagicNumber(Tensor<GPUBackend> &t) {
  uint8_t *ptr;
  std::vector<uint8_t> buffer(t.shape().num_elements());
  cudaMemcpy(buffer.data(), t.data<uint8_t>(), t.shape().num_elements(), cudaMemcpyDeviceToHost);
  for (auto b : buffer) {
    EXPECT_EQ(b, kMagicNumber);
  }
}


template <typename Backend>
Tensor<Backend> ReturnTvAsTensor() {
  TensorVector<Backend> tv;
  tv.SetContiguous(true);
  tv.Resize(uniform_list_shape(4, {1, 2, 3}), DALI_UINT8);
  tv.SetLayout("HWC");
  FillWithMagicNumber(tv);
  return tv.AsTensor();
}

}  // namespace


TYPED_TEST(TensorVectorSuite, TensorVectorAsTensorAccess) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(true);
  auto shape = TensorListShape<>{{1, 2, 3}, {1, 2, 4}};
  tv.Resize(shape, DALI_INT32);
  EXPECT_TRUE(tv.IsContiguousInMemory());
  EXPECT_FALSE(tv.IsDenseTensor());

  {
    auto tensor_shape = TensorShape<>{2, 7};
    auto tensor = tv.AsReshapedTensor(tensor_shape);
    EXPECT_EQ(tensor.shape(), tensor_shape);
    EXPECT_EQ(tensor.type(), DALI_INT32);
    EXPECT_EQ(tensor.raw_data(), tv.raw_tensor(0));
  }
  tv.Resize(uniform_list_shape(3, {2, 3, 4}));
  tv.SetLayout("HWC");

  EXPECT_TRUE(tv.IsContiguousInMemory());
  EXPECT_TRUE(tv.IsDenseTensor());

  {
    auto expected_shape = TensorShape<>{3, 2, 3, 4};
    auto tensor = tv.AsTensor();
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.type(), DALI_INT32);
    EXPECT_EQ(tensor.GetLayout(), "NHWC");
    EXPECT_EQ(tensor.raw_data(), tv.raw_tensor(0));
  }

  {
    // Verify that we can convert and touch the data after TV is already released
    auto tensor = ReturnTvAsTensor<TypeParam>();
    auto expected_shape = TensorShape<>{4, 1, 2, 3};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.type(), DALI_UINT8);
    EXPECT_EQ(tensor.GetLayout(), "NHWC");
    CompareWithMagicNumber(tensor);
  }
}

TYPED_TEST(TensorVectorSuite, EmptyTensorVectorAsTensorAccess) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(true);
  tv.set_type(DALI_INT32);
  EXPECT_TRUE(tv.IsContiguousInMemory());
  EXPECT_TRUE(tv.IsDenseTensor());

  auto shape_0d = TensorShape<>{};
  auto shape_1d = TensorShape<>{0};
  auto shape_2d = TensorShape<>{0, 0};

  {
    EXPECT_THROW(tv.AsReshapedTensor(shape_0d), std::runtime_error);  // empty shape has volume = 1
    auto tensor_1d = tv.AsReshapedTensor(shape_1d);
    auto tensor_2d = tv.AsReshapedTensor(shape_2d);
    EXPECT_EQ(tensor_1d.shape(), shape_1d);
    EXPECT_EQ(tensor_1d.type(), DALI_INT32);
    EXPECT_EQ(tensor_1d.raw_data(), unsafe_raw_data(tv));
    EXPECT_EQ(tensor_1d.raw_data(), nullptr);
    EXPECT_EQ(tensor_2d.shape(), shape_2d);
    EXPECT_EQ(tensor_2d.type(), DALI_INT32);
    EXPECT_EQ(tensor_2d.raw_data(), unsafe_raw_data(tv));
    EXPECT_EQ(tensor_2d.raw_data(), nullptr);
  }

  tv.reserve(1000);

  {
    EXPECT_THROW(tv.AsReshapedTensor(shape_0d), std::runtime_error);  // empty shape has volume = 1
    auto tensor_1d = tv.AsReshapedTensor(shape_1d);
    auto tensor_2d = tv.AsReshapedTensor(shape_2d);
    EXPECT_EQ(tensor_1d.shape(), shape_1d);
    EXPECT_EQ(tensor_1d.type(), DALI_INT32);
    EXPECT_EQ(tensor_1d.raw_data(), unsafe_raw_data(tv));
    EXPECT_NE(tensor_1d.raw_data(), nullptr);
    EXPECT_EQ(tensor_2d.shape(), shape_2d);
    EXPECT_EQ(tensor_2d.type(), DALI_INT32);
    EXPECT_EQ(tensor_2d.raw_data(), unsafe_raw_data(tv));
    EXPECT_NE(tensor_2d.raw_data(), nullptr);
  }
}

TYPED_TEST(TensorVectorSuite, EmptyTensorVectorWithDimAsTensorAccess) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(true);
  tv.set_type(DALI_INT32);
  EXPECT_TRUE(tv.IsContiguousInMemory());
  EXPECT_TRUE(tv.IsDenseTensor());

  auto shape_1d = TensorShape<>{0};
  auto shape_2d = TensorShape<>{0, 0};

  {
    EXPECT_THROW(tv.AsTensor(), std::runtime_error);
    tv.set_sample_dim(0);
    EXPECT_THROW(tv.AsTensor(), std::runtime_error);
    tv.set_sample_dim(1);
    auto tensor_1d = tv.AsTensor();
    EXPECT_EQ(tensor_1d.shape(), shape_1d);
    EXPECT_EQ(tensor_1d.type(), DALI_INT32);
    EXPECT_EQ(tensor_1d.raw_data(), unsafe_raw_data(tv));
    EXPECT_EQ(tensor_1d.raw_data(), nullptr);

    tv.set_sample_dim(2);
    auto tensor_2d = tv.AsTensor();
    EXPECT_EQ(tensor_2d.shape(), shape_2d);
    EXPECT_EQ(tensor_2d.type(), DALI_INT32);
    EXPECT_EQ(tensor_2d.raw_data(), unsafe_raw_data(tv));
    EXPECT_EQ(tensor_2d.raw_data(), nullptr);
  }
}


TYPED_TEST(TensorVectorSuite, BatchResize) {
  TensorVector<TypeParam> tv(5);
  ASSERT_EQ(tv.num_samples(), 5);
  tv.reserve(100);
  tv.reserve(200);
  tv.template set_type<int32_t>();
  tv.Resize(uniform_list_shape(5, {10, 20}));
}

TYPED_TEST(TensorVectorSuite, VariableBatchResizeDown) {
  TensorVector<TypeParam> tv(32);
  ASSERT_EQ(tv.num_samples(), 32);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size, DALI_UINT8);
  ASSERT_EQ(tv.num_samples(), new_size.num_samples());
}

TYPED_TEST(TensorVectorSuite, VariableBatchResizeUp) {
  TensorVector<TypeParam> tv(2);
  ASSERT_EQ(tv.num_samples(), 2);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size, DALI_UINT8);
  ASSERT_EQ(tv.num_samples(), new_size.num_samples());
}

TYPED_TEST(TensorVectorSuite, EmptyShareContiguous) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(true);
  TensorListShape<> shape = {{100, 0, 0}, {42, 0, 0}};
  tv.Resize(shape, DALI_UINT8);
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(tv.raw_tensor(i), nullptr);
  }

  TensorVector<TypeParam> target;
  target.ShareData(tv);
  ASSERT_EQ(target.num_samples(), shape.num_samples());
  ASSERT_EQ(target.type(), DALI_UINT8);
  ASSERT_EQ(target.shape(), shape);
  ASSERT_TRUE(target.IsContiguous());
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(target.raw_tensor(i), nullptr);
    ASSERT_EQ(target.raw_tensor(i), tv.raw_tensor(i));
  }
}

TYPED_TEST(TensorVectorSuite, EmptyShareNonContiguous) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(false);
  TensorListShape<> shape = {{100, 0, 0}, {42, 0, 0}};
  tv.Resize(shape, DALI_UINT8);
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(tv.raw_tensor(i), nullptr);
  }

  TensorVector<TypeParam> target;
  target.ShareData(tv);
  ASSERT_EQ(target.num_samples(), shape.num_samples());
  ASSERT_EQ(target.type(), DALI_UINT8);
  ASSERT_EQ(target.shape(), shape);
  ASSERT_FALSE(target.IsContiguous());
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(target.raw_tensor(i), nullptr);
    ASSERT_EQ(target.raw_tensor(i), tv.raw_tensor(i));
  }
}

template <typename Backend, typename F>
void test_moving_props(const bool is_pinned, const TensorLayout layout,
                       const TensorListShape<> shape, const int sample_dim, const DALIDataType type,
                       F &&mover) {
  constexpr bool is_device = std::is_same_v<Backend, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorVector<Backend> tv;
  tv.set_pinned(is_pinned);
  tv.set_order(order);
  tv.set_sample_dim(sample_dim);
  tv.Resize(shape, type);
  tv.SetLayout(layout);

  const auto check = [&](auto &moved) {
    EXPECT_EQ(moved.order(), order);
    EXPECT_EQ(moved.GetLayout(), layout);
    EXPECT_EQ(moved.sample_dim(), sample_dim);
    EXPECT_EQ(moved.shape(), shape);
    EXPECT_EQ(moved.type(), type);
    EXPECT_EQ(moved.is_pinned(), is_pinned);
  };
  mover(check, tv);
}

TYPED_TEST(TensorVectorSuite, MoveConstructorMetaData) {
  test_moving_props<TypeParam>(true, "XYZ",
                               {{42, 1, 2}, {1, 2, 42}, {2, 42, 1}, {1, 42, 1}, {2, 42, 2}}, 3,
                               DALI_UINT16, [](auto &&check, auto &tv) {
                                 TensorVector<TypeParam> moved{std::move(tv)};
                                 check(moved);
                               });
}

TYPED_TEST(TensorVectorSuite, MoveAssignmentMetaData) {
  test_moving_props<TypeParam>(true, "XYZ",
                               {{42, 1, 2}, {1, 2, 42}, {2, 42, 1}, {1, 42, 1}, {2, 42, 2}}, 3,
                               DALI_UINT16, [](auto &&check, auto &tv) {
                                 TensorVector<TypeParam> moved(2);
                                 moved = std::move(tv);
                                 check(moved);
                               });
}

TYPED_TEST(TensorVectorSuite, DeviceIdPropagationMultiGPU) {
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  }
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr bool is_pinned = !is_device;
  AccessOrder order = AccessOrder::host();
  TensorShape<> shape{42};
  for (int device_id = 0; device_id < num_devices; device_id++) {
    TensorVector<TypeParam> batch;
    batch.SetSize(1);
    DeviceGuard dg(device_id);
    batch.set_device_id(device_id);
    batch.set_pinned(is_pinned);
    batch.set_type(DALI_UINT8);
    batch.set_sample_dim(shape.sample_dim());
    batch.set_order(order);
    void *data_ptr;
    std::shared_ptr<void> ptr;
    if (is_device) {
      CUDA_CALL(cudaMalloc(&data_ptr, shape.num_elements() * sizeof(uint8_t)));
      ptr = std::shared_ptr<void>(data_ptr, [](void *ptr) { cudaFree(ptr); });
    } else {
      CUDA_CALL(cudaMallocHost(&data_ptr, shape.num_elements() * sizeof(uint8_t)));
      ptr = std::shared_ptr<void>(data_ptr, [](void *ptr) { cudaFreeHost(ptr); });
    }
    batch.UnsafeSetSample(0, ptr, shape.num_elements() * sizeof(uint8_t), is_pinned, shape,
                          DALI_UINT8, device_id, order);
    ASSERT_EQ(batch.device_id(), device_id);
    ASSERT_EQ(batch.order().device_id(), AccessOrder::host().device_id());
    ASSERT_NE(batch.order().device_id(), batch.device_id());
  }
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
  if (rhs.num_samples() != lhs.num_samples()) {
    ::testing::AssertionFailure() << make_string("[Testing: ", testing_values,
                                                 "] Inconsistent number of tensors");
  }
  for (int tensor_idx = 0; tensor_idx < rhs.num_samples(); tensor_idx++) {
    if (rhs.tensor_shape(tensor_idx) != lhs.tensor_shape(tensor_idx)) {
      ::testing::AssertionFailure()
          << make_string("[Testing: ", testing_values, "] Inconsistent shapes");
    }
    auto vol = volume(rhs.tensor_shape(tensor_idx));
    auto lptr = reinterpret_cast<const char *>(lhs.raw_tensor(tensor_idx));
    auto rptr = reinterpret_cast<const char *>(rhs.raw_tensor(tensor_idx));
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
    test_tv_.Resize(shape_, DALI_FLOAT);
    for (int i = 0; i < shape_.num_samples(); i++) {
      UniformRandomFill(view<float>(test_tv_[i]), rng_, 0.f, 1.f);
    }
  }

  void GenerateTestTl() {
    test_tl_.Resize(shape_, DALI_FLOAT);
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
  tv.ShareData(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvShareWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.ShareData(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlShareWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.ShareData(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlShareWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.ShareData(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvCopyWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.Copy(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvCopyWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.Copy(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlCopyWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.Copy(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlCopyWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.Copy(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

}  // namespace test
}  // namespace dali
