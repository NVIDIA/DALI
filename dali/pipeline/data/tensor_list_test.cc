// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "dali/core/access_order.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/dali_test.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/timing.h"

namespace dali {
namespace test {

/***************************************************************************************************
 The section below is contains tests of original TensorList from tensor_list_test.cc
 with the only changes being:
 * Remove APIs that will be not ported: tensor_offset
 * Tests were generalized over contiguous/noncontiguous option (see the pairs in
   TensorListBackendContiguous)
 * SetContiguity(true) was added in few places, as the original TensorList was contiguous by default
 **************************************************************************************************/

template <typename TypeParam>
class TensorListTest : public DALITest {
 public:
  using Backend = std::tuple_element_t<0, TypeParam>;
  static constexpr auto kContiguity = std::tuple_element_t<1, TypeParam>::value;

  TensorListShape<> GetRandShape() {
    int num_tensor = this->RandInt(1, 64);
    int dims = this->RandInt(2, 3);
    TensorListShape<> shape(num_tensor, dims);
    for (int i = 0; i < num_tensor; ++i) {
      TensorShape<> tensor_shape;
      tensor_shape.resize(dims);
      for (int j = 0; j < dims; ++j) {
        tensor_shape[j] = this->RandInt(1, 200);
      }
      shape.set_tensor_shape(i, tensor_shape);
    }
    return shape;
  }

  TensorListShape<> GetSmallRandShape() {
    int num_tensor = this->RandInt(1, 32);
    int dims = this->RandInt(2, 3);
    TensorListShape<> shape(num_tensor, dims);
    for (int i = 0; i < num_tensor; ++i) {
      TensorShape<> tensor_shape;
      tensor_shape.resize(dims);
      for (int j = 0; j < dims; ++j) {
        tensor_shape[j] = this->RandInt(1, 64);
      }
      shape.set_tensor_shape(i, tensor_shape);
    }
    return shape;
  }

  /**
   * Initialize & check a TensorList based on an input shape
   * Allocate it as float
   */
  void SetupTensorList(TensorList<Backend> *tensor_list, const TensorListShape<> &shape,
                       vector<Index> *offsets) {
    const int num_tensor = shape.size();

    Index offset = 0;

    for (int i = 0; i < shape.size(); i++) {
      offsets->push_back(offset);
      offset += volume(shape[i]);
    }

    // Resize the buffer
    tensor_list->Resize(shape, DALI_FLOAT);

    // Check the internals
    ASSERT_TRUE(tensor_list->has_data());
    ASSERT_EQ(tensor_list->num_samples(), num_tensor);
    for (int i = 0; i < num_tensor; ++i) {
      ASSERT_NE(tensor_list->template mutable_tensor<float>(i), nullptr);
      ASSERT_EQ(tensor_list->tensor_shape(i), shape[i]);
    }
  }

  BatchContiguity inverse(BatchContiguity contiguity) {
    DALI_ENFORCE(contiguity != BatchContiguity::Automatic,
                 "This tests don't support BatchContiguity::Automatic");
    return contiguity == BatchContiguity::Contiguous ? BatchContiguity::Noncontiguous :
                                                       BatchContiguity::Contiguous;
  }
};

// Pairs of BackendType, ContiguousOption to be used in tests
using TensorListBackendContiguous = ::testing::Types<
    std::pair<CPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Contiguous>>,
    std::pair<CPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Noncontiguous>>,
    std::pair<GPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Contiguous>>,
    std::pair<GPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Noncontiguous>>>;

TYPED_TEST_SUITE(TensorListTest, TensorListBackendContiguous);

// Note: A TensorList in a valid state has a type. To get to a valid state, we
// can either set:
// type -> shape : setting shape triggers allocation
// shape & type : Resize triggers allocation
//
// Additionally, `reserve` can be called at any point.
//
// The following tests attempt to verify the correct behavior for all of
// these cases

TYPED_TEST(TensorListTest, TestGetTypeSizeBytes) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  // Give the tensor a type
  tl.template set_type<float>();

  ASSERT_EQ(tl._num_elements(), 0);
  ASSERT_EQ(tl.nbytes(), 0);
  ASSERT_FALSE(tl.has_data());

  // Give the tensor list a size. This
  // should trigger an allocation
  auto shape = this->GetRandShape();
  tl.Resize(shape);

  int num_tensor = shape.size();
  vector<Index> offsets;
  Index size = 0;
  for (int i = 0; i < shape.size(); i++) {
    offsets.push_back(size);
    size += volume(shape[i]);
  }

  // Validate the internals
  ASSERT_TRUE(tl.has_data());
  ASSERT_EQ(tl.num_samples(), num_tensor);
  ASSERT_EQ(tl._num_elements(), size);
  ASSERT_EQ(tl.nbytes(), size * sizeof(float));
  ASSERT_TRUE(IsType<float>(tl.type()));

  tl.reserve(shape.num_elements() * sizeof(float));

  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_NE(tl.raw_tensor(i), nullptr);
  }
}

TYPED_TEST(TensorListTest, ConsistentDeviceAndOrder) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  auto shape = uniform_list_shape(10, {1, 2, 3});
  TensorList<Backend> non_pinned, cpu_pinned, empty;
  non_pinned.SetContiguity(this->kContiguity);
  cpu_pinned.SetContiguity(this->kContiguity);
  empty.SetContiguity(this->kContiguity);

  non_pinned.set_pinned(false);
  // pin only for CPU, it doesn't make sense for GPU.
  cpu_pinned.set_pinned(std::is_same_v<CPUBackend, Backend>);

  non_pinned.Resize(shape, DALI_INT32);
  cpu_pinned.Resize(shape, DALI_INT32);

  // By default we use host order
  EXPECT_EQ(non_pinned.order(), AccessOrder::host());
  EXPECT_EQ(cpu_pinned.order(), AccessOrder::host());
  EXPECT_EQ(empty.order(), AccessOrder::host());

  if (std::is_same_v<CPUBackend, Backend>) {
    // On CPU pinned memory is associated with device after allocation, regular memory is not
    EXPECT_EQ(non_pinned.device_id(), CPU_ONLY_DEVICE_ID);
    EXPECT_EQ(cpu_pinned.device_id(), 0);
  } else {
    // On GPU we associate all allocations with current device, by default 0.
    EXPECT_EQ(non_pinned.device_id(), 0);
    EXPECT_EQ(cpu_pinned.device_id(), 0);
  }

  // uninitialized is not associated with any device id
  EXPECT_EQ(empty.device_id(), CPU_ONLY_DEVICE_ID);

  // No zeroing of order
  EXPECT_THROW(empty.set_order({}), std::runtime_error);
}

TYPED_TEST(TensorListTest, TestReserveResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl;

  auto shape = this->GetRandShape();
  tl.reserve(shape.num_elements() * sizeof(float));  // This reserve already makes it contiguous
  // Can't change the pinned status after allocation happened (reserve)
  ASSERT_THROW(tl.set_pinned(true), std::runtime_error);

  ASSERT_TRUE(tl.has_data());
  ASSERT_EQ(tl.capacity(), shape.num_elements() * sizeof(float));
  ASSERT_EQ(tl.nbytes(), 0);
  ASSERT_EQ(tl._num_elements(), 0);
  ASSERT_NE(contiguous_raw_data(tl), nullptr);

  // Give the tensor a type
  tl.template set_type<float>();

  ASSERT_EQ(tl._num_elements(), 0);
  ASSERT_EQ(tl.nbytes(), 0);
  ASSERT_TRUE(tl.has_data());

  // We already had the allocation, just give it a shape and a type
  tl.Resize(shape, DALI_FLOAT);

  int num_tensor = shape.size();
  vector<Index> offsets;
  Index size = 0;
  for (int i = 0; i < shape.size(); i++) {
    offsets.push_back(size);
    size += volume(shape[i]);
  }

  // Validate the internals
  ASSERT_TRUE(tl.has_data());
  ASSERT_EQ(tl.num_samples(), num_tensor);
  ASSERT_EQ(tl._num_elements(), size);
  ASSERT_EQ(tl.nbytes(), size * sizeof(float));
  ASSERT_TRUE(IsType<float>(tl.type()));


  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_NE(tl.raw_tensor(i), nullptr);
  }
}

TYPED_TEST(TensorListTest, TestResizeWithoutType) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  // Give the tensor a size - setting shape on non-typed TL is invalid and results in an error
  auto shape = this->GetRandShape();
  ASSERT_THROW(tl.Resize(shape), std::runtime_error);
}

TYPED_TEST(TensorListTest, TestSetNoType) {
  // After type is set we cannot revert to DALI_NO_TYPE
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  tl.set_type(DALI_FLOAT);
  ASSERT_THROW(tl.set_type(DALI_NO_TYPE), std::runtime_error);

  auto shape = this->GetRandShape();
  ASSERT_THROW(tl.Resize(shape, DALI_NO_TYPE), std::runtime_error);
}

TYPED_TEST(TensorListTest, TestGetContiguousPointer) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl;
  tl.SetContiguity(BatchContiguity::Contiguous);

  // Give the tensor a size and a type - uniform allocation
  auto shape = this->GetRandShape();
  tl.Resize(shape, DALI_UINT32);

  int num_tensor = shape.size();
  int64_t volume = shape.num_elements();

  // Verify the internals
  ASSERT_EQ(tl._num_elements(), volume);
  ASSERT_EQ(tl.num_samples(), num_tensor);
  ASSERT_EQ(tl.nbytes(), volume * sizeof(uint32_t));
  ASSERT_EQ(tl.type(), DALI_UINT32);
  ASSERT_TRUE(tl.IsContiguous());
  ASSERT_NE(contiguous_raw_data(tl), nullptr);
}

TYPED_TEST(TensorListTest, TestGetBytesThenAccess) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl;
  tl.SetContiguity(this->kContiguity);
  TensorList<Backend> sharers[2];
  sharers[0].SetContiguity(this->kContiguity);
  sharers[1].SetContiguity(this->inverse(this->kContiguity));

  // Allocate the sharer
  for (auto &sharer : sharers) {
    sharer.template set_type<float>();
    auto shape = this->GetRandShape();
    sharer.Resize(shape);

    // Share the data to give the tl bytes
    tl.ShareData(sharer);

    int num_tensor = shape.size();
    vector<Index> offsets;
    Index size = 0;
    for (int i = 0; i < shape.size(); i++) {
      offsets.push_back(size);
      size += volume(shape[i]);
    }

    // Verify the internals
    for (int i = 0; i < tl.num_samples(); i++) {
      ASSERT_EQ(tl.raw_tensor(i), sharer.raw_tensor(i));
    }
    ASSERT_EQ(tl._num_elements(), size);
    ASSERT_EQ(tl.nbytes(), size * sizeof(float));
    ASSERT_EQ(tl.type(), sharer.type());
    ASSERT_EQ(tl.num_samples(), num_tensor);
    ASSERT_TRUE(tl.shares_data());

    // Access can't change the underlying data type (which can happen only through Resize)
    ASSERT_THROW(tl.template mutable_tensor<int16_t>(0), std::runtime_error);
    ASSERT_THROW(tl.template mutable_tensor<double>(0), std::runtime_error);
    // We also cannot allocate bigger
    ASSERT_THROW(tl.Resize(tl.shape(), DALI_FLOAT64), std::runtime_error);
  }
}

TYPED_TEST(TensorListTest, TestZeroSizeResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  TensorListShape<> shape;
  tensor_list.template set_type<float>();
  tensor_list.Resize(shape);

  ASSERT_FALSE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), 0);
  ASSERT_EQ(tensor_list._num_elements(), 0);
  ASSERT_FALSE(tensor_list.shares_data());
}

TYPED_TEST(TensorListTest, TestMultipleZeroSizeResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  int num_tensor = this->RandInt(0, 128);
  auto shape = uniform_list_shape(num_tensor, TensorShape<>{0});
  tensor_list.Resize(shape, DALI_FLOAT);

  ASSERT_FALSE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), 0);
  ASSERT_EQ(tensor_list.num_samples(), num_tensor);
  ASSERT_EQ(tensor_list._num_elements(), 0);
  ASSERT_FALSE(tensor_list.shares_data());

  ASSERT_EQ(tensor_list.num_samples(), num_tensor);
  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tensor_list.template tensor<float>(i), nullptr);
    ASSERT_EQ(tensor_list.tensor_shape(i), TensorShape<>{0});
  }
}

TYPED_TEST(TensorListTest, TestFakeScalarResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  int num_scalar = this->RandInt(1, 128);
  auto shape = uniform_list_shape(num_scalar, {1});  // {1} on purpose
  tensor_list.template set_type<float>();
  tensor_list.Resize(shape);

  ASSERT_TRUE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), num_scalar * sizeof(float));
  ASSERT_EQ(tensor_list._num_elements(), num_scalar);
  ASSERT_FALSE(tensor_list.shares_data());

  for (int i = 0; i < num_scalar; ++i) {
    ASSERT_NE(tensor_list.raw_tensor(i), nullptr);
    ASSERT_EQ(tensor_list.tensor_shape(i), TensorShape<>{1});  // {1} on purpose
  }
}

TYPED_TEST(TensorListTest, TestTrueScalarResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  int num_scalar = this->RandInt(1, 128);
  auto shape = uniform_list_shape(num_scalar, TensorShape<>{});
  tensor_list.template set_type<float>();
  tensor_list.Resize(shape);

  ASSERT_TRUE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), num_scalar * sizeof(float));
  ASSERT_EQ(tensor_list._num_elements(), num_scalar);
  ASSERT_FALSE(tensor_list.shares_data());

  for (int i = 0; i < num_scalar; ++i) {
    ASSERT_NE(tensor_list.raw_tensor(i), nullptr);
    ASSERT_EQ(tensor_list.tensor_shape(i), TensorShape<>{});
  }
}

TYPED_TEST(TensorListTest, TestResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  // resize + check called in SetupTensorList
  this->SetupTensorList(&tensor_list, shape, &offsets);
}

TYPED_TEST(TensorListTest, TestMultipleResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  int rand = this->RandInt(1, 20);
  TensorListShape<> shape;
  vector<Index> offsets;
  int num_tensor = 0;
  for (int i = 0; i < rand; ++i) {
    offsets.clear();
    // Setup shape and offsets
    shape = this->GetRandShape();
    num_tensor = shape.size();
    Index offset = 0;
    for (int i = 0; i < shape.size(); i++) {
      offsets.push_back(offset);
      offset += volume(shape[i]);
    }
    // Resize the buffer
    tensor_list.Resize(shape, DALI_FLOAT);

    // Neither of the accessors can cause the allocation
    ASSERT_THROW(tensor_list.template mutable_tensor<double>(0), std::runtime_error);
    ASSERT_TRUE(tensor_list.has_data());
    ASSERT_NE(tensor_list.template mutable_tensor<float>(0), nullptr);

    ASSERT_EQ(tensor_list.num_samples(), num_tensor);
    for (int i = 0; i < num_tensor; ++i) {
      ASSERT_NE(tensor_list.raw_tensor(i), nullptr);
      ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    }
  }
}

TYPED_TEST(TensorListTest, TestCopy) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  tl.template set_type<float>();

  auto shape = this->GetRandShape();
  tl.Resize(shape);

  for (int i = 0; i < shape.num_samples(); i++) {
    tl.SetSourceInfo(i, to_string(i));
  }
  tl.SetLayout(std::string(shape.sample_dim(), 'X'));

  TensorList<Backend> tl2s[2];
  tl2s[0].SetContiguity(this->kContiguity);
  tl2s[1].SetContiguity(this->inverse(this->kContiguity));
  for (auto &tl2 : tl2s) {
    tl2.Copy(tl);

    ASSERT_EQ(tl.num_samples(), tl2.num_samples());
    ASSERT_EQ(tl.type(), tl2.type());
    ASSERT_EQ(tl.shape().num_elements(), tl2.shape().num_elements());
    ASSERT_EQ(tl.GetLayout(), tl2.GetLayout());

    for (int i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(tl.tensor_shape(i), tl2.tensor_shape(i));
      ASSERT_EQ(volume(tl.tensor_shape(i)), volume(tl2.tensor_shape(i)));
      ASSERT_EQ(tl.GetMeta(i).GetSourceInfo(), tl2.GetMeta(i).GetSourceInfo());
    }
  }
}

TYPED_TEST(TensorListTest, TestCopyEmpty) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tl, uninitialized;
  tl.SetContiguity(this->kContiguity);

  tl.template set_type<float>();
  tl.SetLayout("XX");

  TensorList<Backend> tl2s[2];
  tl2s[0].SetContiguity(this->kContiguity);
  tl2s[1].SetContiguity(this->inverse(this->kContiguity));
  for (auto &tl2 : tl2s) {
    tl2.Copy(tl);
    ASSERT_EQ(tl.num_samples(), tl2.num_samples());
    ASSERT_EQ(tl.type(), tl2.type());
    ASSERT_EQ(tl._num_elements(), tl2._num_elements());
    ASSERT_EQ(tl.GetLayout(), tl2.GetLayout());

    tl2.Copy(uninitialized);
    ASSERT_FALSE(tl2.has_data());
    ASSERT_EQ(uninitialized.num_samples(), tl2.num_samples());
    ASSERT_EQ(uninitialized.type(), tl2.type());
    ASSERT_EQ(uninitialized._num_elements(), tl2._num_elements());
    ASSERT_EQ(uninitialized.GetLayout(), tl2.GetLayout());
  }
}

TYPED_TEST(TensorListTest, TestTypeChangeError) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);
  auto shape = this->GetRandShape();

  tensor_list.set_type(DALI_UINT8);
  tensor_list.set_type(DALI_FLOAT);
  tensor_list.set_type(DALI_INT32);
  tensor_list.Resize(shape);
  ASSERT_NE(tensor_list.template mutable_tensor<int32_t>(0), nullptr);

  // After we have a shape, we cannot change the type with set_type
  ASSERT_THROW(tensor_list.set_type(DALI_FLOAT), std::runtime_error);

  tensor_list.Resize(shape, DALI_FLOAT);
  ASSERT_NE(tensor_list.template mutable_tensor<float>(0), nullptr);
}

TYPED_TEST(TensorListTest, TestTypeChange) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  DALIDataType initial_type = DALI_FLOAT;
  std::array<DALIDataType, 4> types = {DALI_FLOAT, DALI_INT32, DALI_UINT8, DALI_FLOAT64};
  const auto *base_ptr =
      this->kContiguity == BatchContiguity::Contiguous ? contiguous_raw_data(tensor_list) : nullptr;
  size_t nbytes = shape.num_elements() * sizeof(float);

  // Save the pointers
  std::vector<const void *> ptrs;
  for (int i = 0; i < tensor_list.num_samples(); i++) {
    ptrs.push_back(tensor_list.raw_tensor(i));
  }

  for (auto new_type : types) {
    if (initial_type != new_type) {
      // Simply changing the type of the buffer is not allowed
      ASSERT_THROW(tensor_list.set_type(new_type), std::runtime_error);
      tensor_list.Resize(shape, new_type);
    }

    // Check the internals
    ASSERT_EQ(tensor_list.num_samples(), shape.num_samples());
    ASSERT_EQ(tensor_list.shape(), shape);
    ASSERT_EQ(tensor_list.sample_dim(), shape.sample_dim());
    ASSERT_EQ(tensor_list.type(), new_type);
    for (int i = 0; i < tensor_list.num_samples(); ++i) {
      ASSERT_NE(tensor_list.raw_tensor(i), nullptr);
      ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    }

    // The side-effects of only reallocating when we need a bigger buffer, we may use padding
    if (TypeTable::GetTypeInfo(new_type).size() <= TypeTable::GetTypeInfo(initial_type).size()) {
      if (this->kContiguity == BatchContiguity::Contiguous) {
        ASSERT_EQ(contiguous_raw_data(tensor_list), base_ptr);
      } else {
        for (int i = 0; i < tensor_list.num_samples(); ++i) {
          ASSERT_EQ(tensor_list.raw_tensor(i), ptrs[i]);
        }
      }
    }

    ASSERT_EQ(nbytes / TypeTable::GetTypeInfo(initial_type).size() *
                  TypeTable::GetTypeInfo(new_type).size(),
              tensor_list.nbytes());
  }
}

TYPED_TEST(TensorListTest, DeviceIdPropagationMultiGPU) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  // This test doesn't pin state to Noncontiguous as it prohibits sharing noncontiguous data
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  }
  constexpr bool is_device = std::is_same_v<Backend, GPUBackend>;
  constexpr bool is_pinned = !is_device;
  AccessOrder order = AccessOrder::host();
  TensorListShape<> shape{{42}};
  for (int device_id = 0; device_id < num_devices; device_id++) {
    TensorList<Backend> batch;
    batch.SetContiguity(BatchContiguity::Automatic);
    DeviceGuard dg(device_id);
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
    batch.ShareData(ptr, shape.num_elements() * sizeof(uint8_t), is_pinned, shape, DALI_UINT8,
                    device_id, order);
    ASSERT_EQ(batch.device_id(), device_id);
    ASSERT_EQ(batch.order().device_id(), AccessOrder::host().device_id());
    ASSERT_NE(batch.order().device_id(), batch.device_id());
  }
}

TYPED_TEST(TensorListTest, TestShareData) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorList<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  // Create a new tensor_list w/ a smaller data type
  TensorList<Backend> tensor_lists[2];
  tensor_lists[0].SetContiguity(this->kContiguity);
  tensor_lists[1].SetContiguity(this->inverse(this->kContiguity));

  for (auto &tensor_list2 : tensor_lists) {
    // Share the data
    tensor_list2.ShareData(tensor_list);
    ASSERT_EQ(tensor_list2.is_pinned(), tensor_list.is_pinned());
    ASSERT_EQ(tensor_list2.order(), tensor_list.order());
    // We need to use the same size as the underlying buffer
    // N.B. using other type is UB in most cases
    auto flattened_shape = collapse_dims(shape, {std::make_pair(0, shape.sample_dim())});
    tensor_list2.template set_type<float>();
    tensor_list2.Resize(flattened_shape);

    // Make sure the pointers match
    for (int i = 0; i < tensor_list.num_samples(); ++i) {
      ASSERT_EQ(tensor_list.raw_tensor(i), tensor_list2.raw_tensor(i));
    }
    ASSERT_TRUE(tensor_list2.shares_data());

    // Verify the default dims of the tensor_list 2
    ASSERT_EQ(tensor_list2._num_elements(), tensor_list._num_elements());

    // Resize the tensor_list2 to match the shape of tensor_list
    tensor_list2.Resize(shape);

    // Check the internals
    ASSERT_TRUE(tensor_list2.shares_data());
    ASSERT_EQ(tensor_list2.nbytes(), tensor_list.nbytes());
    ASSERT_EQ(tensor_list2.num_samples(), tensor_list.num_samples());
    ASSERT_EQ(tensor_list2._num_elements(), tensor_list._num_elements());
    for (int i = 0; i < tensor_list.num_samples(); ++i) {
      ASSERT_EQ(tensor_list.raw_tensor(i), tensor_list2.raw_tensor(i));
      ASSERT_EQ(tensor_list2.tensor_shape(i), shape[i]);
    }

    // Trigger allocation through buffer API, verify we cannot do that
    ASSERT_THROW(tensor_list2.template mutable_tensor<double>(0), std::runtime_error);
    tensor_list2.Reset();
    ASSERT_FALSE(tensor_list2.shares_data());

    // Check the internals
    ASSERT_EQ(tensor_list2._num_elements(), 0);
    ASSERT_EQ(tensor_list2.nbytes(), 0);
    ASSERT_EQ(tensor_list2.num_samples(), 0);
    ASSERT_EQ(tensor_list2.shape(), TensorListShape<>());
  }
}

/***************************************************************************************************
 The section below is contains tests of original TensorList from tensor_vector_test.cc
 with the only changes being rename to the TensorList.
 **************************************************************************************************/

template <typename T>
class TensorListSuite : public ::testing::Test {
 protected:
  void validate(const TensorList<T> &tv) {
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

TYPED_TEST_SUITE(TensorListSuite, Backends);

TYPED_TEST(TensorListSuite, SetupAndSetSizeNoncontiguous) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorList<TypeParam> tv;
  tv.set_pinned(false);
  tv.set_order(order);
  tv.set_sample_dim(2);
  tv.SetLayout("XY");
  tv.SetContiguity(BatchContiguity::Noncontiguous);
  tv.SetSize(3);


  auto empty_2d = TensorShape<>{0, 0};
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_NO_TYPE);
  }

  // Setting just the type
  tv.set_type(DALI_INT32);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }

  Tensor<TypeParam> t;
  t.set_pinned(false);
  t.set_order(order);
  t.Resize({2, 3}, DALI_INT32);
  t.SetLayout("XY");

  // We need to propagate device id and order
  tv.set_device_id(t.device_id());
  tv.set_order(t.order());

  for (int i = 0; i < 3; i++) {
    tv.SetSample(i, t);
    EXPECT_EQ(tv[i].raw_data(), t.raw_data());
    EXPECT_EQ(tv[i].shape(), t.shape());
    EXPECT_EQ(tv[i].type(), t.type());
  }

  tv.SetSize(4);
  // New one should be empty
  EXPECT_EQ(tv[3].raw_data(), nullptr);
  EXPECT_EQ(tv[3].shape(), empty_2d);
  EXPECT_EQ(tv[3].type(), DALI_INT32);

  tv.SetSize(2);
  tv.SetSize(3);
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(tv[i].raw_data(), t.raw_data());
    EXPECT_EQ(tv[i].shape(), t.shape());
    EXPECT_EQ(tv[i].type(), t.type());
  }

  // As we were sharing, the share is removed and no allocation should happen
  for (int i = 2; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }

  // There should be no access to the out of bounds one
  EXPECT_THROW(tv[3], std::runtime_error);

  // We are sharing, no way to make it bigger
  EXPECT_THROW(tv.Resize(uniform_list_shape(3, {10, 12, 3})), std::runtime_error);
}

TYPED_TEST(TensorListSuite, ResizeWithRecreate) {
  // Check if the tensors that get back to the scope are correctly typed
  for (auto contiguity : {BatchContiguity::Contiguous, BatchContiguity::Noncontiguous}) {
    TensorList<TypeParam> tv;
    tv.SetContiguity(contiguity);
    tv.SetLayout("HWC");
    tv.set_pinned(true);
    auto shape0 = TensorListShape<>({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
    tv.Resize(shape0, DALI_UINT8);
    EXPECT_EQ(tv.shape().num_samples(), 3);
    for (int i = 0; i < 3; i++) {
      EXPECT_EQ(tv[i].type(), DALI_UINT8);
      EXPECT_EQ(tv[i].shape(), shape0[i]);
      EXPECT_EQ(tv.GetMeta(i).GetLayout(), "HWC");
    }

    auto shape1 = TensorListShape<>({{1, 2}, {3, 4}});
    tv.Resize(shape1, DALI_FLOAT);
    tv.SetLayout("WC");
    EXPECT_EQ(tv.shape().num_samples(), 2);
    for (int i = 0; i < 2; i++) {
      EXPECT_EQ(tv[i].type(), DALI_FLOAT);
      EXPECT_EQ(tv[i].shape(), shape1[i]);
      EXPECT_EQ(tv.GetMeta(i).GetLayout(), "WC");
    }

    auto shape2 = TensorListShape<>({{2, 3}, {4, 5}, {5, 6}, {6, 7}, {7, 8}});
    tv.Resize(shape2, DALI_FLOAT64);
    EXPECT_EQ(tv.shape().num_samples(), 5);
    for (int i = 0; i < 5; i++) {
      EXPECT_EQ(tv[i].type(), DALI_FLOAT64);
      EXPECT_EQ(tv[i].shape(), shape2[i]);
      EXPECT_EQ(tv.GetMeta(i).GetLayout(), "WC");
    }

    auto shape3 = TensorListShape<>({{2, 3}, {4, 5}, {5, 6}});
    tv.SetSize(3);
    EXPECT_EQ(tv.shape().num_samples(), 3);
    for (int i = 0; i < 3; i++) {
      EXPECT_EQ(tv[i].type(), DALI_FLOAT64);
      EXPECT_EQ(tv[i].shape(), shape3[i]);
      EXPECT_EQ(tv.GetMeta(i).GetLayout(), "WC");
    }

    auto shape4 = TensorListShape<>({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
    tv.SetLayout("HWC");
    tv.Resize(shape4, DALI_UINT8);
    EXPECT_EQ(tv.shape().num_samples(), 3);
    for (int i = 0; i < 3; i++) {
      EXPECT_EQ(tv[i].type(), DALI_UINT8);
      EXPECT_EQ(tv[i].shape(), shape4[i]);
      EXPECT_EQ(tv.GetMeta(i).GetLayout(), "HWC");
    }

    auto shape5 =
        TensorListShape<>({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    tv.SetSize(6);
    EXPECT_EQ(tv.shape().num_samples(), 6);
    for (int i = 0; i < 6; i++) {
      EXPECT_EQ(tv[i].type(), DALI_UINT8);
      EXPECT_EQ(tv[i].shape(), shape5[i]);
      EXPECT_EQ(tv.GetMeta(i).GetLayout(), "HWC");
    }
  }
}


TYPED_TEST(TensorListSuite, ResizeWithRecreate0d) {
  // Check if the tensors that get back to the scope are correctly typed
  // Scalar values cannot exist as 0-volume tensors, so they always have some allocation
  for (auto contiguity : {BatchContiguity::Contiguous, BatchContiguity::Noncontiguous}) {
    TensorList<TypeParam> tv;
    tv.SetContiguity(contiguity);
    tv.set_pinned(true);
    auto shape0 = uniform_list_shape(3, TensorShape<0>{});
    tv.Resize(shape0, DALI_UINT8);
    EXPECT_EQ(tv.shape().num_samples(), 3);
    EXPECT_EQ(tv.shape().sample_dim(), 0);
    for (int i = 0; i < 3; i++) {
      EXPECT_EQ(tv[i].type(), DALI_UINT8);
      EXPECT_EQ(tv[i].shape(), TensorShape<0>{});
      EXPECT_NE(tv[i].raw_data(), nullptr);
    }

    tv.SetSize(4);

    EXPECT_EQ(tv.shape().num_samples(), 4);
    EXPECT_EQ(tv.shape().sample_dim(), 0);

    for (int i = 0; i < 4; i++) {
      EXPECT_EQ(tv[i].type(), DALI_UINT8);
      EXPECT_EQ(tv[i].shape(), TensorShape<0>{});
      EXPECT_NE(tv[i].raw_data(), nullptr);
    }

    auto shape1 = uniform_list_shape(5, TensorShape<2>{1, 2});
    tv.Resize(shape1);
    EXPECT_EQ(tv.shape().num_samples(), 5);
    EXPECT_EQ(tv.shape().sample_dim(), 2);

    auto shape2 = uniform_list_shape(6, TensorShape<0>{});
    tv.Resize(shape2, DALI_UINT8);
    tv.SetSize(6);
    EXPECT_EQ(tv.shape().num_samples(), 6);
    for (int i = 0; i < 6; i++) {
      EXPECT_EQ(tv[i].type(), DALI_UINT8);
      EXPECT_EQ(tv[i].shape(), TensorShape<0>{});
      EXPECT_NE(tv[i].raw_data(), nullptr);
    }
  }
}


TYPED_TEST(TensorListSuite, SetupLikeMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = is_device ? AccessOrder(cuda_stream, device) : AccessOrder::host();
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  TensorList<TypeParam> tv_like_t;
  tv_like_t.SetupLike(t);

  EXPECT_EQ(t.device_id(), tv_like_t.device_id());
  EXPECT_EQ(t.order(), tv_like_t.order());
  EXPECT_EQ(t.is_pinned(), tv_like_t.is_pinned());
  EXPECT_EQ(t.shape().sample_dim(), tv_like_t.sample_dim());
  EXPECT_EQ(t.GetLayout(), tv_like_t.GetLayout());

  TensorList<TypeParam> tv_like_tv;
  tv_like_tv.SetupLike(tv_like_t);

  EXPECT_EQ(tv_like_t.device_id(), tv_like_tv.device_id());
  EXPECT_EQ(tv_like_t.order(), tv_like_tv.order());
  EXPECT_EQ(tv_like_t.is_pinned(), tv_like_tv.is_pinned());
  EXPECT_EQ(tv_like_t.sample_dim(), tv_like_tv.sample_dim());
  EXPECT_EQ(tv_like_t.GetLayout(), tv_like_tv.GetLayout());
}

template <typename Backend>
std::vector<std::pair<std::string, std::function<void(TensorList<Backend> &)>>> SetRequiredSetters(
    int sample_dim, DALIDataType type, TensorLayout layout, bool pinned, int device_id) {
  return {
      {"sample dim", [sample_dim](TensorList<Backend> &t) { t.set_sample_dim(sample_dim); }},
      {"type", [type](TensorList<Backend> &t) { t.set_type(type); }},
      {"layout", [layout](TensorList<Backend> &t) { t.SetLayout(layout); }},
      {"device id", [device_id](TensorList<Backend> &t) { t.set_device_id(device_id); }},
      {"pinned", [pinned](TensorList<Backend> &t) { t.set_pinned(pinned); }},
  };
}

TYPED_TEST(TensorListSuite, PartialSetupSetMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = AccessOrder(cuda_stream, device);
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  // set size to be checked. Copy doesn't make sense
  auto setups = SetRequiredSetters<TypeParam>(3, DALI_INT32, "HWC", true, device);
  for (size_t excluded = 0; excluded < setups.size(); excluded++) {
    std::vector<size_t> idxs(setups.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    do {
      TensorList<TypeParam> tv;
      tv.SetContiguity(BatchContiguity::Noncontiguous);
      tv.SetSize(4);
      tv.set_pinned(false);
      for (auto idx : idxs) {
        if (idx == excluded) {
          continue;
        }
        setups[idx].second(tv);
      }
      try {
        tv.SetSample(0, t);
        FAIL() << "Exception was expected with excluded: " << setups[excluded].first;
      } catch (std::runtime_error &e) {
        auto expected = "Sample must have the same " + setups[excluded].first;
        EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
            << expected << "\n====\nvs\n====\n"
            << e.what();
      } catch (...) { FAIL() << "Unexpected exception"; }
    } while (std::next_permutation(idxs.begin(), idxs.end()));
  }
}


template <typename Backend>
std::vector<std::pair<std::string, std::function<void(TensorList<Backend> &)>>> CopyRequiredSetters(
    int sample_dim, DALIDataType type, TensorLayout layout) {
  return {{"sample dim", [sample_dim](TensorList<Backend> &t) { t.set_sample_dim(sample_dim); }},
          {"type", [type](TensorList<Backend> &t) { t.set_type(type); }},
          {"layout", [layout](TensorList<Backend> &t) { t.SetLayout(layout); }}};
}


TYPED_TEST(TensorListSuite, PartialSetupCopyMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = is_device ? AccessOrder(cuda_stream, device) : AccessOrder::host();
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  // set size to be checked. Copy doesn't make sense
  auto setups = CopyRequiredSetters<TypeParam>(3, DALI_INT32, "HWC");
  for (size_t excluded = 0; excluded < setups.size(); excluded++) {
    std::vector<size_t> idxs(setups.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    do {
      TensorList<TypeParam> tv;
      tv.SetContiguity(BatchContiguity::Noncontiguous);
      tv.SetSize(4);
      for (auto idx : idxs) {
        if (idx == excluded) {
          continue;
        }
        setups[idx].second(tv);
      }
      try {
        tv.CopySample(0, t);
        FAIL() << "Exception was expected with excluded: " << setups[excluded].first;
      } catch (std::runtime_error &e) {
        auto expected = "Sample must have the same " + setups[excluded].first;
        EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
            << expected << "\n====\nvs\n====\n"
            << e.what();
      } catch (...) { FAIL() << "Unexpected exception"; }
    } while (std::next_permutation(idxs.begin(), idxs.end()));
  }
}


TYPED_TEST(TensorListSuite, FullSetupSetMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = AccessOrder(cuda_stream, device);
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  // set size to be checked. Copy doesn't make sense
  auto setups = SetRequiredSetters<TypeParam>(3, DALI_INT32, "HWC", true, device);
  std::vector<size_t> idxs(setups.size());
  std::iota(idxs.begin(), idxs.end(), 0);
  do {
    TensorList<TypeParam> tv;
    tv.SetContiguity(BatchContiguity::Noncontiguous);
    tv.SetSize(4);
    for (auto idx : idxs) {
      setups[idx].second(tv);
    }
    tv.SetSample(0, t);
    EXPECT_EQ(tv[0].raw_data(), t.raw_data());
  } while (std::next_permutation(idxs.begin(), idxs.end()));
}


namespace {

void FillWithNumber(TensorList<CPUBackend> &tv, uint8_t number) {
  for (int i = 0; i < tv.shape().num_elements(); i++) {
    // We utilize the contiguity of the TensorList
    tv.mutable_tensor<uint8_t>(0)[i] = number;
  }
}

template <typename T>
void FillWithNumber(SampleView<CPUBackend> sample, T number) {
  for (int i = 0; i < sample.shape().num_elements(); i++) {
    // We utilize the contiguity of the TensorList
    sample.mutable_data<T>()[i] = number;
  }
}

void FillWithNumber(TensorList<GPUBackend> &tv, uint8_t number) {
  // We utilize the contiguity of the TensorList
  cudaMemset(tv.mutable_tensor<uint8_t>(0), number, tv.shape().num_elements());
}

template <typename T>
void FillWithNumber(SampleView<GPUBackend> sample, T number) {
  std::vector<T> buffer(sample.shape().num_elements(), number);
  cudaMemcpy(sample.mutable_data<T>(), buffer.data(), sample.shape().num_elements() * sizeof(T),
             cudaMemcpyHostToDevice);
}


void CompareWithNumber(Tensor<CPUBackend> &t, uint8_t number) {
  for (int i = 0; i < t.shape().num_elements(); i++) {
    EXPECT_EQ(t.data<uint8_t>()[i], number);
  }
}

template <typename T>
void CompareWithNumber(ConstSampleView<CPUBackend> t, T number) {
  for (int i = 0; i < t.shape().num_elements(); i++) {
    EXPECT_EQ(t.data<T>()[i], number);
  }
}


void CompareWithNumber(Tensor<GPUBackend> &t, uint8_t number) {
  std::vector<uint8_t> buffer(t.shape().num_elements());
  cudaMemcpy(buffer.data(), t.data<uint8_t>(), t.shape().num_elements(), cudaMemcpyDeviceToHost);
  for (auto b : buffer) {
    EXPECT_EQ(b, number);
  }
}

template <typename T>
void CompareWithNumber(ConstSampleView<GPUBackend> sample, T number) {
  std::vector<T> buffer(sample.shape().num_elements());
  cudaMemcpy(buffer.data(), sample.data<T>(), sample.shape().num_elements() * sizeof(T),
             cudaMemcpyDeviceToHost);
  for (auto b : buffer) {
    EXPECT_EQ(b, number);
  }
}

template <typename Backend>
Tensor<Backend> ReturnTvAsTensor(uint8_t number) {
  TensorList<Backend> tv;
  tv.SetContiguity(BatchContiguity::Contiguous);
  tv.Resize(uniform_list_shape(4, {1, 2, 3}), DALI_UINT8);
  tv.SetLayout("HWC");
  FillWithNumber(tv, number);
  return tv.AsTensor();
}

}  // namespace

TYPED_TEST(TensorListSuite, ResizeSetSize) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorList<TypeParam> tv;
  tv.set_pinned(false);
  tv.set_order(order);
  tv.set_sample_dim(2);
  tv.SetLayout("XY");
  tv.SetContiguity(BatchContiguity::Contiguous);
  tv.SetSize(3);


  auto empty_2d = TensorShape<>{0, 0};
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_NO_TYPE);
  }

  // Setting just the type
  tv.set_type(DALI_INT32);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape);
  tv.SetLayout("HWC");

  const auto *base = static_cast<const uint8_t *>(contiguous_raw_data(tv));

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), base);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
    base += sizeof(int32_t) * new_shape[i].num_elements();
  }

  tv.SetSize(4);
  EXPECT_TRUE(tv.IsContiguous());

  auto empty_3d = TensorShape<>{0, 0, 0};

  // New one should be empty
  EXPECT_EQ(tv[3].raw_data(), nullptr);
  EXPECT_EQ(tv[3].shape(), empty_3d);
  EXPECT_EQ(tv[3].type(), DALI_INT32);

  tv.SetSize(2);
  tv.SetSize(3);
  EXPECT_TRUE(tv.IsContiguous());

  base = static_cast<const uint8_t *>(contiguous_raw_data(tv));

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(tv[i].raw_data(), base);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
    base += sizeof(int32_t) * new_shape[i].num_elements();
  }

  // As we are contiguous, thus we are sharing the contiguous buffer via every sample,
  // the share is removed and no allocation should happen when we made the size bigger
  for (int i = 2; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_3d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
}


TYPED_TEST(TensorListSuite, NoForcedChangeContiguousToNoncontiguous) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Contiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  // State was forced to be contiguous, cannot change it with just Resize.
  EXPECT_THROW(tv.Resize(new_shape, DALI_FLOAT, BatchContiguity::Noncontiguous),
               std::runtime_error);
}

TYPED_TEST(TensorListSuite, NoForcedChangeNoncontiguousToContiguous) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Noncontiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  // State was forced to be noncontiguous, cannot change it with just Resize.
  EXPECT_THROW(tv.Resize(new_shape, DALI_FLOAT, BatchContiguity::Contiguous), std::runtime_error);
}


TYPED_TEST(TensorListSuite, ContiguousResize) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorList<TypeParam> tv;
  tv.set_order(order);
  tv.SetContiguity(BatchContiguity::Contiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT);
  EXPECT_TRUE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    FillWithNumber(tv[i], 1 + i * 1.f);
  }

  for (int i = 0; i < 3; i++) {
    tv.CopySample(i, tv, i);
  }

  const auto *base = static_cast<const uint8_t *>(contiguous_raw_data(tv));

  EXPECT_TRUE(tv.IsContiguous());
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), base);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
    base += sizeof(float) * new_shape[i].num_elements();
    CompareWithNumber(tv[i], 1 + i * 1.f);
  }

  // Cannot copy without exact shape match when contiguous
  EXPECT_THROW(tv.CopySample(0, tv, 1), std::runtime_error);
  EXPECT_THROW(tv.CopySample(2, tv, 1), std::runtime_error);
}


TYPED_TEST(TensorListSuite, NoncontiguousResize) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorList<TypeParam> tv;
  tv.set_order(order);
  tv.SetContiguity(BatchContiguity::Noncontiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT);
  EXPECT_FALSE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    FillWithNumber(tv[i], 1 + i * 1.f);
  }

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
    CompareWithNumber(tv[i], 1 + i * 1.f);
  }

  // Cannot copy without exact shape match when contiguous
  tv.CopySample(0, tv, 1);
  tv.CopySample(2, tv, 1);

  EXPECT_FALSE(tv.IsContiguous());
  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), new_shape[1]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
    CompareWithNumber(tv[i], 2.f);
  }
}


TYPED_TEST(TensorListSuite, ResizeSample) {
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Automatic);

  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT, BatchContiguity::Contiguous);
  EXPECT_TRUE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    FillWithNumber(tv[i], 1 + i * 1.f);
  }

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
    CompareWithNumber(tv[i], 1 + i * 1.f);
  }

  auto new_sample_shape = TensorShape<>{10, 10, 3};
  new_shape.set_tensor_shape(1, new_sample_shape);

  tv.ResizeSample(1, new_sample_shape);

  EXPECT_FALSE(tv.IsContiguous());
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
  }

  FillWithNumber(tv[1], 42.f);

  EXPECT_EQ(tv[1].shape(), new_sample_shape);
  EXPECT_EQ(tv[1].type(), DALI_FLOAT);
  CompareWithNumber(tv[1], 42.f);

  auto new_smaller_sample_shape = TensorShape<>{5, 5, 3};
  new_shape.set_tensor_shape(1, new_smaller_sample_shape);

  tv.ResizeSample(1, new_smaller_sample_shape);

  EXPECT_FALSE(tv.IsContiguous());
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
  }

  FillWithNumber(tv[1], 42.f);

  EXPECT_EQ(tv[1].shape(), new_smaller_sample_shape);
  EXPECT_EQ(tv[1].type(), DALI_FLOAT);
  CompareWithNumber(tv[1], 42.f);
}


TYPED_TEST(TensorListSuite, ResizeSampleProhibited) {
  TensorList<TypeParam> tv;
  auto sample_shape = TensorShape<>{10, 10};
  tv.SetContiguity(BatchContiguity::Automatic);
  EXPECT_THROW(tv.ResizeSample(0, sample_shape), std::runtime_error);
  auto shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(shape, DALI_FLOAT, BatchContiguity::Contiguous);

  EXPECT_THROW(tv.ResizeSample(1, sample_shape), std::runtime_error);
}


TYPED_TEST(TensorListSuite, BreakContiguity) {
  TensorList<TypeParam> tv;
  // anything goes
  tv.SetContiguity(BatchContiguity::Automatic);

  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT, BatchContiguity::Contiguous);
  EXPECT_TRUE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    FillWithNumber(tv[i], 1 + i * 1.f);
  }

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
    CompareWithNumber(tv[i], 1 + i * 1.f);
  }

  Tensor<TypeParam> t;
  t.Resize({2, 3, 4}, DALI_FLOAT);
  SampleView<TypeParam> v{t.template mutable_data<float>(), t.shape(), DALI_FLOAT};
  FillWithNumber(v, 42.f);

  tv.SetSample(1, t);
  EXPECT_FALSE(tv.IsContiguous());

  EXPECT_EQ(tv[1].raw_data(), t.raw_data());
  EXPECT_EQ(tv[1].shape(), t.shape());
  EXPECT_EQ(tv[1].type(), DALI_FLOAT);
  CompareWithNumber(tv[1], 42.f);
}


TYPED_TEST(TensorListSuite, CoalescingAllocation) {
  TensorList<TypeParam> tv;
  // anything goes
  tv.SetContiguity(BatchContiguity::Automatic);

  auto first_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(first_shape, DALI_FLOAT, BatchContiguity::Noncontiguous);
  EXPECT_FALSE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), first_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  auto smaller_shape = TensorListShape<>{{1, 2, 1}, {1, 3, 1}, {3, 4, 1}};
  tv.Resize(smaller_shape, DALI_FLOAT, BatchContiguity::Noncontiguous);
  EXPECT_FALSE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), smaller_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  // Second and third sample are significantly bigger, reallocate and coalesce
  auto bigger_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 40}, {3, 4, 50}};
  tv.Resize(bigger_shape, DALI_FLOAT);
  EXPECT_TRUE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), bigger_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  // Go back
  tv.Resize(bigger_shape, DALI_FLOAT, BatchContiguity::Noncontiguous);
  EXPECT_FALSE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), bigger_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  // Go back again
  tv.Resize(bigger_shape, DALI_FLOAT, BatchContiguity::Contiguous);
  EXPECT_TRUE(tv.IsContiguous());

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), bigger_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  // Just stay
  tv.Resize(bigger_shape, DALI_FLOAT);
  EXPECT_TRUE(tv.IsContiguous());
}


TYPED_TEST(TensorListSuite, Reserve) {
  // Verify that we still keep the memory reserved in sample mode
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Automatic);
  tv.reserve(100, 4);

  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 50}};
  tv.Resize(new_shape, DALI_FLOAT, BatchContiguity::Noncontiguous);

  EXPECT_FALSE(tv.IsContiguous());

  tv.SetSize(4);
  auto capacity = tv._chunks_capacity();

  for (int i = 0; i < 4; i++) {
    auto expected_capacity =
        std::max<int>(100, i < 3 ? sizeof(float) * new_shape[i].num_elements() : 0);
    EXPECT_EQ(capacity[i], expected_capacity);
  }
}

// Check if interleaving any of
// * set_pinned
// * set_type
// * Resize(shape)
// * Resize(shape, type)
// * reserve
// behaves as it is supposed to - that is set_type always first, set_type before Reshape,
// reserve can be without type.

// TODO(klecki): reverse pinned and capacity tests

TYPED_TEST(TensorListSuite, PinnedAfterReserveThrows) {
  // pinned status cannot be changed after allocation
  TensorList<TypeParam> tv_0, tv_1;
  tv_0.reserve(100);
  EXPECT_THROW(tv_0.set_pinned(false), std::runtime_error);
  tv_1.reserve(100, 2);
  EXPECT_THROW(tv_1.set_pinned(false), std::runtime_error);
  TensorList<TypeParam> tv_2(2), tv_3(2);
  tv_2.reserve(100);
  EXPECT_THROW(tv_2.set_pinned(false), std::runtime_error);
  tv_3.reserve(100, 2);
  EXPECT_THROW(tv_3.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorListSuite, PinnedAfterResizeThrows) {
  TensorList<TypeParam> tv;
  tv.reserve(100);
  // Resize can only be used when type is known
  EXPECT_THROW(tv.Resize({{2, 4}, {4, 2}}), std::runtime_error);
  tv.Resize({{2, 4}, {4, 2}}, DALI_INT32);
  ASSERT_EQ(tv.num_samples(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv[0].type(), DALI_INT32);
  EXPECT_EQ(tv[1].type(), DALI_INT32);
  // pinned status cannot be changed after allocation
  ASSERT_THROW(tv.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorListSuite, PinnedBeforeResizeContiguous) {
  TensorList<TypeParam> tv;
  tv.set_pinned(false);
  tv.reserve(100);
  // Resize can only be used when type is known
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

TYPED_TEST(TensorListSuite, PinnedBeforeResizeNoncontiguous) {
  TensorList<TypeParam> tv;
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


TYPED_TEST(TensorListSuite, TensorListAsTensorAccess) {
  TensorList<TypeParam> tv;
  constexpr uint8_t kMagicNumber = 42;
  tv.SetContiguity(BatchContiguity::Contiguous);
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
    auto tensor = ReturnTvAsTensor<TypeParam>(kMagicNumber);
    auto expected_shape = TensorShape<>{4, 1, 2, 3};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.type(), DALI_UINT8);
    EXPECT_EQ(tensor.GetLayout(), "NHWC");
    CompareWithNumber(tensor, kMagicNumber);
  }
}

TYPED_TEST(TensorListSuite, EmptyTensorListAsTensorAccess) {
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Contiguous);
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
    EXPECT_EQ(tensor_1d.raw_data(), contiguous_raw_data(tv));
    EXPECT_EQ(tensor_1d.raw_data(), nullptr);
    EXPECT_EQ(tensor_2d.shape(), shape_2d);
    EXPECT_EQ(tensor_2d.type(), DALI_INT32);
    EXPECT_EQ(tensor_2d.raw_data(), contiguous_raw_data(tv));
    EXPECT_EQ(tensor_2d.raw_data(), nullptr);
  }

  tv.reserve(1000);

  {
    EXPECT_THROW(tv.AsReshapedTensor(shape_0d), std::runtime_error);  // empty shape has volume = 1
    auto tensor_1d = tv.AsReshapedTensor(shape_1d);
    auto tensor_2d = tv.AsReshapedTensor(shape_2d);
    EXPECT_EQ(tensor_1d.shape(), shape_1d);
    EXPECT_EQ(tensor_1d.type(), DALI_INT32);
    EXPECT_EQ(tensor_1d.raw_data(), contiguous_raw_data(tv));
    EXPECT_NE(tensor_1d.raw_data(), nullptr);
    EXPECT_EQ(tensor_2d.shape(), shape_2d);
    EXPECT_EQ(tensor_2d.type(), DALI_INT32);
    EXPECT_EQ(tensor_2d.raw_data(), contiguous_raw_data(tv));
    EXPECT_NE(tensor_2d.raw_data(), nullptr);
  }
}

TYPED_TEST(TensorListSuite, EmptyTensorListWithDimAsTensorAccess) {
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Contiguous);
  tv.set_type(DALI_INT32);
  EXPECT_TRUE(tv.IsContiguousInMemory());
  EXPECT_TRUE(tv.IsDenseTensor());

  auto shape_1d = TensorShape<>{0};
  auto shape_2d = TensorShape<>{0, 0};

  {
    // Cannot access empty batch as tensor if we don't have positive dimensionality
    // so we can produce 0-shape of correct dim.
    EXPECT_THROW(tv.AsTensor(), std::runtime_error);
    tv.set_sample_dim(0);
    EXPECT_THROW(tv.AsTensor(), std::runtime_error);
    tv.set_sample_dim(1);
    auto tensor_1d = tv.AsTensor();
    EXPECT_EQ(tensor_1d.shape(), shape_1d);
    EXPECT_EQ(tensor_1d.type(), DALI_INT32);
    EXPECT_EQ(tensor_1d.raw_data(), contiguous_raw_data(tv));
    EXPECT_EQ(tensor_1d.raw_data(), nullptr);

    tv.set_sample_dim(2);
    auto tensor_2d = tv.AsTensor();
    EXPECT_EQ(tensor_2d.shape(), shape_2d);
    EXPECT_EQ(tensor_2d.type(), DALI_INT32);
    EXPECT_EQ(tensor_2d.raw_data(), contiguous_raw_data(tv));
    EXPECT_EQ(tensor_2d.raw_data(), nullptr);
  }
}


TYPED_TEST(TensorListSuite, BatchResize) {
  TensorList<TypeParam> tv(5);
  ASSERT_EQ(tv.num_samples(), 5);
  tv.reserve(100);
  tv.reserve(200);
  tv.template set_type<int32_t>();
  tv.Resize(uniform_list_shape(5, {10, 20}));
}

TYPED_TEST(TensorListSuite, VariableBatchResizeDown) {
  TensorList<TypeParam> tv(32);
  ASSERT_EQ(tv.num_samples(), 32);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size, DALI_UINT8);
  ASSERT_EQ(tv.num_samples(), new_size.num_samples());
}

TYPED_TEST(TensorListSuite, VariableBatchResizeUp) {
  TensorList<TypeParam> tv(2);
  ASSERT_EQ(tv.num_samples(), 2);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size, DALI_UINT8);
  ASSERT_EQ(tv.num_samples(), new_size.num_samples());
}

TYPED_TEST(TensorListSuite, EmptyShareContiguous) {
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Contiguous);
  TensorListShape<> shape = {{100, 0, 0}, {42, 0, 0}};
  tv.Resize(shape, DALI_UINT8);
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(tv.raw_tensor(i), nullptr);
  }

  TensorList<TypeParam> target;
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

TYPED_TEST(TensorListSuite, EmptyShareNonContiguous) {
  TensorList<TypeParam> tv;
  tv.SetContiguity(BatchContiguity::Noncontiguous);
  TensorListShape<> shape = {{100, 0, 0}, {42, 0, 0}};
  tv.Resize(shape, DALI_UINT8);
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(tv.raw_tensor(i), nullptr);
  }

  TensorList<TypeParam> target;
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
  TensorList<Backend> tv;
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

TYPED_TEST(TensorListSuite, MoveConstructorMetaData) {
  test_moving_props<TypeParam>(true, "XYZ",
                               {{42, 1, 2}, {1, 2, 42}, {2, 42, 1}, {1, 42, 1}, {2, 42, 2}}, 3,
                               DALI_UINT16, [](auto &&check, auto &tv) {
                                 TensorList<TypeParam> moved{std::move(tv)};
                                 check(moved);
                               });
}

TYPED_TEST(TensorListSuite, MoveAssignmentMetaData) {
  test_moving_props<TypeParam>(true, "XYZ",
                               {{42, 1, 2}, {1, 2, 42}, {2, 42, 1}, {1, 42, 1}, {2, 42, 2}}, 3,
                               DALI_UINT16, [](auto &&check, auto &tv) {
                                 TensorList<TypeParam> moved(2);
                                 moved = std::move(tv);
                                 check(moved);
                               });
}

TYPED_TEST(TensorListSuite, DeviceIdPropagationMultiGPU) {
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
    TensorList<TypeParam> batch;
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
    batch.SetSample(0, ptr, shape.num_elements() * sizeof(uint8_t), is_pinned, shape, DALI_UINT8,
                    device_id, order);
    ASSERT_EQ(batch.device_id(), device_id);
    ASSERT_EQ(batch.order().device_id(), AccessOrder::host().device_id());
    ASSERT_NE(batch.order().device_id(), batch.device_id());
  }
}

namespace {

/**
 * GTest predicate formatter. Compares a batch of data contained in TensorList or TensorList
 * @tparam T TensorList<CPUBackend> or TensorList<CPUBackend>
 * @tparam U TensorList<CPUBackend> or TensorList<CPUBackend>
 */
template <typename T, typename U>
::testing::AssertionResult Compare(const char *rhs_expr, const char *lhs_expr, const T &rhs,
                                   const U &lhs) {
  static_assert(std::is_same<T, TensorList<CPUBackend>>::value ||
                    std::is_same<T, TensorList<CPUBackend>>::value,
                "T must be either TensorList<CPUBackend> or TensorList<CPUBackend>");
  static_assert(std::is_same<U, TensorList<CPUBackend>>::value ||
                    std::is_same<U, TensorList<CPUBackend>>::value,
                "U must be either TensorList<CPUBackend> or TensorList<CPUBackend>");
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

class TensorListVariableBatchSizeTest : public ::testing::Test {
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
  TensorList<CPUBackend> test_tv_;
};

TEST_F(TensorListVariableBatchSizeTest, SelfTest) {
  for (int i = 0; i < shape_.num_samples(); i++) {
    EXPECT_EQ(test_tv_.tensor_shape(i), shape_[i]);
    EXPECT_EQ(test_tl_.tensor_shape(i), shape_[i]);
  }
  EXPECT_PRED_FORMAT2(Compare, test_tv_, test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, test_tl_);
}

TEST_F(TensorListVariableBatchSizeTest, TvShareWithResizeUp) {
  TensorList<CPUBackend> tv(2);
  tv.ShareData(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, TvShareWithResizeDown) {
  TensorList<CPUBackend> tv(32);
  tv.ShareData(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, TlShareWithResizeUp) {
  TensorList<CPUBackend> tv(2);
  tv.ShareData(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, TlShareWithResizeDown) {
  TensorList<CPUBackend> tv(32);
  tv.ShareData(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, TvCopyWithResizeUp) {
  TensorList<CPUBackend> tv(2);
  tv.Copy(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, TvCopyWithResizeDown) {
  TensorList<CPUBackend> tv(32);
  tv.Copy(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, TlCopyWithResizeUp) {
  TensorList<CPUBackend> tv(2);
  tv.Copy(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, TlCopyWithResizeDown) {
  TensorList<CPUBackend> tv(32);
  tv.Copy(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorListVariableBatchSizeTest, UpdatePropertiesFromSamples) {
  TensorList<CPUBackend> tv(3);
  tv.set_type(DALI_FLOAT);
  auto &t = tv.tensor_handle(0);
  t.set_pinned(true);
  t.Resize(19, DALI_FLOAT);
  tv.UpdatePropertiesFromSamples(false);
  // check if after calling UpdatePropertiesFromSamples even empty samples in TL have consistent
  // metadata with the the whole TL
  tv.SetSample(1, tv.tensor_handle(2));
}

TEST(TensorList, ResizeOverheadPerf) {
  (void)cudaFree(0);
#ifdef DALI_DEBUG
  int niter = 2000;
  int warmup = 500;
#else
  int niter = 20000;
  int warmup = 5000;
#endif
  int total_size = 256 << 10;
  int nsamples = 1024;
  auto shape = uniform_list_shape(nsamples, {total_size / nsamples});
  for (int i = 0; i < warmup; i++) {
    TensorList<CPUBackend> tl;
    tl.set_pinned(false);
    tl.Resize(shape, DALI_UINT8);
  }
  auto start = perf_timer::now();
  for (int i = 0; i < niter; i++) {
    TensorList<CPUBackend> tl;
    tl.set_pinned(false);
    tl.Resize(shape, DALI_UINT8);
  }
  auto end = perf_timer::now();
  std::cout << format_time((end - start) / niter) << std::endl;
}

}  // namespace test
}  // namespace dali
