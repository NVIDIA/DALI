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
#include "dali/pipeline/data/tensor_vector.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/dali_test.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace test {

/***************************************************************************************************
 The section below is copied from tensor_list_test.cc with the only changes being:
 * Rename all occurrences of TensorList to TensorVector
 * Remove APIs that will be not ported: tensor_offset
 * Tests were generalized over contiguous/noncontiguous option (see the pairs in
   TensorVectorBackendContiguous)
 * SetContiguity(true) was added in few places, as there is a difference in defaults between
   current TensorVector and TensorList.
 **************************************************************************************************/

template <typename TypeParam>
class TensorVectorTest : public DALITest {
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
   * Initialize & check a TensorVector based on an input shape
   * Allocate it as float
   */
  void SetupTensorVector(TensorVector<Backend> *tensor_list,
                         const TensorListShape<>& shape,
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
using TensorVectorBackendContiguous = ::testing::Types<
    std::pair<CPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Contiguous>>,
    std::pair<CPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Noncontiguous>>,
    std::pair<GPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Contiguous>>,
    std::pair<GPUBackend, std::integral_constant<BatchContiguity, BatchContiguity::Noncontiguous>>>;

TYPED_TEST_SUITE(TensorVectorTest, TensorVectorBackendContiguous);

// Note: A TensorVector in a valid state has a type. To get to a valid state, we
// can either set:
// type -> shape : setting shape triggers allocation
// shape & type : Resize triggers allocation
//
// Additionally, `reserve` can be called at any point.
//
// The following tests attempt to verify the correct behavior for all of
// these cases

TYPED_TEST(TensorVectorTest, TestGetTypeSizeBytes) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
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
  ASSERT_EQ(tl.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(tl.type()));

  tl.reserve(shape.num_elements() * sizeof(float));

  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_NE(tl.raw_tensor(i), nullptr);
  }
}

TYPED_TEST(TensorVectorTest, TestReserveResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;

  auto shape = this->GetRandShape();
  tl.reserve(shape.num_elements() * sizeof(float));  // This reserve already makes it contiguous
  // Can't change the pinned status after allocation happened (reserve)
  ASSERT_THROW(tl.set_pinned(true), std::runtime_error);

  ASSERT_TRUE(tl.has_data());
  ASSERT_EQ(tl.capacity(), shape.num_elements() * sizeof(float));
  ASSERT_EQ(tl.nbytes(), 0);
  ASSERT_EQ(tl._num_elements(), 0);
  ASSERT_NE(unsafe_raw_data(tl), nullptr);

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
  ASSERT_EQ(tl.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(tl.type()));


  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_NE(tl.raw_tensor(i), nullptr);
  }
}

TYPED_TEST(TensorVectorTest, TestResizeWithoutType) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  // Give the tensor a size - setting shape on non-typed TL is invalid and results in an error
  auto shape = this->GetRandShape();
  ASSERT_THROW(tl.Resize(shape), std::runtime_error);
}

TYPED_TEST(TensorVectorTest, TestSetNoType) {
  // After type is set we cannot revert to DALI_NO_TYPE
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  tl.set_type(DALI_FLOAT);
  ASSERT_THROW(tl.set_type(DALI_NO_TYPE), std::runtime_error);

  auto shape = this->GetRandShape();
  ASSERT_THROW(tl.Resize(shape, DALI_NO_TYPE), std::runtime_error);
}

TYPED_TEST(TensorVectorTest, TestGetContiguousPointer) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
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
  ASSERT_NE(unsafe_raw_data(tl), nullptr);
}

TYPED_TEST(TensorVectorTest, TestGetBytesThenAccess) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
  tl.SetContiguity(this->kContiguity);
  TensorVector<Backend> sharers[2];
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
    ASSERT_EQ(tl.nbytes(), size*sizeof(float));
    ASSERT_EQ(tl.type(), sharer.type());
    ASSERT_EQ(tl.num_samples(), num_tensor);
    ASSERT_TRUE(tl.shares_data());

    // Access can't change the underlying data type (which can happen only through Resize)
    ASSERT_THROW(tl.template mutable_tensor<int16>(0), std::runtime_error);
    ASSERT_THROW(tl.template mutable_tensor<double>(0), std::runtime_error);
    // We also cannot allocate bigger
    ASSERT_THROW(tl.Resize(tl.shape(), DALI_FLOAT64), std::runtime_error);
  }
}

TYPED_TEST(TensorVectorTest, TestZeroSizeResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  TensorListShape<> shape;
  tensor_list.template set_type<float>();
  tensor_list.Resize(shape);

  ASSERT_FALSE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), 0);
  ASSERT_EQ(tensor_list._num_elements(), 0);
  ASSERT_FALSE(tensor_list.shares_data());
}

TYPED_TEST(TensorVectorTest, TestMultipleZeroSizeResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  int num_tensor = this->RandInt(0, 128);
  auto shape = uniform_list_shape(num_tensor, TensorShape<>{ 0 });
  tensor_list.Resize(shape, DALI_FLOAT);

  ASSERT_FALSE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), 0);
  ASSERT_EQ(tensor_list.num_samples(), num_tensor);
  ASSERT_EQ(tensor_list._num_elements(), 0);
  ASSERT_FALSE(tensor_list.shares_data());

  ASSERT_EQ(tensor_list.num_samples(), num_tensor);
  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tensor_list.template tensor<float>(i), nullptr);
    ASSERT_EQ(tensor_list.tensor_shape(i), TensorShape<>{ 0 });
  }
}

TYPED_TEST(TensorVectorTest, TestFakeScalarResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  int num_scalar = this->RandInt(1, 128);
  auto shape = uniform_list_shape(num_scalar, {1});  // {1} on purpose
  tensor_list.template set_type<float>();
  tensor_list.Resize(shape);

  ASSERT_TRUE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), num_scalar*sizeof(float));
  ASSERT_EQ(tensor_list._num_elements(), num_scalar);
  ASSERT_FALSE(tensor_list.shares_data());

  for (int i = 0; i < num_scalar; ++i) {
    ASSERT_NE(tensor_list.raw_tensor(i), nullptr);
    ASSERT_EQ(tensor_list.tensor_shape(i), TensorShape<>{1});  // {1} on purpose
  }
}

TYPED_TEST(TensorVectorTest, TestTrueScalarResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  int num_scalar = this->RandInt(1, 128);
  auto shape = uniform_list_shape(num_scalar, TensorShape<>{});
  tensor_list.template set_type<float>();
  tensor_list.Resize(shape);

  ASSERT_TRUE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), num_scalar*sizeof(float));
  ASSERT_EQ(tensor_list._num_elements(), num_scalar);
  ASSERT_FALSE(tensor_list.shares_data());

  for (int i = 0; i < num_scalar; ++i) {
    ASSERT_NE(tensor_list.raw_tensor(i), nullptr);
    ASSERT_EQ(tensor_list.tensor_shape(i), TensorShape<>{});
  }
}

TYPED_TEST(TensorVectorTest, TestResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  // resize + check called in SetupTensorVector
  this->SetupTensorVector(&tensor_list, shape, &offsets);
}

TYPED_TEST(TensorVectorTest, TestMultipleResize) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
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

TYPED_TEST(TensorVectorTest, TestCopy) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  tl.template set_type<float>();

  auto shape = this->GetRandShape();
  tl.Resize(shape);

  for (int i = 0; i < shape.num_samples(); i++) {
    tl.SetSourceInfo(i, to_string(i));
  }
  tl.SetLayout(std::string(shape.sample_dim(), 'X'));

  TensorVector<Backend> tl2s[2];
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

TYPED_TEST(TensorVectorTest, TestCopyEmpty) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
  tl.SetContiguity(this->kContiguity);

  tl.template set_type<float>();
  tl.SetLayout("XX");

  TensorVector<Backend> tl2s[2];
  tl2s[0].SetContiguity(this->kContiguity);
  tl2s[1].SetContiguity(this->inverse(this->kContiguity));
  for (auto &tl2 : tl2s) {
    tl2.Copy(tl);
    ASSERT_EQ(tl.num_samples(), tl2.num_samples());
    ASSERT_EQ(tl.type(), tl2.type());
    ASSERT_EQ(tl._num_elements(), tl2._num_elements());
    ASSERT_EQ(tl.GetLayout(), tl2.GetLayout());
  }
}

TYPED_TEST(TensorVectorTest, TestTypeChangeError) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
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

TYPED_TEST(TensorVectorTest, TestTypeChange) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorVector(&tensor_list, shape, &offsets);

  DALIDataType initial_type = DALI_FLOAT;
  std::array<DALIDataType, 4> types = {DALI_FLOAT, DALI_INT32, DALI_UINT8, DALI_FLOAT64};
  const auto *base_ptr =
      this->kContiguity == BatchContiguity::Contiguous ? unsafe_raw_data(tensor_list) : nullptr;
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
        ASSERT_EQ(unsafe_raw_data(tensor_list), base_ptr);
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

TYPED_TEST(TensorVectorTest, DeviceIdPropagationMultiGPU) {
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
    TensorVector<Backend> batch;
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

TYPED_TEST(TensorVectorTest, TestShareData) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tensor_list;
  tensor_list.SetContiguity(this->kContiguity);

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorVector(&tensor_list, shape, &offsets);

  // Create a new tensor_list w/ a smaller data type
  TensorVector<Backend> tensor_lists[2];
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

}  // namespace test
}  // namespace dali
