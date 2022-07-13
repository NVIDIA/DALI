// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/test/dali_test.h"

namespace dali {

template <typename Backend>
class TensorListTest : public DALITest {
 public:
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
  void SetupTensorList(TensorList<Backend> *tensor_list,
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
      ASSERT_EQ(tensor_list->tensor_offset(i), (*offsets)[i]);
    }
  }
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_SUITE(TensorListTest, Backends);

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
  TensorList<TypeParam> tl;

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
    ASSERT_EQ(tl.tensor_offset(i), offsets[i]);
  }
}

TYPED_TEST(TensorListTest, TestReserveResize) {
  TensorList<TypeParam> tl;

  auto shape = this->GetRandShape();
  tl.reserve(shape.num_elements() * sizeof(float));
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
    ASSERT_EQ(tl.tensor_offset(i), offsets[i]);
  }
}

TYPED_TEST(TensorListTest, TestResizeWithoutType) {
  TensorList<TypeParam> tl;

  // Give the tensor a size - setting shape on non-typed TL is invalid and results in an error
  auto shape = this->GetRandShape();
  ASSERT_THROW(tl.Resize(shape), std::runtime_error);
}

TYPED_TEST(TensorListTest, TestSetNoType) {
  TensorList<TypeParam> tl;

  tl.set_type(DALI_FLOAT);
  ASSERT_THROW(tl.set_type(DALI_NO_TYPE), std::runtime_error);

  auto shape = this->GetRandShape();
  ASSERT_THROW(tl.Resize(shape, DALI_NO_TYPE), std::runtime_error);
}

TYPED_TEST(TensorListTest, TestGetContiguousPointer) {
  TensorList<TypeParam> tl;

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

TYPED_TEST(TensorListTest, TestGetBytesThenNoAlloc) {
  TensorList<TypeParam> tl, sharer;

  // Allocate the sharer
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

  // Give the buffer a type smaller than float.
  // Although we should have enough shared bytes,
  // we don't allow for a partial access to data
  ASSERT_THROW(tl.template mutable_tensor<int16>(0), std::runtime_error);
}

TYPED_TEST(TensorListTest, TestGetBytesThenAlloc) {
  TensorList<TypeParam> tl, sharer;

  // Allocate the sharer
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

  // Give the buffer a type bigger than float.
  // This normally would cause a reallocation,
  // but we that's forbidden when using shared data.
  ASSERT_THROW(tl.template mutable_tensor<double>(0), std::runtime_error);
}

TYPED_TEST(TensorListTest, TestZeroSizeResize) {
  TensorList<TypeParam> tensor_list;

  TensorListShape<> shape;
  tensor_list.template set_type<float>();
  tensor_list.Resize(shape);

  ASSERT_FALSE(tensor_list.has_data());
  ASSERT_EQ(tensor_list.nbytes(), 0);
  ASSERT_EQ(tensor_list._num_elements(), 0);
  ASSERT_FALSE(tensor_list.shares_data());
}

TYPED_TEST(TensorListTest, TestMultipleZeroSizeResize) {
  TensorList<TypeParam> tensor_list;

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
    ASSERT_EQ(tensor_list.tensor_offset(i), 0);
  }
}

TYPED_TEST(TensorListTest, TestFakeScalarResize) {
  TensorList<TypeParam> tensor_list;

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
    ASSERT_EQ(tensor_list.tensor_offset(i), i);
  }
}

TYPED_TEST(TensorListTest, TestTrueScalarResize) {
  TensorList<TypeParam> tensor_list;

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
    ASSERT_EQ(tensor_list.tensor_offset(i), i);
  }
}

TYPED_TEST(TensorListTest, TestResize) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  // resize + check called in SetupTensorList
  this->SetupTensorList(&tensor_list, shape, &offsets);
}

TYPED_TEST(TensorListTest, TestMultipleResize) {
  TensorList<TypeParam> tensor_list;

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
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }
}
TYPED_TEST(TensorListTest, TestCopy) {
  TensorList<TypeParam> tl;

  tl.template set_type<float>();

  auto shape = this->GetRandShape();
  tl.Resize(shape);

  TensorList<TypeParam> tl2;
  tl2.Copy(tl);

  ASSERT_EQ(tl.num_samples(), tl2.num_samples());
  ASSERT_EQ(tl.type(), tl2.type());
  ASSERT_EQ(tl._num_elements(), tl2._num_elements());

  for (int i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tl.tensor_shape(i), tl.tensor_shape(i));
    ASSERT_EQ(volume(tl.tensor_shape(i)), volume(tl2.tensor_shape(i)));
  }
}

TYPED_TEST(TensorListTest, TestCopyEmpty) {
  TensorList<TypeParam> tl;

  tl.template set_type<float>();

  TensorList<TypeParam> tl2;
  tl2.Copy(tl);
  ASSERT_EQ(tl.num_samples(), tl2.num_samples());
  ASSERT_EQ(tl.type(), tl2.type());
  ASSERT_EQ(tl._num_elements(), tl2._num_elements());
}

TYPED_TEST(TensorListTest, TestTypeChangeSameSize) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  // Save the pointers
  std::vector<const void *> ptrs;
  for (int i = 0; i < tensor_list.num_samples(); i++) {
    ptrs.push_back(tensor_list.raw_tensor(i));
  }
  size_t nbytes = tensor_list.nbytes();

  // Change the data type
  tensor_list.template set_type<int>();

  // Check the internals
  ASSERT_EQ(tensor_list.num_samples(), shape.size());
  for (int i = 0; i < tensor_list.num_samples(); ++i) {
    ASSERT_EQ(ptrs[i], tensor_list.raw_tensor(i));
    ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }

  // No memory allocation should have occurred
  ASSERT_EQ(nbytes, tensor_list.nbytes());
}

TYPED_TEST(TensorListTest, TestTypeChangeSmaller) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  size_t nbytes = tensor_list.nbytes();
  const auto *base_ptr = unsafe_raw_data(tensor_list);

  // Change the data type to something smaller
  tensor_list.template set_type<uint8>();

  // Check the internals
  ASSERT_EQ(tensor_list.num_samples(), shape.size());
  for (int i = 0; i < tensor_list.num_samples(); ++i) {
    ASSERT_EQ(unsafe_raw_data(tensor_list), base_ptr);
    ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }

  // nbytes should have reduced by a factor of 4
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), tensor_list.nbytes());
}

TYPED_TEST(TensorListTest, TestTypeChangeLarger) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  size_t nbytes = tensor_list.nbytes();

  // Change the data type to something larger
  tensor_list.template set_type<double>();

  // Check the internals
  ASSERT_EQ(tensor_list.num_samples(), shape.size());
  for (int i = 0; i < tensor_list.num_samples(); ++i) {
    ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }

  // nbytes should have increased by a factor of 2
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(double), tensor_list.nbytes());
}

TYPED_TEST(TensorListTest, DeviceIdPropagationMultiGPU) {
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  }
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr bool is_pinned = !is_device;
  AccessOrder order = AccessOrder::host();
  TensorListShape<> shape{{42}};
  for (int device_id = 0; device_id < num_devices; device_id++) {
    TensorList<TypeParam> batch;
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
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  auto shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  // Create a new tensor_list w/ a smaller data type
  TensorList<TypeParam> tensor_list2;

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
    ASSERT_EQ(tensor_list2.tensor_offset(i), offsets[i]);
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

}  // namespace dali
