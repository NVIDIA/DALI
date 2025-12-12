// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <numeric>
#include <stdexcept>

#include "dali/core/common.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/types.h"
#include "dali/test/dali_test.h"
#include "dali/core/static_switch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/pipeline/data/views.h"

namespace dali {

inline TensorShape<> empty_tensor_shape() {
  return { 0 };
}

template <typename Backend>
class TensorTest : public DALITest {
 public:
  TensorListShape<> GetRandShapeList(int64_t min_elements = 1_u64 << 16,
                                     int64_t max_elements = 1_u64 << 28) {
    for (;;) {
      int num_tensor = this->RandInt(1, 128);
      int dims = this->RandInt(2, 3);
      TensorListShape<> shape(num_tensor, dims);
      for (int i = 0; i < num_tensor; ++i) {
        TensorShape<> tensor_shape;
        tensor_shape.resize(dims);
        for (int j = 0; j < dims; ++j) {
          tensor_shape[j] = this->RandInt(1, 512);
        }
        shape.set_tensor_shape(i, tensor_shape);
      }
      int64_t n = shape.num_elements();
      if (n >= min_elements && n <= max_elements)
        return shape;
    }
  }



  TensorShape<> GetRandShape(int dim_start = 1, int dim_end = 5) {
    int dims = this->RandInt(dim_start, dim_end);
    vector<Index> shape(dims, 0);
    for (auto &val : shape) {
      val = this->RandInt(1, 32);
    }
    return shape;
  }
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_SUITE(TensorTest, Backends);

TYPED_TEST(TensorTest, Move) {
  Tensor<TypeParam> t;

  // Give the tensor a type
  t.template set_type<float>();
  auto shape = this->GetRandShape();
  t.Resize(shape);
  t.SetSourceInfo("test");

  Tensor<TypeParam> target_move_assign;
  target_move_assign = std::move(t);

  ASSERT_TRUE(target_move_assign.has_data());
  ASSERT_NE(target_move_assign.raw_data(), nullptr);
  ASSERT_EQ(target_move_assign.size(), volume(shape));
  ASSERT_EQ(target_move_assign.shape(), shape);
  ASSERT_TRUE(IsType<float>(target_move_assign.type()));
  ASSERT_EQ(target_move_assign.GetSourceInfo(), "test");


  Tensor<TypeParam> target_move_construct(std::move(target_move_assign));
  ASSERT_TRUE(target_move_construct.has_data());
  ASSERT_NE(target_move_construct.raw_data(), nullptr);
  ASSERT_EQ(target_move_construct.size(), volume(shape));
  ASSERT_EQ(target_move_construct.shape(), shape);
  ASSERT_TRUE(IsType<float>(target_move_construct.type()));
  ASSERT_EQ(target_move_construct.GetSourceInfo(), "test");
}

// Sharing data from a raw pointer resets a Tensor to
// and invalid state (no type). To get to a valid state
// we can acquire a type and size in the following orders:
//
// To get to a valid state, we can either set:
// type -> shape : setting shape triggers allocation
// shape & type : Resize triggers allocation
//
// Additionally, `reserve` can be called at any point.
//
// The following tests attempt to verify the correct behavior for
// all of these cases

TYPED_TEST(TensorTest, TestGetTypeSizeBytes) {
  Tensor<TypeParam> t;

  // Give the tensor a type
  t.template set_type<float>();

  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.raw_data(), nullptr);

  // Give the tensor a size. This
  // should trigger an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  t.Resize(shape);

  ASSERT_TRUE(t.has_data());
  ASSERT_NE(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(t.type()));

  t.reserve(shape.num_elements() * sizeof(float));

  ASSERT_NE(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(t.type()));
}

TYPED_TEST(TensorTest, TestReserveResize) {
  Tensor<TypeParam> t;
  // Give the tensor a size. This
  // should trigger an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  t.reserve(size * sizeof(float));
  ASSERT_THROW(t.set_pinned(true), std::runtime_error);

  ASSERT_TRUE(t.has_data());
  ASSERT_EQ(t.capacity(), size * sizeof(float));
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_NE(t.raw_data(), nullptr);

  // Give the tensor a type
  t.template set_type<float>();

  ASSERT_TRUE(t.has_data());
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_NE(t.raw_data(), nullptr);

  t.Resize(shape);

  ASSERT_TRUE(t.has_data());
  ASSERT_NE(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), size * sizeof(float));
  ASSERT_TRUE(IsType<float>(t.type()));
}

TYPED_TEST(TensorTest, TestResizeWithoutType) {
  Tensor<TypeParam> t;

  // Give the tensor a size - setting shape on non-typed TL is invalid and results in an error
  auto shape = this->GetRandShape();
  ASSERT_THROW(t.Resize(shape), std::runtime_error);
}

TYPED_TEST(TensorTest, TestGetBytesTypeSizeNoAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  std::vector<float> source_data(size);

  // Wrap the allocation
  t.ShareData(source_data.data(), size*sizeof(float), false, DALI_NO_TYPE, CPU_ONLY_DEVICE_ID);

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.template set_type<float>();

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<float>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.template set_type<float>();
  t.Resize(shape);

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<float>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.Resize(shape, DALI_INT16);

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(int16_t));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<int16_t>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.Reset();

  ASSERT_EQ(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_FALSE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesTypeSizeAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  std::vector<float> source_data(size);

  // Wrap the allocation
  t.ShareData(source_data.data(), size*sizeof(float), false, DALI_NO_TYPE, CPU_ONLY_DEVICE_ID);

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  t.template set_type<double>();

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size, type is bigger so it will throw
  ASSERT_THROW(t.Resize(shape), std::runtime_error);

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // We still can Resize to this allocation
  t.Resize(shape, DALI_FLOAT);

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<float>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.Reset();
  ASSERT_EQ(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_FALSE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesSizeTypeNoAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  std::vector<float> source_data(size);

  // Wrap the allocation
  t.ShareData(source_data.data(), size*sizeof(float), false, DALI_NO_TYPE, CPU_ONLY_DEVICE_ID);

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());


  ASSERT_THROW(t.Resize(shape), std::runtime_error);

  // Give the Tensor a type
  t.template set_type<float>();
  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<float>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.Reset();
  ASSERT_EQ(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_FALSE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesSizeTypeAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  std::vector<float> source_data(size);

  // Wrap the allocation
  t.ShareData(source_data.data(), size*sizeof(float), false, DALI_NO_TYPE, CPU_ONLY_DEVICE_ID);

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  t.template set_type<float>();
  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_THROW((t.template set_type<double>()), std::runtime_error);

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<float>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.Reset();
  ASSERT_EQ(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_FALSE(t.shares_data());
}


// Sharing data from a TensorList puts a Tensor into
// a valid state (it has a type), as we enforce that
// the input TensorList has a valid type.
TYPED_TEST(TensorTest, TestShareData) {
  TensorList<TypeParam> tl;
  auto shape = this->GetRandShapeList();
  tl.template set_type<float>();
  tl.Resize(shape);

  // Create a tensor and wrap each tensor from the list
  Tensor<TypeParam> tensor;
  int num_tensor = tl.num_samples();
  for (int i = 0; i < num_tensor; ++i) {
    // TODO(klecki): Rework this with proper sample-based tensor batch data structure
    auto sample_shared_ptr = unsafe_sample_owner(tl, i);
    tensor.ShareData(sample_shared_ptr, tl.capacity(), tl.is_pinned(), tl.shape()[i],
                     tl.type(), tl.device_id());
    tensor.SetMeta(tl.GetMeta(i));

    // Verify the internals
    ASSERT_TRUE(tensor.shares_data());
    ASSERT_EQ(tensor.raw_data(), tl.raw_tensor(i));
    ASSERT_EQ(tensor.type(), tl.type());
    ASSERT_EQ(tensor.shape(), tl.tensor_shape(i));
    ASSERT_EQ(tensor.device_id(), tl.device_id());

    Index size = volume(tl.tensor_shape(i));
    ASSERT_EQ(tensor.size(), size);
    ASSERT_EQ(tensor.nbytes(), size*sizeof(float));
  }
}

TYPED_TEST(TensorTest, DeviceIdPropagationMultiGPU) {
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
    Tensor<TypeParam> tensor;
    tensor.set_order(order);
    DeviceGuard dg(device_id);
    void *data_ptr;
    std::shared_ptr<void> ptr;
    if (is_device) {
      CUDA_CALL(cudaMalloc(&data_ptr, shape.num_elements() * sizeof(uint8_t)));
      ptr = std::shared_ptr<void>(data_ptr, [](void *ptr) { cudaFree(ptr); });
    } else {
      CUDA_CALL(cudaMallocHost(&data_ptr, shape.num_elements() * sizeof(uint8_t)));
      ptr = std::shared_ptr<void>(data_ptr, [](void *ptr) { cudaFreeHost(ptr); });
    }
    tensor.ShareData(ptr, shape.num_elements() * sizeof(uint8_t), is_pinned, shape, DALI_UINT8,
                     device_id, order);
    ASSERT_EQ(tensor.device_id(), device_id);
    ASSERT_EQ(tensor.order().device_id(), AccessOrder::host().device_id());
    ASSERT_NE(tensor.order().device_id(), tensor.device_id());
  }
}

TYPED_TEST(TensorTest, TestCopyToTensorList) {
  TensorList<TypeParam> tensors(16);
  TensorListShape<4> shape(16);
  for (int i = 0; i < 16; i++) {
    shape.set_tensor_shape(i, this->GetRandShape(4, 4));
  }
  tensors.Resize(shape, DALI_FLOAT);

  TensorList<TypeParam> tl;
  tl.Copy(tensors);

  int num_tensor = tl.num_samples();
  ASSERT_EQ(num_tensor, tensors.num_samples());
  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tensors[i].type(), tl.type());
    ASSERT_EQ(tensors[i].shape(), tl.tensor_shape(i));
    Index size = volume(tl.tensor_shape(i));
    ASSERT_EQ(tensors[i].shape().num_elements(), size);
    ASSERT_EQ(tensors[i].shape().num_elements() * tensors.type_info().size(), size*sizeof(float));
  }
}

TYPED_TEST(TensorTest, TestCopyEmptyToTensorList) {
  TensorList<TypeParam> tensors(16);
  // Empty tensors
  TensorList<TypeParam> tl;
  tensors.template set_type<float>();
  tl.Copy(tensors);

  Tensor<TypeParam> tensor;
  int num_tensor = tl.num_samples();
  ASSERT_EQ(num_tensor, tensors.num_samples());
  const auto &shape = tl.shape();
  Index total_volume = shape.num_elements();
  ASSERT_EQ(total_volume, 0);
}

TYPED_TEST(TensorTest, TestResize) {
  Tensor<TypeParam> tensor;

  // Get shape
  auto shape = this->GetRandShape();
  tensor.Resize(shape, DALI_FLOAT);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }
}

TYPED_TEST(TensorTest, TestCustomAlloc) {
  Tensor<TypeParam> tensor;

  // Get shape
  auto shape = this->GetRandShape();
  int allocations = 0;
  std::function<uint8_t*(size_t)> alloc_func;
  std::function<void(uint8_t *)> deleter;
  if (std::is_same<TypeParam, CPUBackend>::value) {
    alloc_func = [&](size_t bytes) {
      return new uint8_t[bytes];
    };
    deleter = [&](uint8_t *ptr) {
      delete[] ptr;
      allocations--;
    };
  } else {
    alloc_func = [](size_t bytes) {
      void *ptr;
      CUDA_CALL(cudaMalloc(&ptr, bytes));
      return static_cast<uint8_t*>(ptr);
    };
    deleter = [&](uint8_t *ptr) {
      CUDA_DTOR_CALL(cudaFree(ptr));
      allocations--;
    };
  }

  tensor.set_alloc_func([&](size_t bytes) {
    allocations++;
    return std::shared_ptr<uint8_t>(alloc_func(bytes), deleter);
  });

  tensor.Resize(shape, DALI_FLOAT);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  ASSERT_EQ(allocations, 1);
  for (int i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }
  tensor.Reset();
  ASSERT_EQ(allocations, 0);
}

template <typename Backend, typename T = float>
std::vector<T> to_vec(Tensor<Backend> &tensor) {
  std::vector<T> tmp(tensor.size());
  if (std::is_same<Backend, GPUBackend>::value) {
    CUDA_CALL(
      cudaMemcpyAsync(tmp.data(), tensor.template data<T>(),
                      tensor.nbytes(), cudaMemcpyDeviceToHost, 0));
    CUDA_CALL(cudaStreamSynchronize(0));
  } else if (std::is_same<Backend, CPUBackend>::value) {
    memcpy(tmp.data(), tensor.template data<float>(), tensor.nbytes());
  }
  return tmp;
}

TYPED_TEST(TensorTest, TestCopyFromBuf) {
  std::vector<float> vec(2056);
  float num = 0.0f;
  for (auto &x : vec) {
    x = num;
    num += 1.0f / 2056;
  }

  Tensor<TypeParam> tensor1;
  tensor1.Copy(vec);
  ASSERT_NE(tensor1.template mutable_data<float>(), nullptr);
  ASSERT_EQ(vec.size(), tensor1.size());
  ASSERT_EQ(vec.size() * sizeof(float), tensor1.nbytes());
  ASSERT_EQ(1, tensor1.ndim());

  auto tensor1_data = to_vec(tensor1);
  EXPECT_EQ(0, std::memcmp(vec.data(), tensor1_data.data(), vec.size() * sizeof(float)));

  Tensor<TypeParam> tensor2;
  tensor2.Copy(make_span(vec));
  ASSERT_NE(tensor2.template mutable_data<float>(), nullptr);
  ASSERT_EQ(vec.size(), tensor2.size());
  ASSERT_EQ(1, tensor2.ndim());

  auto tensor2_data = to_vec(tensor2);
  EXPECT_EQ(0, std::memcmp(vec.data(), tensor2_data.data(), vec.size() * sizeof(float)));
}

TYPED_TEST(TensorTest, TestMultipleResize) {
  Tensor<TypeParam> tensor;

  int num = this->RandInt(2, 20);
  for (int i = 0; i < num; ++i) {
    // Get shape
    auto shape = this->GetRandShape();
    tensor.Resize(shape, DALI_FLOAT);

    // Verify the settings
    ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
    ASSERT_EQ(tensor.size(), volume(shape));
    ASSERT_EQ(tensor.ndim(), shape.size());
    for (int i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(tensor.dim(i), shape[i]);
    }
  }
}

TYPED_TEST(TensorTest, TestResizeTrueScalar) {
  Tensor<TypeParam> tensor;

  // Get shape
  TensorShape<> shape = {};
  tensor.Resize(shape, DALI_FLOAT);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestResizeScalar) {
  Tensor<TypeParam> tensor;

  // Get shape
  TensorShape<> shape = { 1 };
  tensor.Resize(shape, DALI_FLOAT);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestResizeZeroSize) {
  Tensor<TypeParam> tensor;

  // Get shape
  TensorShape<> shape = { 0 };
  tensor.Resize(shape, DALI_FLOAT);

  // Verify the settings
  ASSERT_EQ(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestTypeChangeError) {
  Tensor<TypeParam> tensor;
  TensorShape<> shape = { 200, 300, 3 };

  tensor.set_type(DALI_UINT8);
  tensor.set_type(DALI_FLOAT);
  tensor.set_type(DALI_INT32);
  tensor.Resize(shape);
  ASSERT_NE(tensor.template mutable_data<int32_t>(), nullptr);

  ASSERT_THROW(tensor.set_type(DALI_FLOAT), std::runtime_error);

  tensor.Resize(shape, DALI_FLOAT);
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
}

TYPED_TEST(TensorTest, TestTypeChange) {
  Tensor<TypeParam> tensor;

  // Get shape
  TensorShape<> shape = { 4, 480, 640, 3 };
  tensor.Resize(shape, DALI_FLOAT);

  DALIDataType current_type = DALI_FLOAT;
  std::array<DALIDataType, 4> types = {DALI_FLOAT, DALI_INT32, DALI_UINT8, DALI_FLOAT64};
  const auto *ptr = tensor.raw_data();

  for (auto new_type : types) {
    if (current_type != new_type) {
      // Simply changing the type of the buffer is not allowed
      ASSERT_THROW(tensor.set_type(new_type), std::runtime_error);
      tensor.Resize(shape, new_type);
      current_type = new_type;
    }

    // The side-effects of only reallocating when we need a bigger buffer, but we may use padding
    if (TypeTable::GetTypeInfo(current_type).size() <= sizeof(float)) {
      ASSERT_EQ(ptr, tensor.raw_data());
    }

    // Verify the settings
    ASSERT_NE(tensor.raw_data(), nullptr);
    ASSERT_EQ(tensor.shape(), shape);
    ASSERT_EQ(tensor.size(), volume(shape));
    ASSERT_EQ(tensor.ndim(), shape.size());
    ASSERT_EQ(tensor.type(), current_type);
    for (int i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(tensor.dim(i), shape[i]);
    }
    ASSERT_EQ(volume(shape) * TypeTable::GetTypeInfo(current_type).size(), tensor.nbytes());
  }
}

TYPED_TEST(TensorTest, TestReinterpret) {
  using Backend = TypeParam;
  DALIDataType types[] = {
      DALI_UINT8, DALI_INT8, DALI_UINT16, DALI_INT16, DALI_UINT32, DALI_INT32,
      DALI_UINT64, DALI_INT64, DALI_FLOAT, DALI_FLOAT16, DALI_FLOAT64, DALI_BOOL,
      DALI_INTERP_TYPE, DALI_DATA_TYPE, DALI_IMAGE_TYPE,
  };
  for (auto old_t : types) {
    auto old_size = TypeTable::GetTypeInfo(old_t).size();
    for (auto new_t : types) {
      auto new_size = TypeTable::GetTypeInfo(new_t).size();
      Tensor<Backend> t;
      t.Resize(TensorShape<>{2, 3, 4}, old_t);
      if (old_size == new_size) {
        const void *p = t.raw_data();
        EXPECT_NO_THROW(t.Reinterpret(new_t));
        EXPECT_EQ(t.type(), new_t);
        EXPECT_EQ(t.raw_data(), p);
      } else {
        EXPECT_THROW(t.Reinterpret(new_t), std::exception);
      }
    }
  }
}

TYPED_TEST(TensorTest, TestSubspaceTensor) {
  // Insufficient dimensions
  {
    Tensor<TypeParam> empty_tensor;
    TensorShape<> empty_shape = {};
    empty_tensor.Resize(empty_shape, DALI_UINT8);
    ASSERT_ANY_THROW(empty_tensor.SubspaceTensor(0));
  }
  {
    Tensor<TypeParam> one_dim_tensor;
    TensorShape<> one_dim_shape = {42};
    one_dim_tensor.Resize(one_dim_shape, DALI_UINT8);
    ASSERT_ANY_THROW(one_dim_tensor.SubspaceTensor(0));
  }

  // Wrong subspace
  {
    Tensor<TypeParam> tensor;
    auto shape = this->GetRandShape(2, 6);
    tensor.Resize(shape, DALI_UINT8);
    ASSERT_ANY_THROW(tensor.SubspaceTensor(-1));
    ASSERT_ANY_THROW(tensor.SubspaceTensor(shape[0]));
    ASSERT_ANY_THROW(tensor.SubspaceTensor(shape[0] + 1));
  }

  // Valid subspace:
  {
    Tensor<TypeParam> tensor;
    auto shape = this->GetRandShape(2, 6);
    tensor.Resize(shape, DALI_UINT8);
    int plane_size = 1;
    for (int i = 1; i < shape.size(); i++) {
      plane_size *= shape[i];
    }
    auto *base_source_data = tensor.template data<uint8_t>();
    for (Index i = 0; i < shape[0]; i++) {
      auto subspace = tensor.SubspaceTensor(i);
      ASSERT_EQ(subspace.template data<uint8_t>(), base_source_data + plane_size * i);
      ASSERT_EQ(subspace.ndim(), tensor.ndim() - 1);
      for (int j = 0; j < subspace.ndim(); j++) {
        ASSERT_EQ(subspace.dim(j), tensor.dim(j + 1));
      }
    }
  }
}

TEST(TensorTestGPU, TestCrossDeviceCopy_MultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  }

  Tensor<CPUBackend> in, out;
  Tensor<GPUBackend> gpu0;
  Tensor<GPUBackend> gpu1;
  in.Resize(TensorShape<>{10000}, DALI_INT32);
  SequentialFill(view<int>(in), 1234);

  gpu0.set_device_id(0);
  gpu1.set_device_id(1);

  {
    DeviceGuard dg0(0);
    gpu0.Copy(in);
  }

  {
    DeviceGuard dg1(1);
    gpu1.Copy(gpu0);
    out.Copy(gpu1);
  }


  Check(view<int>(in), view<int>(out));
}

}  // namespace dali
