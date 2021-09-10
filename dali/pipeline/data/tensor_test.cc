// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_vector.h"

#include <gtest/gtest.h>

#include <numeric>

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/test/dali_test.h"
#include "dali/core/mm/malloc_resource.h"

namespace dali {

inline TensorShape<> empty_tensor_shape() {
  return { 0 };
}

template <typename Backend>
class TensorTest : public DALITest {
 public:
  TensorListShape<> GetRandShapeList() {
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
    return shape;
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

// Sharing data from a raw pointer resets a Tensor to
// and invalid state (no type). To get to a valid state
// we can aquire a type and size in the following orders:
//
// type -> size (bytes) : getting size triggers allocation
// size -> type (bytes) : getting type triggers allocation
// bytes -> type -> size : shares data, gets type (zero size), gets size
// (may or may not allocate)
// bytes -> size -> type : shares data, gets size (no type), gets type
// (may or may not allocate)
//
// The following tests attempt to verify the correct behavior for
// all of these cases

TYPED_TEST(TensorTest, TestGetTypeSizeBytes) {
  Tensor<TypeParam> t;

  // Give the tensor a type
  t.template mutable_data<float>();

  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.raw_data(), nullptr);

  // Give the tensor a size. This
  // should trigger an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  t.Resize(shape);

  // Validate the internals
  ASSERT_NE(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(t.type()));
}

TYPED_TEST(TensorTest, TestGetSizeTypeBytes) {
  Tensor<TypeParam> t;

  // Give the tensor a size
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), 0);

  // Give the tensor a type. This should
  // trigger an allocation
  t.template mutable_data<float>();

  // Validate the internals
  ASSERT_NE(t.raw_data(), nullptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_EQ(t.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(t.type()));
}

TYPED_TEST(TensorTest, TestGetBytesTypeSizeNoAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  std::vector<float> source_data(size);

  // Wrap the allocation
  t.ShareData(source_data.data(), size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.template mutable_data<int16>();

  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Kind of exception safety test
  ASSERT_EQ(t.raw_data(), source_data.data());
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.set_type(TypeTable::GetTypeInfoFromStatic<float>());
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

TYPED_TEST(TensorTest, TestGetBytesTypeSizeAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  std::vector<float> source_data(size);

  // Wrap the allocation
  t.ShareData(source_data.data(), size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  t.template mutable_data<double>();

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

  t.set_type(TypeTable::GetTypeInfoFromStatic<float>());
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

TYPED_TEST(TensorTest, TestGetBytesSizeTypeNoAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  std::vector<float> source_data(size);

  // Wrap the allocation
  t.ShareData(source_data.data(), size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  ASSERT_THROW(t.set_type(TypeTable::GetTypeInfoFromStatic<int16>()), std::runtime_error);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  t.set_type(TypeTable::GetTypeInfoFromStatic<float>());

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
  t.ShareData(source_data.data(), size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), empty_tensor_shape());
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  ASSERT_THROW(t.set_type(TypeTable::GetTypeInfoFromStatic<double>()), std::runtime_error);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());


  t.set_type(TypeTable::GetTypeInfoFromStatic<float>());

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
  tl.Resize(shape);
  tl.set_type(TypeTable::GetTypeInfoFromStatic<float>());

  // Create a tensor and wrap each tensor from the list
  Tensor<TypeParam> tensor;
  int num_tensor = tl.ntensor();
  for (int i = 0; i < num_tensor; ++i) {
    tensor.ShareData(&tl, i);

    // Verify the internals
    ASSERT_TRUE(tensor.shares_data());
    ASSERT_EQ(tensor.raw_data(), tl.raw_tensor(i));
    ASSERT_EQ(tensor.type(), tl.type());
    ASSERT_EQ(tensor.shape(), tl.tensor_shape(i));

    Index size = volume(tl.tensor_shape(i));
    ASSERT_EQ(tensor.size(), size);
    ASSERT_EQ(tensor.nbytes(), size*sizeof(float));
  }
}

TYPED_TEST(TensorTest, TestCopyToTensorList) {
  TensorVector<TypeParam> tensors(16);
  for (auto& t : tensors) {
    auto shape = this->GetRandShape(4, 4);
    t->Resize(shape);
    t->template mutable_data<float>();
  }

  TensorList<TypeParam> tl;
  tl.Copy(tensors, 0);

  int num_tensor = tl.ntensor();
  ASSERT_EQ(num_tensor, tensors.size());
  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tensors[i].type(), tl.type());
    ASSERT_EQ(tensors[i].shape(), tl.tensor_shape(i));
    Index size = volume(tl.tensor_shape(i));
    ASSERT_EQ(tensors[i].size(), size);
    ASSERT_EQ(tensors[i].nbytes(), size*sizeof(float));
  }
}

TYPED_TEST(TensorTest, TestCopyEmptyToTensorList) {
  TensorVector<TypeParam> tensors(16);
  // Empty tensors
  TensorList<TypeParam> tl;
  tl.set_type(TypeTable::GetTypeInfoFromStatic<float>());
  tl.Copy(tensors, 0);

  Tensor<TypeParam> tensor;
  int num_tensor = tl.ntensor();
  ASSERT_EQ(num_tensor, tensors.size());
  const auto &shape = tl.shape();
  Index total_volume = shape.num_elements();
  ASSERT_EQ(total_volume, 0);
}

TYPED_TEST(TensorTest, TestResize) {
  Tensor<TypeParam> tensor;

  // Get shape
  auto shape = this->GetRandShape();
  tensor.Resize(shape);

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
  std::function<void*(size_t)> alloc_func;
  std::function<void(void *)> deleter;
  if (std::is_same<TypeParam, CPUBackend>::value) {
    alloc_func = [&](size_t bytes) {
      return new uint8_t[bytes];
    };
    deleter = [&](void *ptr) {
      free(ptr);
      allocations--;
    };
  } else {
    alloc_func = [](size_t bytes) {
      void *ptr;
      CUDA_CALL(cudaMalloc(&ptr, bytes));
      return ptr;
    };
    deleter = [&](void *ptr) {
      CUDA_DTOR_CALL(cudaFree(ptr));
      allocations--;
    };
  }

  tensor.set_alloc_func([&](size_t bytes) {
    allocations++;
    return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(alloc_func(bytes)), deleter);
  });

  tensor.Resize(shape);

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
  tensor1.Copy(vec, 0);
  ASSERT_NE(tensor1.template mutable_data<float>(), nullptr);
  ASSERT_EQ(vec.size(), tensor1.size());
  ASSERT_EQ(vec.size() * sizeof(float), tensor1.nbytes());
  ASSERT_EQ(1, tensor1.ndim());

  auto tensor1_data = to_vec(tensor1);
  EXPECT_EQ(0, std::memcmp(vec.data(), tensor1_data.data(), vec.size() * sizeof(float)));

  Tensor<TypeParam> tensor2;
  tensor2.Copy(make_span(vec), 0);
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
    tensor.Resize(shape);

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
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestResizeScalar) {
  Tensor<TypeParam> tensor;

  // Get shape
  TensorShape<> shape = { 1 };
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestResizeZeroSize) {
  Tensor<TypeParam> tensor;

  // Get shape
  TensorShape<> shape = { 0 };
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_EQ(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestTypeChange) {
  Tensor<TypeParam> tensor;

  // Get shape
  TensorShape<> shape = { 4, 480, 640, 3 };
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  size_t num_elements = volume(shape);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }
  ASSERT_EQ(num_elements * sizeof(float), tensor.nbytes());

  // Save the pointer
  const void *source_data = tensor.raw_data();

  // Change the type of the buffer
  tensor.template mutable_data<int>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }

  // No re-allocation should have occured
  ASSERT_EQ(source_data, tensor.raw_data());
  ASSERT_EQ(num_elements * sizeof(int), tensor.nbytes());

  // Change the type to a smaller type
  tensor.template mutable_data<uint8>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }

  // No re-allocation should have occured
  ASSERT_EQ(source_data, tensor.raw_data());
  ASSERT_EQ(num_elements * sizeof(uint8), tensor.nbytes());

  // Change the type to a larger type
  tensor.template mutable_data<double>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }

  ASSERT_EQ(num_elements * sizeof(double), tensor.nbytes());
}

TYPED_TEST(TensorTest, TestSubspaceTensor) {
  // Insufficient dimensions
  {
    Tensor<TypeParam> empty_tensor;
    TensorShape<> empty_shape = {};
    empty_tensor.Resize(empty_shape);
    empty_tensor.set_type(TypeTable::GetTypeInfoFromStatic<uint8_t>());
    ASSERT_ANY_THROW(empty_tensor.SubspaceTensor(0));
  }
  {
    Tensor<TypeParam> one_dim_tensor;
    TensorShape<> one_dim_shape = {42};
    one_dim_tensor.Resize(one_dim_shape);
    one_dim_tensor.set_type(TypeTable::GetTypeInfoFromStatic<uint8_t>());
    ASSERT_ANY_THROW(one_dim_tensor.SubspaceTensor(0));
  }

  // Wrong subspace
  {
    Tensor<TypeParam> tensor;
    auto shape = this->GetRandShape(2, 6);
    tensor.Resize(shape);
    tensor.set_type(TypeTable::GetTypeInfoFromStatic<uint8_t>());
    ASSERT_ANY_THROW(tensor.SubspaceTensor(-1));
    ASSERT_ANY_THROW(tensor.SubspaceTensor(shape[0]));
    ASSERT_ANY_THROW(tensor.SubspaceTensor(shape[0] + 1));
  }

  // Valid subspace:
  {
    Tensor<TypeParam> tensor;
    auto shape = this->GetRandShape(2, 6);
    tensor.Resize(shape);
    tensor.set_type(TypeTable::GetTypeInfoFromStatic<uint8_t>());
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

}  // namespace dali
