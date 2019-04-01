// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>

#include <numeric>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/test/dali_test.h"

namespace dali {

template <typename Backend>
class TensorTest : public DALITest {
 public:
  vector<Dims> GetRandShapeList() {
    int num_tensor = this->RandInt(1, 128);
    vector<Dims> shape(num_tensor);
    int dims = this->RandInt(2, 3);
    for (int i = 0; i < num_tensor; ++i) {
      vector<Index> tensor_shape(dims, 0);
      for (int j = 0; j < dims; ++j) {
        tensor_shape[j] = this->RandInt(1, 512);
      }
      shape[i] = tensor_shape;
    }
    return shape;
  }


  vector<Index> GetRandShape(int dims = -1) {
    if (dims < 0) {
      dims = this->RandInt(1, 5);
    }
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
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  t.template mutable_data<int16>();

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size - should not trigger allocation
  t.Resize(shape);

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(int16));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesTypeSizeAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  t.template mutable_data<double>();

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size - should not trigger allocation
  t.Resize(shape);

  ASSERT_NE(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(double));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_FALSE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesSizeTypeNoAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  t.template mutable_data<int16>();

  ASSERT_EQ(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(int16));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<int16>(t.type()));
  ASSERT_TRUE(t.shares_data());
}

TYPED_TEST(TensorTest, TestGetBytesSizeTypeAlloc) {
  Tensor<TypeParam> t;

  // Get an allocation
  auto shape = this->GetRandShape();
  auto size = volume(shape);
  float *ptr = new float[size];

  // Wrap the allocation
  t.ShareData(ptr, size*sizeof(float));

  // Verify internals
  ASSERT_EQ(t.size(), 0);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), vector<Index>{});
  ASSERT_TRUE(IsType<NoType>(t.type()));
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a size
  t.Resize(shape);

  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), 0);
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(t.shares_data());

  // Give the Tensor a type
  t.template mutable_data<double>();

  ASSERT_NE(t.raw_data(), ptr);
  ASSERT_EQ(t.size(), size);
  ASSERT_EQ(t.nbytes(), size*sizeof(double));
  ASSERT_EQ(t.shape(), shape);
  ASSERT_TRUE(IsType<double>(t.type()));
  ASSERT_FALSE(t.shares_data());
}


// Sharing data from a TensorList puts a Tensor into
// a valid state (it has a type), as we enforce that
// the input TensorList has a valid type.
TYPED_TEST(TensorTest, TestShareData) {
  TensorList<TypeParam> tl;
  auto shape = this->GetRandShapeList();
  tl.Resize(shape);
  tl.template mutable_data<float>();

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
  std::vector<Tensor<TypeParam>> tensors(16);
  for (auto& t : tensors) {
    vector<Index> shape = this->GetRandShape(4);
    t.Resize(shape);
    t.template mutable_data<float>();
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
  std::vector<Tensor<TypeParam>> tensors(16);
  // Empty tensors
  TensorList<TypeParam> tl;
  tl.template mutable_data<float>();
  tl.Copy(tensors, 0);

  Tensor<TypeParam> tensor;
  int num_tensor = tl.ntensor();
  ASSERT_EQ(num_tensor, tensors.size());
  const std::vector<Dims>& shape = tl.shape();
  Index total_volume = std::accumulate(shape.begin(), shape.end(), 0,
                                     [](dali::Index acc, const dali::Dims& s) {
                                       return acc + dali::volume(s);
                                     });
  ASSERT_EQ(total_volume, 0);
}

TYPED_TEST(TensorTest, TestResize) {
  Tensor<TypeParam> tensor;

  // Get shape
  vector<Index> shape = this->GetRandShape();
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }
}

TYPED_TEST(TensorTest, TestMultipleResize) {
  Tensor<TypeParam> tensor;

  int num = this->RandInt(2, 20);
  for (int i = 0; i < num; ++i) {
    // Get shape
    vector<Index> shape = this->GetRandShape();
    tensor.Resize(shape);

    // Verify the settings
    ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
    ASSERT_EQ(tensor.size(), volume(shape));
    ASSERT_EQ(tensor.ndim(), shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(tensor.dim(i), shape[i]);
    }
  }
}

TYPED_TEST(TensorTest, TestResizeScalar) {
  Tensor<TypeParam> tensor;

  // Get shape
  vector<Index> shape = {1};
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestResizeZeroSize) {
  Tensor<TypeParam> tensor;

  // Get shape
  vector<Index> shape = {};
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_EQ(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
}

TYPED_TEST(TensorTest, TestTypeChange) {
  Tensor<TypeParam> tensor;

  // Get shape
  vector<Index> shape = this->GetRandShape();
  tensor.Resize(shape);

  // Verify the settings
  ASSERT_NE(tensor.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }

  // Save the pointer
  const void *ptr = tensor.raw_data();
  size_t nbytes = tensor.nbytes();

  // Change the type of the buffer
  tensor.template mutable_data<int>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }

  // No re-allocation should have occured
  ASSERT_EQ(ptr, tensor.raw_data());
  ASSERT_EQ(nbytes, tensor.nbytes());

  // Change the type to a smaller type
  tensor.template mutable_data<uint8>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }

  // No re-allocation should have occured
  ASSERT_EQ(ptr, tensor.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), tensor.nbytes());

  // Change the type to a larger type
  tensor.template mutable_data<double>();

  // Verify the settings
  ASSERT_EQ(tensor.size(), volume(shape));
  ASSERT_EQ(tensor.ndim(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tensor.dim(i), shape[i]);
  }

  // The memory should have been re-allocated
  ASSERT_NE(ptr, tensor.raw_data());
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(double), tensor.nbytes());
}

TYPED_TEST(TensorTest, TestSubspaceTensor) {
  // Insufficient dimensions
  {
    Tensor<TypeParam> empty_tensor;
    vector<Index> empty_shape = {};
    empty_tensor.Resize(empty_shape);
    empty_tensor.set_type(TypeInfo::Create<uint8_t>());
    ASSERT_ANY_THROW(empty_tensor.SubspaceTensor(0));
  }
  {
    Tensor<TypeParam> one_dim_tensor;
    vector<Index> one_dim_shape = {42};
    one_dim_tensor.Resize(one_dim_shape);
    one_dim_tensor.set_type(TypeInfo::Create<uint8_t>());
    ASSERT_ANY_THROW(one_dim_tensor.SubspaceTensor(0));
  }

  // Wrong subspace
  {
    Tensor<TypeParam> tensor;
    auto shape = this->GetRandShape();
    shape.push_back(42);  // ensure we have at least two dims
    tensor.Resize(shape);
    tensor.set_type(TypeInfo::Create<uint8_t>());
    ASSERT_ANY_THROW(tensor.SubspaceTensor(-1));
    ASSERT_ANY_THROW(tensor.SubspaceTensor(shape[0]));
    ASSERT_ANY_THROW(tensor.SubspaceTensor(shape[0] + 1));
  }

  // Valid subspace:
  {
    Tensor<TypeParam> tensor;
    auto shape = this->GetRandShape();
    shape.push_back(42);  // ensure we have at least two dims
    tensor.Resize(shape);
    tensor.set_type(TypeInfo::Create<uint8_t>());
    int plane_size = 1;
    for (size_t i = 1; i < shape.size(); i++) {
      plane_size *= shape[i];
    }
    auto *base_ptr = tensor.template data<uint8_t>();
    for (Index i = 0; i < shape[0]; i++) {
      auto subspace = tensor.SubspaceTensor(i);
      ASSERT_EQ(subspace.template data<uint8_t>(), base_ptr + plane_size * i);
      ASSERT_EQ(subspace.ndim(), tensor.ndim() - 1);
      for (int j = 0; j < subspace.ndim(); j++) {
        ASSERT_EQ(subspace.dim(j), tensor.dim(j + 1));
      }
    }
  }
}

}  // namespace dali
