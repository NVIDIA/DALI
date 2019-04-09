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

#include "dali/pipeline/data/tensor_list.h"

#include <gtest/gtest.h>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/test/dali_test.h"

namespace dali {

template <typename Backend>
class TensorListTest : public DALITest {
 public:
  vector<Dims> GetRandShape() {
    int num_tensor = this->RandInt(1, 64);
    vector<Dims> shape(num_tensor);
    int dims = this->RandInt(2, 3);
    for (int i = 0; i < num_tensor; ++i) {
      vector<Index> tensor_shape(dims, 0);
      for (int j = 0; j < dims; ++j) {
        tensor_shape[j] = this->RandInt(1, 200);
      }
      shape[i] = tensor_shape;
    }
    return shape;
  }

  vector<Dims> GetSmallRandShape() {
    int num_tensor = this->RandInt(1, 32);
    vector<Dims> shape(num_tensor);
    int dims = this->RandInt(2, 3);
    for (int i = 0; i < num_tensor; ++i) {
      vector<Index> tensor_shape(dims, 0);
      for (int j = 0; j < dims; ++j) {
        tensor_shape[j] = this->RandInt(1, 64);
      }
      shape[i] = tensor_shape;
    }
    return shape;
  }

  /**
   * Initialize & check a TensorList based on an input shape
   */
  void SetupTensorList(TensorList<Backend> *tensor_list,
                       const vector<Dims>& shape,
                       vector<Index> *offsets) {
    const int num_tensor = shape.size();

    Index offset = 0;
    for (auto &tmp : shape) {
      offsets->push_back(offset);
      offset += volume(tmp);
    }

    // Resize the buffer
    tensor_list->Resize(shape);

    // Check the internals
    ASSERT_NE(tensor_list->template mutable_data<float>(), nullptr);
    ASSERT_EQ(tensor_list->ntensor(), num_tensor);
    for (int i = 0; i < num_tensor; ++i) {
      ASSERT_EQ(tensor_list->tensor_shape(i), shape[i]);
      ASSERT_EQ(tensor_list->tensor_offset(i), (*offsets)[i]);
    }
  }
};

typedef ::testing::Types<CPUBackend,
                         GPUBackend> Backends;
TYPED_TEST_SUITE(TensorListTest, Backends);

// Note: A TensorList in a valid state has a type. To get to a valid state, we
// can aquire our type relative to the allocation and the size of the buffer
// in the following orders:
//
// type -> size (bytes) : getting size triggers allocation
// size -> type (bytes) : getting type triggers allocation
// bytes (comes w/ type & size)
//
// The following tests attempt to verify the correct behavior for all of
// these cases

TYPED_TEST(TensorListTest, TestGetTypeSizeBytes) {
  TensorList<TypeParam> tl;

  // Give the tensor a type
  tl.template mutable_data<float>();

  ASSERT_EQ(tl.size(), 0);
  ASSERT_EQ(tl.nbytes(), 0);
  ASSERT_EQ(tl.raw_data(), nullptr);

  // Give the tensor list a size. This
  // should trigger an allocation
  auto shape = this->GetRandShape();
  tl.Resize(shape);

  int num_tensor = shape.size();
  vector<Index> offsets;
  Index size = 0;
  for (auto &tmp : shape) {
    offsets.push_back(size);
    size += volume(tmp);
  }

  // Validate the internals
  ASSERT_NE(tl.raw_data(), nullptr);
  ASSERT_EQ(tl.ntensor(), num_tensor);
  ASSERT_EQ(tl.size(), size);
  ASSERT_EQ(tl.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(tl.type()));

  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tl.tensor_offset(i), offsets[i]);
  }
}

TYPED_TEST(TensorListTest, TestGetSizeTypeBytes) {
  TensorList<TypeParam> tl;

  // Give the tensor a size
  auto shape = this->GetRandShape();
  tl.Resize(shape);

  int num_tensor = shape.size();
  vector<Index> offsets;
  Index size = 0;
  for (auto& tmp : shape) {
    offsets.push_back(size);
    size += volume(tmp);
  }

  // Verify the internals
  ASSERT_EQ(tl.size(), size);
  ASSERT_EQ(tl.ntensor(), num_tensor);
  ASSERT_EQ(tl.nbytes(), 0);
  ASSERT_TRUE(IsType<NoType>(tl.type()));

  // Note: We cannot access the underlying pointer yet because
  // the buffer is not in a valid state (it has no type).

  // Give the tensor a type & test internals
  // This should trigger an allocation
  ASSERT_NE(tl.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tl.ntensor(), num_tensor);
  ASSERT_EQ(tl.size(), size);
  ASSERT_EQ(tl.nbytes(), size*sizeof(float));
  ASSERT_TRUE(IsType<float>(tl.type()));

  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tl.tensor_offset(i), offsets[i]);
  }
}

TYPED_TEST(TensorListTest, TestGetBytesThenNoAlloc) {
  TensorList<TypeParam> tl, sharer;

  // Allocate the sharer
  sharer.template mutable_data<float>();
  auto shape = this->GetRandShape();
  sharer.Resize(shape);

  // Share the data to give the tl bytes
  tl.ShareData(&sharer);

  int num_tensor = shape.size();
  vector<Index> offsets;
  Index size = 0;
  for (auto &tmp : shape) {
    offsets.push_back(size);
    size += volume(tmp);
  }

  // Verify the internals
  ASSERT_EQ(tl.raw_data(), sharer.raw_data());
  ASSERT_EQ(tl.size(), size);
  ASSERT_EQ(tl.nbytes(), size*sizeof(float));
  ASSERT_EQ(tl.type(), sharer.type());
  ASSERT_EQ(tl.ntensor(), num_tensor);
  ASSERT_TRUE(tl.shares_data());

  // Give the buffer a type smaller than float.
  // We should have enough shared bytes, so no
  // re-allocation should happen
  tl.template mutable_data<int16>();

  // Verify the internals
  ASSERT_TRUE(tl.shares_data());
  ASSERT_EQ(tl.raw_data(), sharer.raw_data());
  ASSERT_EQ(tl.size(), size);
  ASSERT_EQ(tl.nbytes(), sizeof(int16)*size);
  ASSERT_TRUE(IsType<int16>(tl.type()));
  ASSERT_EQ(tl.ntensor(), num_tensor);

  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tl.tensor_offset(i), offsets[i]);
  }
}

TYPED_TEST(TensorListTest, TestGetBytesThenAlloc) {
  TensorList<TypeParam> tl, sharer;

  // Allocate the sharer
  sharer.template mutable_data<float>();
  auto shape = this->GetRandShape();
  sharer.Resize(shape);

  // Share the data to give the tl bytes
  tl.ShareData(&sharer);

  int num_tensor = shape.size();
  vector<Index> offsets;
  Index size = 0;
  for (auto &tmp : shape) {
    offsets.push_back(size);
    size += volume(tmp);
  }

  // Verify the internals
  ASSERT_EQ(tl.raw_data(), sharer.raw_data());
  ASSERT_EQ(tl.size(), size);
  ASSERT_EQ(tl.nbytes(), size*sizeof(float));
  ASSERT_EQ(tl.type(), sharer.type());
  ASSERT_EQ(tl.ntensor(), num_tensor);
  ASSERT_TRUE(tl.shares_data());

  // Give the buffer a type bigger than float.
  // We do not have enough bytes shared, so
  // this should trigger a reallocation
  tl.template mutable_data<double>();

  // Verify the internals
  ASSERT_FALSE(tl.shares_data());
  ASSERT_NE(tl.raw_data(), sharer.raw_data());
  ASSERT_EQ(tl.size(), size);
  ASSERT_EQ(tl.nbytes(), sizeof(double)*size);
  ASSERT_TRUE(IsType<double>(tl.type()));
  ASSERT_EQ(tl.ntensor(), num_tensor);

  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tl.tensor_offset(i), offsets[i]);
  }
}

TYPED_TEST(TensorListTest, TestZeroSizeResize) {
  TensorList<TypeParam> tensor_list;

  vector<Dims> shape;
  tensor_list.Resize(shape);

  ASSERT_EQ(tensor_list.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor_list.nbytes(), 0);
  ASSERT_EQ(tensor_list.size(), 0);
  ASSERT_FALSE(tensor_list.shares_data());
}

TYPED_TEST(TensorListTest, TestMultipleZeroSizeResize) {
  TensorList<TypeParam> tensor_list;

  int num_tensor = this->RandInt(0, 128);
  vector<Dims> shape(num_tensor);
  tensor_list.Resize(shape);

  ASSERT_EQ(tensor_list.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor_list.nbytes(), 0);
  ASSERT_EQ(tensor_list.size(), 0);
  ASSERT_FALSE(tensor_list.shares_data());

  ASSERT_EQ(tensor_list.ntensor(), num_tensor);
  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tensor_list.tensor_shape(i), vector<Index>{});
    ASSERT_EQ(tensor_list.tensor_offset(i), 0);
  }
}

TYPED_TEST(TensorListTest, TestScalarResize) {
  TensorList<TypeParam> tensor_list;

  int num_scalar = this->RandInt(1, 128);
  vector<Dims> shape(num_scalar, {(Index)1});
  tensor_list.Resize(shape);

  ASSERT_NE(tensor_list.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor_list.nbytes(), num_scalar*sizeof(float));
  ASSERT_EQ(tensor_list.size(), num_scalar);
  ASSERT_FALSE(tensor_list.shares_data());

  for (int i = 0; i < num_scalar; ++i) {
    ASSERT_EQ(tensor_list.tensor_shape(i), vector<Index>{1});
    ASSERT_EQ(tensor_list.tensor_offset(i), i);
  }
}

TYPED_TEST(TensorListTest, TestResize) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  vector<Index> offsets;

  // resize + check called in SetupTensorList
  this->SetupTensorList(&tensor_list, shape, &offsets);
}

TYPED_TEST(TensorListTest, TestMultipleResize) {
  TensorList<TypeParam> tensor_list;

  int rand = this->RandInt(1, 20);
  vector<Dims> shape;
  vector<Index> offsets;
  int num_tensor = 0;
  for (int i = 0; i < rand; ++i) {
    offsets.clear();
    // Setup shape and offsets
    shape = this->GetRandShape();
    num_tensor = shape.size();
    Index offset = 0;
    for (auto &tmp : shape) {
      offsets.push_back(offset);
      offset += volume(tmp);
    }
  }

  // Resize the buffer
  tensor_list.Resize(shape);

  // The only thing that should matter is the resize
  // after the call to 'mutable_data<T>()'
  ASSERT_NE(tensor_list.template mutable_data<float>(), nullptr);
  ASSERT_EQ(tensor_list.ntensor(), num_tensor);
  for (int i = 0; i < num_tensor; ++i) {
    ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }
}
TYPED_TEST(TensorListTest, TestCopy) {
  TensorList<TypeParam> tl;

  tl.template mutable_data<float>();

  auto shape = this->GetRandShape();
  tl.Resize(shape);

  TensorList<TypeParam> tl2;
  tl2.Copy(tl, 0);

  ASSERT_EQ(tl.ntensor(), tl2.ntensor());
  ASSERT_EQ(tl.type(), tl2.type());
  ASSERT_EQ(tl.size(), tl2.size());

  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(tl.tensor_shape(i), tl.tensor_shape(i));
    ASSERT_EQ(volume(tl.tensor_shape(i)), volume(tl2.tensor_shape(i)));
  }
}

TYPED_TEST(TensorListTest, TestCopyEmpty) {
  TensorList<TypeParam> tl;

  tl.template mutable_data<float>();

  TensorList<TypeParam> tl2;
  tl2.Copy(tl, 0);
  ASSERT_EQ(tl.ntensor(), tl2.ntensor());
  ASSERT_EQ(tl.type(), tl2.type());
  ASSERT_EQ(tl.size(), tl2.size());
}

TYPED_TEST(TensorListTest, TestTypeChangeSameSize) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  // Save the pointer
  const void *ptr = tensor_list.raw_data();
  size_t nbytes = tensor_list.nbytes();

  // Change the data type
  tensor_list.template mutable_data<int>();

  // Check the internals
  ASSERT_EQ(tensor_list.ntensor(), shape.size());
  for (size_t i = 0; i < tensor_list.ntensor(); ++i) {
    ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }

  // No memory allocation should have occured
  ASSERT_EQ(ptr, tensor_list.raw_data());
  ASSERT_EQ(nbytes, tensor_list.nbytes());
}

TYPED_TEST(TensorListTest, TestTypeChangeSmaller) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  // Save the pointer
  const void *ptr = tensor_list.raw_data();
  size_t nbytes = tensor_list.nbytes();

  // Change the data type to something smaller
  tensor_list.template mutable_data<uint8>();

  // Check the internals
  ASSERT_EQ(tensor_list.ntensor(), shape.size());
  for (size_t i = 0; i < tensor_list.ntensor(); ++i) {
    ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }

  // No memory allocation should have occured
  ASSERT_EQ(ptr, tensor_list.raw_data());

  // nbytes should have reduced by a factor of 4
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(uint8), tensor_list.nbytes());
}

TYPED_TEST(TensorListTest, TestTypeChangeLarger) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  // Save the pointer
  const void *ptr = tensor_list.raw_data();
  size_t nbytes = tensor_list.nbytes();

  // Change the data type to something larger
  tensor_list.template mutable_data<double>();

  // Check the internals
  ASSERT_EQ(tensor_list.ntensor(), shape.size());
  for (size_t i = 0; i < tensor_list.ntensor(); ++i) {
    ASSERT_EQ(tensor_list.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list.tensor_offset(i), offsets[i]);
  }

  // Size doubled, memory allocation should have occured
  ASSERT_NE(ptr, tensor_list.raw_data());

  // nbytes should have increased by a factor of 2
  ASSERT_EQ(nbytes / sizeof(float) * sizeof(double), tensor_list.nbytes());
}

TYPED_TEST(TensorListTest, TestShareData) {
  TensorList<TypeParam> tensor_list;

  // Setup shape and offsets
  vector<Dims> shape = this->GetRandShape();
  vector<Index> offsets;

  this->SetupTensorList(&tensor_list, shape, &offsets);

  // Create a new tensor_list w/ a smaller data type
  TensorList<TypeParam> tensor_list2;

  // Share the data
  tensor_list2.ShareData(&tensor_list);
  tensor_list2.Resize(vector<Dims>{{tensor_list.size()}});
  tensor_list2.template mutable_data<uint8>();

  // Make sure the pointers match
  ASSERT_EQ(tensor_list.raw_data(), tensor_list2.raw_data());
  ASSERT_TRUE(tensor_list2.shares_data());

  // Verify the default dims of the tensor_list 2
  ASSERT_EQ(tensor_list2.size(), tensor_list.size());

  // Resize the tensor_list2 to match the shape of tensor_list
  tensor_list2.Resize(shape);

  // Check the internals
  ASSERT_TRUE(tensor_list2.shares_data());
  ASSERT_EQ(tensor_list2.raw_data(), tensor_list.raw_data());
  ASSERT_EQ(tensor_list2.nbytes(), tensor_list.nbytes() / sizeof(float) * sizeof(uint8));
  ASSERT_EQ(tensor_list2.ntensor(), tensor_list.ntensor());
  ASSERT_EQ(tensor_list2.size(), tensor_list.size());
  for (size_t i = 0; i < tensor_list.ntensor(); ++i) {
    ASSERT_EQ(tensor_list2.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list2.tensor_offset(i), offsets[i]);
  }

  // Trigger allocation through buffer API, verify we no longer share
  tensor_list2.template mutable_data<double>();
  ASSERT_FALSE(tensor_list2.shares_data());

  // Check the internals
  ASSERT_EQ(tensor_list2.size(), tensor_list.size());
  ASSERT_EQ(tensor_list2.nbytes(), tensor_list.nbytes() / sizeof(float) * sizeof(double));
  ASSERT_EQ(tensor_list2.ntensor(), tensor_list.ntensor());
  for (size_t i = 0; i < tensor_list.ntensor(); ++i) {
    ASSERT_EQ(tensor_list2.tensor_shape(i), shape[i]);
    ASSERT_EQ(tensor_list2.tensor_offset(i), offsets[i]);
  }
}

}  // namespace dali
