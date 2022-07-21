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
 * SetContiguous(true) was added in few places, as there is a difference in defaults between
   current TensorVector and TensorList.
 **************************************************************************************************/

template <typename TypeParam>
class TensorVectorTest : public DALITest {
 public:
  using Backend = std::tuple_element_t<0, TypeParam>;
  static constexpr bool kState = std::tuple_element_t<1, TypeParam>::value;

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
};

// Pairs of BackendType, ContiguousOption to be used in tests
using TensorVectorBackendContiguous =
    ::testing::Types<std::pair<CPUBackend, std::true_type>, std::pair<CPUBackend, std::false_type>,
                     std::pair<GPUBackend, std::true_type>, std::pair<GPUBackend, std::false_type>>;

TYPED_TEST_SUITE(TensorVectorTest, TensorVectorBackendContiguous);

TYPED_TEST(TensorVectorTest, TestCopy) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
  tl.SetContiguous(this->kState);

  tl.template set_type<float>();

  auto shape = this->GetRandShape();
  tl.Resize(shape);

  TensorVector<Backend> tl2s[2];
  tl2s[0].SetContiguous(this->kState);
  tl2s[1].SetContiguous(!this->kState);
  for (auto &tl2 : tl2s) {
    tl2.Copy(tl);

    ASSERT_EQ(tl.num_samples(), tl2.num_samples());
    ASSERT_EQ(tl.type(), tl2.type());
    ASSERT_EQ(tl._num_elements(), tl2._num_elements());

    for (int i = 0; i < shape.size(); ++i) {
      ASSERT_EQ(tl.tensor_shape(i), tl.tensor_shape(i));
      ASSERT_EQ(volume(tl.tensor_shape(i)), volume(tl2.tensor_shape(i)));
    }
  }
}

TYPED_TEST(TensorVectorTest, TestCopyEmpty) {
  using Backend = std::tuple_element_t<0, TypeParam>;
  TensorVector<Backend> tl;
  tl.SetContiguous(this->kState);

  tl.template set_type<float>();

  TensorVector<Backend> tl2s[2];
  tl2s[0].SetContiguous(this->kState);
  tl2s[1].SetContiguous(!this->kState);
  for (auto &tl2 : tl2s) {
    tl2.Copy(tl);
    ASSERT_EQ(tl.num_samples(), tl2.num_samples());
    ASSERT_EQ(tl.type(), tl2.type());
    ASSERT_EQ(tl._num_elements(), tl2._num_elements());
  }
}

}  // namespace test
}  // namespace dali
