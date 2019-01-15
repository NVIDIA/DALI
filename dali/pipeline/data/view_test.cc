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
#include "dali/pipeline/data/views.h"

#define EXPECT_ENFORCE_FAIL(statement) EXPECT_THROW(statement, std::runtime_error)

namespace dali {

TEST(TensorList, View) {
  TensorList<CPUBackend> tl;
  std::vector<Dims> shapes = { { 640, 480, 3}, { 320, 240, 1 } };
  tl.Resize(shapes);
  auto tlv = view<float>(tl);

  ASSERT_EQ(static_cast<int>(tlv.num_samples()), static_cast<int>(shapes.size()));

  for (int i = 0; i < tlv.num_samples(); i++) {
    EXPECT_EQ(tlv.shape[i], kernels::TensorShape<>(shapes[i]));
  }
}

TEST(TensorList, View_StaticDim) {
  TensorList<CPUBackend> tl;
  std::vector<Dims> shapes = { { 640, 480, 3}, { 320, 240, 1 } };
  tl.Resize(shapes);
  auto tlv = view<float, 3>(tl);

  ASSERT_EQ(static_cast<int>(tlv.num_samples()), static_cast<int>(shapes.size()));

  for (int i = 0; i < tlv.num_samples(); i++) {
    EXPECT_EQ(tlv.shape[i], kernels::TensorShape<>(shapes[i]));
  }

  EXPECT_ENFORCE_FAIL((view<float, 2>(tl)));
}

TEST(TensorList, ViewAsTensor_Fail_NonUniform) {
  TensorList<CPUBackend> tl;
  std::vector<Dims> shapes = { { 640, 480, 3}, { 640, 360, 3 } };
  tl.Resize(shapes);
  EXPECT_ENFORCE_FAIL((view_as_tensor<float>(tl)))
    << "Non-uniform tensor list cannot be viewed as a tensor and view_as_tensor should throw";
}

TEST(TensorList, ViewAsTensor_StaticDim) {
  TensorList<CPUBackend> tl;
  std::vector<Dims> shapes = { { 640, 480, 3}, { 640, 480, 3 } };
  tl.Resize(shapes);
  EXPECT_ENFORCE_FAIL((view_as_tensor<float, 3>(tl)))
    << "List of 3D tensor should yield a 4D flattened tensor";
  kernels::TensorView<kernels::StorageCPU, float, 4> tv;
  EXPECT_NO_THROW((tv = view_as_tensor<float, 4>(tl)))
    << "List of 3D tensor should yield a 4D flattened tensor";

  ASSERT_EQ(tv.dim(), static_cast<int>(shapes[0].size()) + 1)
    << "Resulting tensor must be " << shapes[0].size()+1 << "D";

  EXPECT_EQ(tv.shape[0], shapes.size())
    << "Outermost size must be equal to the number of samples in the input";

  for (int i = 0; i < static_cast<int>(shapes[0].size()); i++) {
    EXPECT_EQ(tv.shape[i + 1], shapes[0][i]) << "Wrong size along dimension " << i+1;
  }
}

TEST(TensorList, ViewAsTensor) {
  TensorList<CPUBackend> tl;
  std::vector<Dims> shapes = { { 640, 480, 3}, { 640, 480, 3 } };
  tl.Resize(shapes);
  auto tv = view_as_tensor<float>(tl);

  ASSERT_EQ(tv.dim(), static_cast<int>(shapes[0].size()) + 1)
    << "Resulting tensor must be " << shapes[0].size()+1 << "D";

  EXPECT_EQ(tv.shape[0], shapes.size())
    << "Outermost size must be equal to the number of samples in the input";

  for (int i = 0; i < static_cast<int>(shapes[0].size()); i++) {
    EXPECT_EQ(tv.shape[i + 1], shapes[0][i]) << "Wrong size along dimension " << i+1;
  }
}

TEST(Tensor, ViewAsTensor) {
  Tensor<CPUBackend> t;
  Dims shape = { 320, 240, 3};
  t.Resize(shape);
  auto tv = view<float>(t);
  ASSERT_EQ(tv.dim(), 3) << "Expected a 3D tensor as a result";
  kernels::TensorView<kernels::StorageCPU, float, 3> tv3;
  EXPECT_NO_THROW((tv3 = view<float, 3>(t)));
  EXPECT_ENFORCE_FAIL((view<float, 4>(t)));
  EXPECT_EQ(tv.shape, kernels::TensorShape<>(shape));
  EXPECT_EQ(tv.shape, tv3.shape);
}

}  // namespace dali
