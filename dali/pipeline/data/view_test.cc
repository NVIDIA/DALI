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
#include <random>
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"

#define EXPECT_ENFORCE_FAIL(statement) EXPECT_THROW(statement, DALIException)

namespace dali {

TEST(TensorList, View) {
  TensorList<CPUBackend> tl;
  TensorListShape<> shapes = { { {640, 480, 3}, {320, 240, 1} } };
  tl.Resize(shapes);
  auto tlv = view<float>(tl);

  ASSERT_EQ(static_cast<int>(tlv.num_samples()), static_cast<int>(shapes.size()));

  for (int i = 0; i < tlv.num_samples(); i++) {
    EXPECT_EQ(tlv.shape[i], shapes[i]);
  }
}

TEST(TensorList, View_StaticDim) {
  TensorList<CPUBackend> tl;
  TensorListShape<>  shapes = { { {640, 480, 3}, {320, 240, 1} } };
  tl.Resize(shapes);
  auto tlv = view<float, 3>(tl);

  ASSERT_EQ(static_cast<int>(tlv.num_samples()), static_cast<int>(shapes.size()));

  for (int i = 0; i < tlv.num_samples(); i++) {
    EXPECT_EQ(tlv.shape[i], shapes[i]);
  }

  EXPECT_ENFORCE_FAIL((view<float, 2>(tl)));
}

TEST(TensorList, ViewAsTensor_Fail_NonUniform) {
  TensorList<CPUBackend> tl;
  TensorListShape<> shapes = { { {640, 480, 3}, {640, 360, 3} } };
  tl.Resize(shapes);
  EXPECT_ENFORCE_FAIL((view_as_tensor<float>(tl)))
    << "Non-uniform tensor list cannot be viewed as a tensor and view_as_tensor should throw";
}

TEST(TensorList, ViewAsTensor_StaticDim) {
  TensorList<CPUBackend> tl;
  TensorListShape<> shapes = { { {640, 480, 3}, {640, 480, 3} } };
  tl.Resize(shapes);
  EXPECT_ENFORCE_FAIL((view_as_tensor<float, 3>(tl)))
    << "List of 3D tensor should yield a 4D flattened tensor";
  TensorView<StorageCPU, float, 4> tv;
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
  TensorListShape<> shapes = { { {640, 480, 3}, {640, 480, 3} } };
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
  TensorShape<> shape = { 320, 240, 3};
  t.Resize(shape);
  auto tv = view<float>(t);
  ASSERT_EQ(tv.dim(), 3) << "Expected a 3D tensor as a result";
  TensorView<StorageCPU, float, 3> tv3;
  EXPECT_NO_THROW((tv3 = view<float, 3>(t)));
  EXPECT_ENFORCE_FAIL((view<float, 4>(t)));
  EXPECT_EQ(tv.shape, shape);
  EXPECT_EQ(tv.shape, tv3.shape);
}

TEST(TensorVector, View) {
  TensorVector<CPUBackend> tvec(10);
  TypeInfo type = TypeInfo::Create<int>();
  tvec.set_type(type);
  std::mt19937_64 rng;
  for (int i = 0; i < 10; i++) {
    tvec[i].Resize(TensorShape<3>(100+i, 40+i, 3+i));
    tvec[i].set_type(type);
    UniformRandomFill(view<int>(tvec[i]), rng, 0, 10000);
  }

  auto tlv = view<int, 3>(tvec);
  const TensorVector<CPUBackend> &ctvec = tvec;
  auto tlv2 = view<const int, 3>(ctvec);

  auto tv_shape = tvec.shape();
  EXPECT_EQ(tv_shape, tlv.shape);
  EXPECT_EQ(tlv2.shape, tlv.shape);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(tlv[i].data, tvec[i].data<int>());
    EXPECT_EQ(tlv2[i].data, tvec[i].data<int>());
    Check(tlv[i], view<int>(tvec[i]));
  }
}

TEST(TensorVector, ReinterpretView) {
  TensorVector<CPUBackend> tvec(10);
  TypeInfo type = TypeInfo::Create<int>();
  tvec.set_type(type);
  std::mt19937_64 rng;
  for (int i = 0; i < 10; i++) {
    tvec[i].Resize(TensorShape<3>(100+i, 40+i, 3+i));
    tvec[i].set_type(type);
    UniformRandomFill(view<int>(tvec[i]), rng, 0, 10000);
  }

  auto tlv = view<int, 3>(tvec);
  auto tlv_i16 = reinterpret_view<int16_t, 3>(tvec);
  const auto& ctvec = tvec;
  auto tlv_u8 = reinterpret_view<const uint8_t, 3>(ctvec);

  auto tv_shape = tvec.shape();
  ASSERT_EQ(tv_shape, tlv.shape);
  for (int i = 0; i < 10; i++) {
    auto s = tv_shape[i];
    TensorShape<3> expected_tlv_i16_shape{s[0], s[1], s[2] * 2};
    EXPECT_EQ(expected_tlv_i16_shape, tlv_i16[i].shape);
    EXPECT_EQ(static_cast<const void*>(tlv[i].data), static_cast<const void*>(tlv_i16[i].data));

    TensorShape<3> expected_tlv_iu8_shape{s[0], s[1], s[2] * 4};
    EXPECT_EQ(expected_tlv_iu8_shape, tlv_u8[i].shape);
    EXPECT_EQ(static_cast<const void*>(tlv[i].data), static_cast<const void*>(tlv_u8[i].data));
  }
}


}  // namespace dali
