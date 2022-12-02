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
#include "dali/pipeline/data/view_as_higher_ndim.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {


TEST(TensorList, ViewAsHigherNdim) {
  TensorList<CPUBackend> tl;
  TensorListShape<>  shapes = { { {640, 480}, {320, 240} } };
  tl.Resize(shapes, DALI_FLOAT);

  auto tlv0 = view_as_higher_ndim<float, 3>(tl, true);
  EXPECT_EQ(3, tlv0.shape.sample_dim());
  TensorListShape<> expected_sh0{{ {1, 640, 480}, {1, 320, 240} }};
  EXPECT_EQ(expected_sh0, tlv0.shape);

  auto tlv1 = view_as_higher_ndim<float, 3>(tl, false);
  EXPECT_EQ(3, tlv1.shape.sample_dim());
  TensorListShape<> expected_sh1{{ {640, 480, 1}, {320, 240,  1} }};
  EXPECT_EQ(expected_sh1, tlv1.shape);

  auto tlv2 = view_as_higher_ndim<float, 5>(tl, true);
  EXPECT_EQ(5, tlv2.shape.sample_dim());
  TensorListShape<> expected_sh2{{ {1, 1, 1, 640, 480}, {1, 1, 1, 320, 240} }};
  EXPECT_EQ(expected_sh2, tlv2.shape);

  auto tlv3 = view_as_higher_ndim<float, 5>(tl, false);
  EXPECT_EQ(5, tlv3.shape.sample_dim());
  TensorListShape<> expected_sh3{{ {640, 480, 1, 1, 1}, {320, 240, 1, 1, 1} }};
  EXPECT_EQ(expected_sh3, tlv3.shape);

  auto tlv4 = view_as_higher_ndim<float, 2>(tl, false);
  EXPECT_EQ(2, tlv4.shape.sample_dim());
  EXPECT_EQ(shapes, tlv4.shape);
}


}  // namespace dali
