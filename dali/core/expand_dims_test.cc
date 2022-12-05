// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/expand_dims.h"
#include "dali/core/tensor_shape.h"

namespace dali {

TEST(ExpandDims, TensorListShape_dynamic) {
  TensorListShape<>  shapes = { { {640, 480}, {320, 240} } };

  auto shapes0 = expand_dims<DynamicDimensions>(shapes, 0, 3);
  // EXPECT_EQ(3, shapes0.sample_dim());
  // TensorListShape<> expected_sh0{{ {1, 640, 480}, {1, 320, 240} }};
  // EXPECT_EQ(expected_sh0, shapes0);

//   auto shapes1 = expand_dims<>(shapes, -1, 3);
//   EXPECT_EQ(3, shapes1.sample_dim());
//   TensorListShape<> expected_sh1{{ {640, 480, 1}, {320, 240,  1} }};
//   EXPECT_EQ(expected_sh1, shapes1);

//   auto shapes2 = expand_dims<>(shapes, 0, 5);
//   EXPECT_EQ(5, shapes2.sample_dim());
//   TensorListShape<> expected_sh2{{ {1, 1, 1, 640, 480}, {1, 1, 1, 320, 240} }};
//   EXPECT_EQ(expected_sh2, shapes2);

//   auto shapes3 = expand_dims(shapes, -1, 5);
//   EXPECT_EQ(5, shapes3.sample_dim());
//   TensorListShape<> expected_sh3{{ {640, 480, 1, 1, 1}, {320, 240, 1, 1, 1} }};
//   EXPECT_EQ(expected_sh3, shapes3);

//   auto shapes4 = expand_dims<>(shapes, -1, 2);  // no op
//   EXPECT_EQ(2, shapes4.sample_dim());
//   EXPECT_EQ(shapes, shapes4);
}


}  // namespace dali
