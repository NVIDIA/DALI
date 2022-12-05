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

TEST(ExpandDims, TensorListShape_dynamic_ndim) {
  TensorListShape<>  shapes = { { {640, 480}, {320, 240} } };
  int ndim = shapes.sample_dim();

  auto shapes0 = expand_dims<>(shapes, 0, 3);
  EXPECT_EQ(3, shapes0.sample_dim());
  TensorListShape<> expected_sh0{{ {1, 640, 480}, {1, 320, 240} }};
  EXPECT_EQ(expected_sh0, shapes0);

  auto shapes1 = expand_dims<>(shapes, ndim, 3);
  EXPECT_EQ(3, shapes1.sample_dim());
  TensorListShape<> expected_sh1{{ {640, 480, 1}, {320, 240,  1} }};
  EXPECT_EQ(expected_sh1, shapes1);

  auto shapes2 = expand_dims<>(shapes, 0, 5);
  EXPECT_EQ(5, shapes2.sample_dim());
  TensorListShape<> expected_sh2{{ {1, 1, 1, 640, 480}, {1, 1, 1, 320, 240} }};
  EXPECT_EQ(expected_sh2, shapes2);

  auto shapes3 = expand_dims<>(shapes, ndim, 5);
  EXPECT_EQ(5, shapes3.sample_dim());
  TensorListShape<> expected_sh3{{ {640, 480, 1, 1, 1}, {320, 240, 1, 1, 1} }};
  EXPECT_EQ(expected_sh3, shapes3);

  auto shapes4 = expand_dims<>(shapes, 0, ndim);  // no op
  EXPECT_EQ(2, shapes4.sample_dim());
  EXPECT_EQ(shapes, shapes4);
}

TEST(ExpandDims, TensorListShape_static_ndim) {
  TensorListShape<2>  shapes = { { {640, 480}, {320, 240} } };
  int ndim = shapes.sample_dim();

  auto shapes0 = expand_dims<3>(shapes, 0);
  EXPECT_EQ(3, shapes0.sample_dim());
  TensorListShape<3> expected_sh0{{ {1, 640, 480}, {1, 320, 240} }};
  EXPECT_EQ(expected_sh0, shapes0);

  auto shapes1 = expand_dims<3>(shapes, ndim);
  EXPECT_EQ(3, shapes1.sample_dim());
  TensorListShape<> expected_sh1{{ {640, 480, 1}, {320, 240,  1} }};
  EXPECT_EQ(expected_sh1, shapes1);

  auto shapes2 = expand_dims<5>(shapes, 0);
  EXPECT_EQ(5, shapes2.sample_dim());
  TensorListShape<5> expected_sh2{{ {1, 1, 1, 640, 480}, {1, 1, 1, 320, 240} }};
  EXPECT_EQ(expected_sh2, shapes2);

  auto shapes3 = expand_dims<5>(shapes, ndim);
  EXPECT_EQ(5, shapes3.sample_dim());
  TensorListShape<5> expected_sh3{{ {640, 480, 1, 1, 1}, {320, 240, 1, 1, 1} }};
  EXPECT_EQ(expected_sh3, shapes3);

  auto shapes4 = expand_dims<2>(shapes, 0);  // no op
  EXPECT_EQ(2, shapes4.sample_dim());
  EXPECT_EQ(shapes, shapes4);
}

TEST(ExpandDims, TensorShape_dynamic_ndim) {
  TensorShape<> shape = {640, 480};
  int ndim = shape.sample_dim();

  EXPECT_EQ(TensorShape<>(1, 640, 480), expand_dims<>(shape, 0, 3));
  EXPECT_EQ(TensorShape<>(640, 480, 1), expand_dims<>(shape, ndim, 3));
  EXPECT_EQ(TensorShape<>(1, 1, 1, 640, 480), expand_dims<>(shape, 0, 5));
  EXPECT_EQ(TensorShape<>(640, 480, 1, 1, 1), expand_dims<>(shape, ndim, 5));
  EXPECT_EQ(shape, expand_dims<>(shape, 0, ndim));
}

TEST(ExpandDims, TensorShape_static_ndim) {
  TensorShape<2> shape = {640, 480};
  int ndim = shape.sample_dim();

  EXPECT_EQ(TensorShape<3>(1, 640, 480), expand_dims<3>(shape, 0));
  EXPECT_EQ(TensorShape<3>(640, 480, 1), expand_dims<3>(shape, ndim));
  EXPECT_EQ(TensorShape<5>(1, 1, 1, 640, 480), expand_dims<5>(shape, 0));
  EXPECT_EQ(TensorShape<5>(640, 480, 1, 1, 1), expand_dims<5>(shape, ndim));
  EXPECT_EQ(shape, expand_dims<>(shape, 0, ndim));
}


}  // namespace dali
