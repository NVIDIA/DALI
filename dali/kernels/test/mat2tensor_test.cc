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
#include "dali/kernels/test/mat2tensor.h"

namespace dali {
namespace testing {

using TensorShape = kernels::TensorShape<>;
using kernels::view_as_tensor;
using kernels::tensor_shape;

TEST(Mat2Tensor, Shape) {
  cv::Mat mat;
  mat.create(480, 640, CV_8UC3);
  EXPECT_EQ(tensor_shape<3>(mat), TensorShape(480, 640, 3));
  mat.create(123, 321, CV_32FC2);
  EXPECT_EQ(tensor_shape<kernels::DynamicDimensions>(mat), TensorShape(123, 321, 2));
  mat.create(123, 321, CV_32F);
  EXPECT_EQ(tensor_shape<3>(mat), TensorShape(123, 321, 1));
  EXPECT_EQ(tensor_shape<2>(mat), TensorShape(123, 321));
}

TEST(Mat2Tensor, View) {
  cv::Mat mat;
  mat.create(480, 640, CV_32FC3);
  auto tensor = view_as_tensor<float, 3>(mat);
  EXPECT_EQ(tensor.data, mat.ptr<float>(0));
  EXPECT_EQ(tensor.shape, TensorShape(480, 640, 3));
  const cv::Mat &cmat = mat;
  auto tensor2 = view_as_tensor<const float, 3>(mat);
  EXPECT_EQ(tensor2.data, mat.ptr<float>(0));
  EXPECT_EQ(tensor2.shape, TensorShape(480, 640, 3));
  auto tensor3 = view_as_tensor<const float, 3>(cmat);
  EXPECT_EQ(tensor3.shape, TensorShape(480, 640, 3));
  EXPECT_EQ(tensor3.data, cmat.ptr<float>(0));
}

}  // namespace testing
}  // namespace dali
