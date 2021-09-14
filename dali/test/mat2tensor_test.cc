// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <opencv2/opencv.hpp>
#include "dali/test/dali_test_config.h"
#include "dali/core/mm/memory.h"
#include "dali/test/mat2tensor.h"
#include "dali/util/image.h"

namespace dali {
namespace testing {

using TensorShape = TensorShape<>;
using kernels::view_as_tensor;
using kernels::tensor_shape;

TEST(Mat2Tensor, Shape) {
  cv::Mat mat;
  mat.create(480, 640, CV_8UC3);
  EXPECT_EQ(tensor_shape<3>(mat), TensorShape(480, 640, 3));
  mat.create(123, 321, CV_32FC2);
  EXPECT_EQ(tensor_shape<DynamicDimensions>(mat), TensorShape(123, 321, 2));
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


namespace {

void CopyAsTensorGpuTest(const cv::Mat &mat) {
  try {
    auto tvpair = kernels::copy_as_tensor<mm::memory_kind::managed>(mat);
    CUDA_CALL(cudaDeviceSynchronize());
    auto imgptr = mat.data;
    auto tvptr = tvpair.first.data;
    ASSERT_EQ(mat.rows * mat.cols * mat.channels(), volume(tvpair.first.shape))
                          << "Sizes don't match";
    for (int i = 0; i < mat.cols * mat.rows * mat.channels(); i++) {
      EXPECT_EQ(imgptr[i], tvptr[i]) << "Test failed at i=" << i;
    }
  } catch (const CUDAError &e) {
    if ((e.is_drv_api() && e.drv_error() == CUDA_ERROR_NOT_SUPPORTED) ||
        (e.is_rt_api() && e.rt_error() == cudaErrorNotSupported)) {
      GTEST_SKIP() << "Unified memory not supported on this platform";
    }
  }
}

}  // namespace

TEST(Mat2Tensor, CopyAsTensorGpuTest) {
  cv::Mat img = cv::imread(ImageList(testing::dali_extra_path()
                           + "/db/single/jpeg", {".jpg", ".jpeg"}, 1)[0]);
  CopyAsTensorGpuTest(img);
}

}  // namespace testing
}  // namespace dali
