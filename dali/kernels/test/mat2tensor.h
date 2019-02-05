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


#ifndef DALI_KERNELS_TEST_MAT2TENSOR_H_
#define DALI_KERNELS_TEST_MAT2TENSOR_H_

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "dali/kernels/tensor_view.h"

namespace dali {
namespace kernels {

template <typename T, int ndim = 3>
TensorView<StorageCPU, const T, ndim> view_as_tensor(const cv::Mat &mat) {
  TensorShape<ndim> shape;
  if (ndim != DynamicDimensions) {
    ASSERT_EQ(ndim, mat.size.dims() + 1), TensorView<StorageCPU, const T, ndim>();
  } else {
    shape.resize(mat.size.dims() + 1);
  }
  int depth = cv::DataDepth<T>::value;
  ASSERT_EQ(mat.depth(), depth), TensorView<StorageCPU, const T, ndim>();
  for (int i = 0; i < mat.size.dims(); i++)
    shape[i] = mat.size[i];
  shape[mat.size.dims()] = mat.channels();
  return { mat.data, shape };
}

template <typename T, int ndim = 3>
TensorView<StorageCPU, T, ndim> view_as_tensor(cv::Mat &mat) {
  TensorShape<ndim> shape;
  if (ndim != DynamicDimensions) {
    ASSERT_EQ(ndim, mat.size.dims() + 1), TensorView<StorageCPU, T, ndim>();
  } else {
    shape.resize(mat.size.dims() + 1);
  }
  int depth = cv::DataDepth<T>::value;
  ASSERT_EQ(mat.depth(), depth), TensorView<StorageCPU, T, ndim>();
  for (int i = 0; i < mat.size.dims(); i++)
    shape[i] = mat.size[i];
  shape[mat.size.dims()] = mat.channels();
  return { mat.data, shape };
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TEST_MAT2TENSOR_H_
