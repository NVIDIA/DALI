// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef DALI_TEST_MAT2TENSOR_H_
#define DALI_TEST_MAT2TENSOR_H_

#include <opencv2/core.hpp>
#include <stdexcept>
#include <utility>
#include "dali/core/tensor_view.h"
#include "dali/core/mm/memory.h"
#include "dali/kernels/common/copy.h"

namespace dali {
namespace kernels {

template <int ndim>
TensorShape<ndim> tensor_shape(const cv::Mat &mat) {
  TensorShape<ndim> shape;
#if CV_VERSION_MAJOR >= 5
  // OpenCV 5 bounds-checks MatSize indexing, so the legacy convention of
  // reading the number of dimensions from `mat.size[-1]` is no longer valid.
  const int mat_dims = mat.dims;
#else
  // OpenCV 4 stores the number of dimensions immediately before MatSize.
  const int mat_dims = mat.size[-1];
#endif
  if (ndim != DynamicDimensions) {
    // require extra dimension for channels, if #channels > 1
    if (ndim != mat_dims + 1 && !(ndim == mat_dims && mat.channels() == 1))
      throw std::logic_error("Invalid number of dimensions");
  } else {
    shape.resize(mat_dims + 1);
  }
#if CV_VERSION_MAJOR >= 5
  for (int i = 0; i < mat_dims; i++)
    shape[i] = mat.size[i];
  if (shape.size() > mat_dims)
    shape[mat_dims] = mat.channels();
#else
  for (int i = 0; i < shape.size(); i++)
    shape[i] = mat.size[i];
  if (shape.size() > mat_dims)
    shape[mat_dims] = mat.channels();
#endif
  return shape;
}

template <typename T>
void enforce_type(const cv::Mat &mat) {
  using U = std::remove_const_t<T>;
  int depth = cv::DataDepth<U>::value;
  if (depth != mat.depth())
    throw std::logic_error("Invalid matrix data type");
}

template <typename T, int ndim = 3>
TensorView<StorageCPU, T, ndim> view_as_tensor(const cv::Mat &mat) {
  static_assert(std::is_const<T>::value,
    "Cannot create a non-const view of a const cv::Mat (Missing `const T`?)");

  enforce_type<T>(mat);
  return { mat.ptr<T>(0), tensor_shape<ndim>(mat) };
}

template <typename T, int ndim = 3>
TensorView<StorageCPU, T, ndim> view_as_tensor(cv::Mat &mat) {
  enforce_type<T>(mat);
  return { mat.ptr<T>(0), tensor_shape<ndim>(mat) };
}


template <typename MemoryKind = mm::memory_kind::device, typename T = uint8_t, int ndims = 3>
std::pair<TensorView<kind2storage_t<MemoryKind>, T, ndims>, mm::uptr<T>>
copy_as_tensor(const cv::Mat &mat) {
  static_assert(
      mm::is_device_accessible<MemoryKind>,
      "A GPU-accessible memory kind is required.");
  auto tvin = kernels::view_as_tensor<const T, ndims>(mat);
  return copy<MemoryKind>(tvin);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_TEST_MAT2TENSOR_H_
