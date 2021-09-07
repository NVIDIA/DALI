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
  if (ndim != DynamicDimensions) {
    // require extra dimension for channels, if #channels > 1
    // number of cv::Mat dimensions is kept at [-1] index of .size field
    if (ndim != mat.size[-1] + 1 && !(ndim == mat.size[-1] && mat.channels() == 1))
      throw std::logic_error("Invalid number of dimensions");
  } else {
    shape.resize(mat.size[-1] + 1);
  }
  for (int i = 0; i < shape.size(); i++)
    shape[i] = mat.size[i];
  if (shape.size() > mat.size[-1])
    shape[mat.size[-1]] = mat.channels();
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
  static_assert(cuda::kind_has_property<MemoryKind, cuda::memory_access::device>::value,
                "A GPU-accessible memory kind is required.");
  auto tvin = kernels::view_as_tensor<const T, ndims>(mat);
  return copy<MemoryKind>(tvin);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_TEST_MAT2TENSOR_H_
