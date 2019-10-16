// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_TENSOR_SHAPE_PRINT_H_
#define DALI_CORE_TENSOR_SHAPE_PRINT_H_

#include <sstream>
#include <string>
#include "dali/core/tensor_shape.h"

namespace dali {

template <int ndim>
std::ostream &operator<<(std::ostream &os, const TensorShape<ndim> &shape) {
  for (int i = 0; i < shape.size(); i++) {
    if (i) os << " x ";
    os << shape[i];
  }
  return os;
}

template <int ndim>
std::ostream &operator<<(std::ostream &os, const TensorListShape<ndim> &shape) {
  os << "{";
  for (int i = 0; i < shape.num_samples(); i++) {
    if (i) os << ",\n ";
    os << shape[i];
  }
  os << "}";
  return os;
}

template <int ndim>
inline string to_string(const TensorShape<ndim> &shape) {
  std::stringstream ss;
  ss << shape;
  return ss.str();
}

template <int ndim>
inline string to_string(const TensorListShape<ndim> &shape) {
  std::stringstream ss;
  ss << shape;
  return ss.str();
}

}  // namespace dali

#endif  // DALI_CORE_TENSOR_SHAPE_PRINT_H_
