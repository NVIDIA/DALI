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

#ifndef DALI_UTIL_CROP_WINDOW_H_
#define DALI_UTIL_CROP_WINDOW_H_

#include <functional>
#include <utility>
#include "dali/kernels/tensor_shape.h"

namespace dali {

struct CropWindow {
  kernels::TensorShape<> anchor;
  kernels::TensorShape<> shape;

  CropWindow()
    : anchor{0, 0}, shape{0, 0}
  {}

  operator bool() const {
    for (int dim = 0; dim < shape.size(); dim++)
      if (shape[dim] <= 0)
        return false;
    return true;
  }

  inline bool operator==(const CropWindow& oth) const {
    return anchor == oth.anchor && shape == oth.shape;
  }

  inline bool operator!=(const CropWindow& oth) const {
    return !operator==(oth);
  }

  inline bool IsInRange(const kernels::TensorShape<>& input_shape) const {
    DALI_ENFORCE(input_shape.size() == anchor.size()
              && input_shape.size() == shape.size(),
      "Input shape doesn't match number of dimensions of the anchor and/or shape");
    for (int dim = 0; dim < input_shape.size(); dim++)
      if (anchor[dim] < 0 || anchor[dim] + shape[dim] > input_shape[dim])
        return false;
    return true;
  }

  void SetAnchor(kernels::TensorShape<> new_anchor) {
    anchor = std::move(new_anchor);
  }

  void SetShape(kernels::TensorShape<> new_shape) {
    shape = std::move(new_shape);
  }
};

using CropWindowGenerator = std::function<CropWindow(const kernels::TensorShape<>& shape)>;

}  // namespace dali

#endif  // DALI_UTIL_CROP_WINDOW_H_
