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

#ifndef DALI_OPERATORS_GENERIC_FLIP_UTIL_H_
#define DALI_OPERATORS_GENERIC_FLIP_UTIL_H_

#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

constexpr int flip_ndim = 5;

inline TensorShape<flip_ndim> TransformShapeDHWC(const TensorShape<> &shape) {
  std::array<Index, flip_ndim> result{1, 1, 1, 1, 1};
  for (int i = flip_ndim - shape.size(); i < flip_ndim; ++i) {
    result[i] = shape[i + shape.size() - flip_ndim];
  }
  return TensorShape<flip_ndim>(result);
}

inline TensorShape<flip_ndim> TransformShapeHWC(const TensorShape<> &shape) {
  std::array<Index, flip_ndim> result{1, 1, 1, 1, 1};
  for (int i = 2; i < flip_ndim; ++i) {
    result[i] = shape[i + shape.size() - flip_ndim];
  }
  if (shape.size() > 3) {
    result[0] = shape[0];
  }
  return TensorShape<flip_ndim>(result);
}

inline TensorShape<flip_ndim> TransformShapeCDHW(const TensorShape<> &shape) {
  std::array<Index, flip_ndim> result{1, 1, 1, 1, 1};
  result[1] = shape[shape.size() - 3];
  result[2] = shape[shape.size() - 2];
  result[3] = shape[shape.size() - 1];
  result[0] = shape[shape.size() - 4];
  // merge channel and frame dimensions
  if (shape.size() == flip_ndim) {
    result[0] *= shape[0];
  }
  return TensorShape<flip_ndim>(result);
}

inline TensorShape<flip_ndim> TransformShapeCHW(const TensorShape<> &shape) {
  std::array<Index, flip_ndim> result{1, 1, 1, 1, 1};
  result[2] = shape[shape.size() - 2];
  result[3] = shape[shape.size() - 1];
  result[0] = shape[shape.size() - 3];
  // merge channel and frame dimensions
  if (shape.size() == 4) {
    result[0] *= shape[0];
  }
  return TensorShape<flip_ndim>(result);
}

inline TensorListShape<flip_ndim> TransformShapes(const TensorListShape<> &shapes,
                                          const TensorLayout &layout) {
  TensorListShape<flip_ndim> result(shapes.size());
  if (ImageLayoutInfo::IsChannelLast(layout)) {
    if (layout.find('D') >= 0) {
      for (int i = 0; i < shapes.size(); i++) {
        result.set_tensor_shape(i, TransformShapeDHWC(shapes[i]));
      }
    } else {
      for (int i = 0; i < shapes.size(); i++) {
        result.set_tensor_shape(i, TransformShapeHWC(shapes[i]));
      }
    }
  } else {
    if (layout.find('D') >= 0) {
      for (int i = 0; i < shapes.size(); i++) {
        result.set_tensor_shape(i, TransformShapeCDHW(shapes[i]));
      }
    } else {
      for (int i = 0; i < shapes.size(); i++) {
        result.set_tensor_shape(i, TransformShapeCHW(shapes[i]));
      }
    }
  }
  return result;
}

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_FLIP_UTIL_H_
