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

#ifndef DALI_PIPELINE_OPERATORS_GEOMETRIC_FLIP_UTIL_H_
#define DALI_PIPELINE_OPERATORS_GEOMETRIC_FLIP_UTIL_H_

#include <vector>
#include "dali/kernels/tensor_shape.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

inline kernels::TensorShape<4> TransformShapeNHWC(const Dims &shape) {
  return kernels::TensorShape<4>(std::array<Index, 4>{1, shape[0], shape[1], shape[2]});
}

// In the NCHW layout every channel is treated as a separate plane in a volumetric image
inline kernels::TensorShape<4> TransformShapeNCHW(const Dims &shape) {
  return kernels::TensorShape<4>(std::array<Index, 4>{shape[0], shape[1], shape[2], 1});
}

inline kernels::TensorListShape<4> TransformShapes(const std::vector<Dims> &shapes,
                                                   bool nhwc_layout) {
  std::vector<kernels::TensorShape<4>> result(shapes.size());
  if (nhwc_layout) {
    std::transform(shapes.begin(), shapes.end(), result.begin(), TransformShapeNHWC);
  } else {
    std::transform(shapes.begin(), shapes.end(), result.begin(), TransformShapeNCHW);
  }
  return kernels::TensorListShape<4>(result);
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_GEOMETRIC_FLIP_UTIL_H_
