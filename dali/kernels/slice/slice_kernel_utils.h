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

#ifndef DALI_KERNELS_SLICE_SLICE_KERNEL_UTILS_H_
#define DALI_KERNELS_SLICE_SLICE_KERNEL_UTILS_H_

#include <vector>
#include <utility>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

template <std::size_t Dims>
struct SliceArgs {
  std::array<int64_t, Dims> anchor;
  std::array<int64_t, Dims> shape;
};

template <std::size_t Dims, typename Shape>
std::array<int64_t, Dims> GetStrides(const Shape& shape) {
  std::array<int64_t, Dims> strides;
  strides[Dims - 1] = 1;
  for (std::size_t d = Dims - 1; d > 0; d--) {
    strides[d - 1] = strides[d] * shape[d];
  }
  return strides;
}

template <typename std::size_t Dims>
TensorShape<Dims> GetOutputShape(const TensorShape<Dims>& in_sample_shape,
                                 const SliceArgs<Dims>& slice_args) {
  TensorShape<Dims> out_sample_shape(slice_args.shape);
  auto &anchor = slice_args.anchor;

  for (std::size_t d = 0; d < Dims; d++) {
    DALI_ENFORCE(anchor[d] >= 0 && (anchor[d] + out_sample_shape[d]) <= in_sample_shape[d],
      "Slice dimension " + std::to_string(d) +
      " is out of bounds : " + "anchor[" + std::to_string(anchor[d]) +
      "] size[" + std::to_string(out_sample_shape[d]) + "] input dimension size[" +
      std::to_string(in_sample_shape[d]) + "]");
  }
  return out_sample_shape;
}

template <std::size_t Dims>
TensorListShape<Dims> GetOutputShapes(const TensorListShape<Dims>& in_shapes,
                                      const std::vector<SliceArgs<Dims>> &slice_args) {
    DALI_ENFORCE(slice_args.size() == static_cast<std::size_t>(in_shapes.size()),
      "Number of samples and size of slice arguments should match");

    TensorListShape<Dims> output_shapes;
    output_shapes.resize(in_shapes.size(), Dims);
    for (int i = 0; i < in_shapes.size(); i++) {
      auto out_sample_shape = GetOutputShape<Dims>(in_shapes[i], slice_args[i]);
      output_shapes.set_tensor_shape(i, out_sample_shape);
    }
    return output_shapes;
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_KERNEL_UTILS_H_
