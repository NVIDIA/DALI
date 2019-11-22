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
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {

template <int Dims>
struct SliceArgs {
  TensorShape<Dims> anchor;
  TensorShape<Dims> shape;
};

template <int Dims, typename Args>
void CheckValidOutputShape(const TensorShape<Dims>& in_sample_shape,
                           const TensorShape<Dims>& out_sample_shape,
                           const Args& args) {
  for (size_t d = 0; d < Dims; d++) {
    DALI_ENFORCE(
      args.anchor[d] >= 0 && (args.anchor[d] + args.shape[d]) <= in_sample_shape[d],
      "Slice dimension " + std::to_string(d) + " is out of bounds : anchor["
      + std::to_string(args.anchor[d]) + "] size[" + std::to_string(args.shape[d])
      + "] input dimension size[" + std::to_string(in_sample_shape[d]) + "]");
    DALI_ENFORCE(args.shape[d] <= out_sample_shape[d],
      "Output shape dimension " + std::to_string(d) + " is too small");
  }
}

template <int Dims, typename Args>
TensorShape<Dims> GetOutputShape(const TensorShape<Dims>& in_sample_shape,
                                 const Args& args) {
  TensorShape<Dims> out_sample_shape(args.shape);
  CheckValidOutputShape(in_sample_shape, out_sample_shape, args);
  return out_sample_shape;
}

template <int Dims, typename Args>
TensorListShape<Dims> GetOutputShapes(const TensorListShape<Dims>& in_shapes,
                                      const std::vector<Args> &args) {
    DALI_ENFORCE(args.size() == static_cast<size_t>(in_shapes.size()),
      "Number of samples and size of slice arguments should match");

    TensorListShape<Dims> output_shapes(in_shapes.size(), Dims);
    for (int i = 0; i < in_shapes.size(); i++) {
      auto out_sample_shape = GetOutputShape(in_shapes[i], args[i]);
      output_shapes.set_tensor_shape(i, out_sample_shape);
    }
    return output_shapes;
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_KERNEL_UTILS_H_
