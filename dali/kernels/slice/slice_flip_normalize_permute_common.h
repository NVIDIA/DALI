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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_COMMON_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_COMMON_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_kernel_test.h"

namespace dali {
namespace kernels {

template <size_t Dims>
struct SliceFlipNormalizePermuteArgs {
  template <typename Shape>
  explicit SliceFlipNormalizePermuteArgs(const Shape &_shape) {
    for (size_t d = 0; d < Dims; d++) {
      anchor[d] = 0;
      shape[d] = _shape[d];
      padded_shape[d] = _shape[d];
      flip[d] = false;
      permuted_dims[d] = d;
    }
  }

  std::array<int64_t, Dims> anchor;
  std::array<int64_t, Dims> shape;
  std::array<int64_t, Dims> padded_shape;
  std::array<bool, Dims> flip;
  std::array<int64_t, Dims> permuted_dims;
  size_t normalization_dim = Dims-1;
  size_t normalization_index = 0;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
};

namespace detail {

template <typename OutputType, size_t Dims>
struct SliceFlipNormalizePermuteProcessedArgs {
  size_t input_offset;
  std::array<int64_t, Dims> in_shape;
  std::array<int64_t, Dims> in_strides;
  std::array<int64_t, Dims> out_shape;
  std::array<int64_t, Dims> padded_out_shape;
  std::array<int64_t, Dims> out_strides;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
  size_t normalization_dim;
};

template <size_t Dims, typename Container>
Container permute(const Container &container, const std::array<int64_t, Dims> &permuted_dims) {
  auto permuted_container = container;
  for (size_t d = 0; d < Dims; d++) {
    permuted_container[d] = container[permuted_dims[d]];
  }
  return permuted_container;
}

template <size_t Dims>
std::array<int64_t, Dims> inverse_permutation(const std::array<int64_t, Dims> &permutation) {
  std::array<int64_t, Dims> inv_perm = permutation;
  for (size_t d = 0; d < Dims; d++) {
    auto perm_d = permutation[d];
    inv_perm[perm_d] = d;
  }
  return inv_perm;
}

template <typename OutputType, size_t Dims, typename Shape>
SliceFlipNormalizePermuteProcessedArgs<OutputType, Dims> ProcessArgs(
    const SliceFlipNormalizePermuteArgs<Dims> &args,
    const Shape &in_shape) {
  SliceFlipNormalizePermuteProcessedArgs<OutputType, Dims> processed_args;

  processed_args.input_offset = 0;
  processed_args.in_strides = GetStrides<Dims>(in_shape);

  auto slice_shape = args.shape;
  processed_args.out_shape = detail::permute<Dims>(slice_shape, args.permuted_dims);
  processed_args.padded_out_shape =
    detail::permute<Dims>(args.padded_shape, args.permuted_dims);
  processed_args.out_strides = GetStrides<Dims>(processed_args.padded_out_shape);

  // Flip operation is implemented by manipulating the anchor and the sign of the input strides
  for (size_t d = 0; d < Dims; d++) {
    if (args.flip[d]) {
      processed_args.input_offset +=
          (args.anchor[d] + slice_shape[d] - 1) * processed_args.in_strides[d];
      processed_args.in_strides[d] = -processed_args.in_strides[d];
    } else {
      processed_args.input_offset += args.anchor[d] * processed_args.in_strides[d];
    }
  }

  processed_args.in_strides = detail::permute(processed_args.in_strides, args.permuted_dims);
  processed_args.normalization_dim = args.normalization_dim;
  if (!args.mean.empty() && !args.inv_stddev.empty()) {
    processed_args.mean = args.mean;
    processed_args.inv_stddev = args.inv_stddev;
    processed_args.normalization_dim =
      detail::inverse_permutation(args.permuted_dims)[args.normalization_dim];
  }
  return processed_args;
}
}  // namespace detail

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_COMMON_H_
