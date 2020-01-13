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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_COMMON_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_COMMON_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_kernel_utils.h"

namespace dali {
namespace kernels {

template <int Dims>
struct SliceFlipNormalizePermutePadArgs {
  SliceFlipNormalizePermutePadArgs() = default;

  template <typename Shape>
  explicit SliceFlipNormalizePermutePadArgs(const Shape &_shape) {
    for (int d = 0; d < Dims; d++) {
      anchor[d] = 0;
      shape[d] = _shape[d];
      padded_shape[d] = _shape[d];
      flip[d] = false;
      permuted_dims[d] = d;
    }
  }

  TensorShape<Dims> anchor;
  TensorShape<Dims> shape;
  TensorShape<Dims> padded_shape;
  std::array<bool, Dims> flip;
  std::array<int, Dims> permuted_dims;
  int normalization_dim = Dims-1;
  int normalization_index = 0;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
  float padding_val = 0.0f;
};

namespace detail {

template <int Dims>
struct SliceFlipNormalizePermutePadProcessedArgs {
  size_t input_offset;
  TensorShape<Dims> in_strides;
  TensorShape<Dims> out_shape;
  TensorShape<Dims> padded_out_shape;
  TensorShape<Dims> out_strides;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
  int normalization_dim;
  float padding_val = 0.0f;
};

template <typename Container, typename Permutation>
Container permute(const Container &container, const Permutation &source_indices) {
  auto permuted_container = container;
  for (int d = 0, n = size(source_indices); d < n; d++) {
    permuted_container[d] = container[source_indices[d]];
  }
  return permuted_container;
}

template <typename Permutation>
Permutation inverse_permutation(const Permutation &permutation) {
  Permutation inv_perm = permutation;
  for (int d = 0, n = size(permutation); d < n; d++) {
    auto perm_d = permutation[d];
    inv_perm[perm_d] = d;
  }
  return inv_perm;
}

template <int Dims, typename Shape>
SliceFlipNormalizePermutePadProcessedArgs<Dims> ProcessArgs(
    const SliceFlipNormalizePermutePadArgs<Dims> &args,
    const Shape &in_shape) {
  SliceFlipNormalizePermutePadProcessedArgs<Dims> processed_args;

  processed_args.input_offset = 0;
  processed_args.in_strides = GetStrides(in_shape);

  auto slice_shape = args.shape;
  processed_args.out_shape = detail::permute(slice_shape, args.permuted_dims);
  processed_args.padded_out_shape =
    detail::permute(args.padded_shape, args.permuted_dims);
  processed_args.padding_val = args.padding_val;
  processed_args.out_strides = GetStrides(processed_args.padded_out_shape);

  // Flip operation is implemented by manipulating the anchor and the sign of the input strides
  for (int d = 0; d < Dims; d++) {
    if (args.flip[d]) {
      processed_args.input_offset +=
          (args.anchor[d] + slice_shape[d] - 1) * processed_args.in_strides[d];
      processed_args.in_strides[d] = -processed_args.in_strides[d];
    } else {
      processed_args.input_offset += args.anchor[d] * processed_args.in_strides[d];
    }
  }

  processed_args.in_strides = detail::permute(processed_args.in_strides, args.permuted_dims);

  DALI_ENFORCE(args.mean.size() == args.inv_stddev.size());
  const bool should_normalize = !args.mean.empty();
  processed_args.normalization_dim = Dims + 1;
  if (should_normalize) {
    processed_args.mean = args.mean;
    processed_args.inv_stddev = args.inv_stddev;
    const bool is_scalar_norm = args.mean.size() == 1;
    if (!is_scalar_norm) {
      processed_args.normalization_dim =
        detail::inverse_permutation(args.permuted_dims)[args.normalization_dim];
      DALI_ENFORCE(args.mean.size() ==
                   static_cast<size_t>(processed_args.out_shape[processed_args.normalization_dim]));
    }
  }
  return processed_args;
}
}  // namespace detail

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_COMMON_H_
