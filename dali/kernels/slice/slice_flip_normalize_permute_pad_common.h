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

  template <typename OutShape, typename InShape>
  explicit SliceFlipNormalizePermutePadArgs(const OutShape &out_sh, const InShape &in_sh) {
    for (int d = 0; d < Dims; d++) {
      anchor[d] = 0;
      shape[d] = out_sh[d];
      in_shape[d] = in_sh[d];
      flip[d] = false;
      permuted_dims[d] = d;
    }
    fill_values.push_back(0.0f);
  }

  TensorShape<Dims> anchor;
  TensorShape<Dims> shape;
  TensorShape<Dims> in_shape;
  std::array<bool, Dims> flip;
  std::array<int, Dims> permuted_dims;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
  std::vector<float> fill_values;
  int channel_dim = -1;
};

namespace detail {

template <int Dims>
struct SliceFlipNormalizePermutePadProcessedArgs {
  size_t input_offset;
  TensorShape<Dims> in_strides;
  TensorShape<Dims> out_shape;
  TensorShape<Dims> out_strides;
  TensorShape<Dims> anchor;
  TensorShape<Dims> in_shape;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
  std::vector<float> fill_values;
  int channel_dim = - 1;
};

template <int Dims, typename Shape>
SliceFlipNormalizePermutePadProcessedArgs<Dims> ProcessArgs(
    const SliceFlipNormalizePermutePadArgs<Dims> &args,
    const Shape &in_shape) {
  SliceFlipNormalizePermutePadProcessedArgs<Dims> processed_args;

  processed_args.input_offset = 0;
  processed_args.in_strides = GetStrides(in_shape);

  auto slice_shape = args.shape;
  permute(processed_args.out_shape, slice_shape, args.permuted_dims);
  processed_args.fill_values = args.fill_values;
  processed_args.out_strides = GetStrides(processed_args.out_shape);

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

  processed_args.in_strides = permute(processed_args.in_strides, args.permuted_dims);
  processed_args.in_shape = permute(args.in_shape, args.permuted_dims);
  processed_args.anchor = permute(args.anchor, args.permuted_dims);

  DALI_ENFORCE(args.mean.size() == args.inv_stddev.size());
  bool should_normalize = !args.mean.empty();
  processed_args.channel_dim = -1;
  bool has_channels = args.mean.size() > 1 || processed_args.fill_values.size() > 1;
  if (has_channels) {
    int channel_dim = args.channel_dim < 0 ? Dims - 1 : args.channel_dim;
    processed_args.channel_dim = inverse_permutation(args.permuted_dims)[channel_dim];
    auto nchannels = static_cast<size_t>(processed_args.out_shape[processed_args.channel_dim]);
    if (should_normalize)
      DALI_ENFORCE(args.mean.size() == nchannels);
    if (processed_args.fill_values.size() == 1) {
      for (size_t i = 1; i < nchannels; i++) {
        processed_args.fill_values.push_back(processed_args.fill_values[0]);
      }
    }
    DALI_ENFORCE(processed_args.fill_values.size() == nchannels);
  }

  if (should_normalize) {
    processed_args.mean = args.mean;
    processed_args.inv_stddev = args.inv_stddev;
    const bool is_scalar_norm = args.mean.size() == 1;
    if (!is_scalar_norm) {
      int channel_dim = args.channel_dim < 0 ? Dims - 1 : args.channel_dim;
      processed_args.channel_dim =
        inverse_permutation(args.permuted_dims)[channel_dim];
      DALI_ENFORCE(args.mean.size() ==
                   static_cast<size_t>(processed_args.out_shape[processed_args.channel_dim]));
    }
  }
  return processed_args;
}
}  // namespace detail

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_COMMON_H_
