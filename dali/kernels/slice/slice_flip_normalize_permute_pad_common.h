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
  std::array<bool, Dims> flip{};
  std::array<int, Dims> permuted_dims{};
  SmallVector<float, 8> mean;
  SmallVector<float, 8> inv_stddev;
  SmallVector<float, 8> fill_values;
  int channel_dim = -1;
};

namespace detail {

template <int Dims>
struct SliceFlipNormalizePermutePadProcessedArgs {
  size_t input_offset = 0;
  TensorShape<Dims> in_strides;
  TensorShape<Dims> out_shape;
  TensorShape<Dims> out_strides;
  TensorShape<Dims> anchor;
  TensorShape<Dims> in_shape;
  SmallVector<float, 4> mean;
  SmallVector<float, 4> inv_stddev;
  SmallVector<float, 4> fill_values;
  int channel_dim = - 1;
};

template <int Dims, typename Shape>
SliceFlipNormalizePermutePadProcessedArgs<Dims> ProcessArgs(
    const SliceFlipNormalizePermutePadArgs<Dims> &args,
    const Shape &in_shape) {
  SliceFlipNormalizePermutePadProcessedArgs<Dims> processed_args;

  processed_args.input_offset = 0;
  processed_args.in_strides = GetStrides(in_shape);
  int channel_dim = args.channel_dim < 0 ? Dims - 1 : args.channel_dim;
  int out_channel_dim = inverse_permutation(args.permuted_dims)[channel_dim];

  auto slice_shape = args.shape;
  permute(processed_args.out_shape, slice_shape, args.permuted_dims);
  processed_args.fill_values = args.fill_values;
  processed_args.out_strides = GetStrides(processed_args.out_shape);
  processed_args.channel_dim = -1;

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

  bool should_normalize = !args.mean.empty();
  bool has_channels = args.mean.size() > 1 || processed_args.fill_values.size() > 1;
  int per_channel_arg_size = std::max(args.mean.size(), processed_args.fill_values.size());
  int nchannels = 0;
  if (has_channels) {
    DALI_ENFORCE(channel_dim >= 0 && channel_dim < Dims,
      "Channel dim must be valid for multi-channel normalization arguments");
    processed_args.channel_dim = out_channel_dim;
    nchannels = processed_args.out_shape[out_channel_dim];
  }

  if (should_normalize) {
    processed_args.mean = args.mean;
    processed_args.inv_stddev = args.inv_stddev;
    if (processed_args.mean.size() == 1 && per_channel_arg_size > 1)
      processed_args.mean.resize(per_channel_arg_size, processed_args.mean[0]);
    else if (processed_args.inv_stddev.size() == 1 && per_channel_arg_size > 1)
      processed_args.inv_stddev.resize(per_channel_arg_size, processed_args.inv_stddev[0]);
  }

  int fill_values_size = processed_args.fill_values.size();
  if (fill_values_size == 0) {
    processed_args.fill_values.resize(std::max(1, per_channel_arg_size), 0.0f);
  } else if (fill_values_size == 1 && per_channel_arg_size > 1) {
    processed_args.fill_values.resize(per_channel_arg_size, processed_args.fill_values[0]);
  } else if (fill_values_size < per_channel_arg_size) {
    processed_args.fill_values.resize(per_channel_arg_size, 0.0f);
  }
  DALI_ENFORCE(per_channel_arg_size == 1 || per_channel_arg_size == nchannels,
    "The number of per-channel arguments should match the number of channels in the output slice");
  return processed_args;
}
}  // namespace detail

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_COMMON_H_
