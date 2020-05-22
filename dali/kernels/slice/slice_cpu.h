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

#ifndef DALI_KERNELS_SLICE_SLICE_CPU_H_
#define DALI_KERNELS_SLICE_SLICE_CPU_H_

#include <vector>
#include <utility>
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename OutputType, typename InputType, int Dims, bool OutOfBounds, bool NeedPad>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const TensorShape<Dims> &in_strides,
                     const TensorShape<Dims> &out_strides,
                     const TensorShape<Dims> &anchor,
                     const TensorShape<Dims> &in_shape,
                     const TensorShape<Dims> &out_shape,
                     const OutputType *fill_values,
                     int channel_dim,
                     std::integral_constant<int, 1>,
                     std::integral_constant<bool, OutOfBounds>,
                     std::integral_constant<bool, NeedPad>) {
  constexpr auto d = Dims - 1;  // NOLINT
  if (OutOfBounds) {
    for (int i = 0; i < out_shape[d]; i++) {
      output[i] = *fill_values;
      if (d == channel_dim)
        fill_values++;
    }
  } else {
    int in_idx = anchor[d];
    int out_idx = 0;

    if (NeedPad) {
      // out of bounds (left side)
      for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
        output[out_idx] = *fill_values;
        if (d == channel_dim)
          fill_values++;
      }
    }

    // within input bounds
    for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
      output[out_idx] = clamp<OutputType>(input[in_idx]);
      if (NeedPad && d == channel_dim)
        fill_values++;
    }

    if (NeedPad) {
      // out of bounds (right side)
      for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
        output[out_idx] = *fill_values;
        if (d == channel_dim)
          fill_values++;
      }
    }
  }
}

template <typename OutputType, typename InputType, int Dims,
          bool OutOfBounds, bool NeedPad, int DimsLeft>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const TensorShape<Dims> &in_strides,
                     const TensorShape<Dims> &out_strides,
                     const TensorShape<Dims> &anchor,
                     const TensorShape<Dims> &in_shape,
                     const TensorShape<Dims> &out_shape,
                     const OutputType *fill_values,
                     int channel_dim,
                     std::integral_constant<int, DimsLeft>,
                     std::integral_constant<bool, OutOfBounds>,
                     std::integral_constant<bool, NeedPad>) {
  constexpr auto d = Dims - DimsLeft;  // NOLINT
  int in_idx = anchor[d];
  int out_idx = 0;

  if (anchor[d] > 0 && anchor[d] < in_shape[d])
    input += anchor[d] * in_strides[d];

  if (NeedPad) {
    // out of bounds (left side)
    for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
      SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                      fill_values, channel_dim,
                      std::integral_constant<int, DimsLeft - 1>(),
                      std::integral_constant<bool, true>(),
                      std::integral_constant<bool, NeedPad>());
      output += out_strides[d];
      if (d == channel_dim)
        fill_values++;
    }
  }

  // within input bounds
  for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
    SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                    fill_values, channel_dim,
                    std::integral_constant<int, DimsLeft - 1>(),
                    std::integral_constant<bool, OutOfBounds>(),
                    std::integral_constant<bool, NeedPad>());
    output += out_strides[d];
    if (!OutOfBounds)
      input += in_strides[d];
    if (NeedPad && d == channel_dim)
      fill_values++;
  }

  if (NeedPad) {
    // out of bounds (right side)
    for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
      SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                      fill_values, channel_dim,
                      std::integral_constant<int, DimsLeft - 1>(),
                      std::integral_constant<bool, true>(),
                      std::integral_constant<bool, NeedPad>());
      output += out_strides[d];
      if (d == channel_dim)
        fill_values++;
    }
  }
}

}  // namespace detail


template <typename OutputType, typename InputType, int Dims>
void SliceKernel(OutputType *output,
                 const InputType *input,
                 const TensorShape<Dims> &in_strides,
                 const TensorShape<Dims> &out_strides,
                 const TensorShape<Dims> &anchor,
                 const TensorShape<Dims> &in_shape,
                 const TensorShape<Dims> &out_shape,
                 const OutputType *fill_values,
                 int channel_dim = -1) {
  bool need_pad = false;
  for (int d = 0; d < Dims && !need_pad; d++) {
    need_pad = (anchor[d] < 0) || ((anchor[d] + out_shape[d]) > in_shape[d]);
  }
  if (need_pad) {
    detail::SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                            fill_values, channel_dim,
                            std::integral_constant<int, Dims>(),
                            std::integral_constant<bool, false>(),
                            std::integral_constant<bool, true>());
  } else {
    detail::SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                            fill_values, channel_dim,
                            std::integral_constant<int, Dims>(),
                            std::integral_constant<bool, false>(),
                            std::integral_constant<bool, false>());
  }
}

template <typename OutputType, typename InputType, int Dims>
class SliceCPU {
 public:
  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const SliceArgs<OutputType, Dims> &slice_args) {
    KernelRequirements req;
    auto shape = GetOutputShape(in.shape, slice_args);
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> &out,
           const InTensorCPU<InputType, Dims> &in,
           const SliceArgs<OutputType, Dims> &slice_args) {
    const auto &in_shape = in.shape;
    const auto &out_shape = out.shape;
    const auto &anchor = slice_args.anchor;
    auto in_strides = GetStrides(in_shape);
    auto out_strides = GetStrides(out_shape);
    const InputType *in_ptr = in.data;
    OutputType *out_ptr = out.data;

    // fill values should not be empty. It should be left default if not used
    assert(!slice_args.fill_values.empty());
    int channel_dim = -1;  // channel dim is only used if a multi-channel fill_values is provided
    const OutputType* fill_values = slice_args.fill_values.data();
    int fill_values_size = slice_args.fill_values.size();
    if (fill_values_size > 1) {
      channel_dim = slice_args.channel_dim;
      DALI_ENFORCE(channel_dim >= 0 && channel_dim < Dims,
        "Channels dimension needs to be specified if multi-channel fill_values is provided");
      DALI_ENFORCE(fill_values_size == out_shape[channel_dim],
        "Multi-channel fill value does not match the number of channels in the input");
    }

    SliceKernel(out_ptr, in_ptr, in_strides, out_strides, anchor, in_shape, out_shape,
                fill_values, channel_dim);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_CPU_H_
