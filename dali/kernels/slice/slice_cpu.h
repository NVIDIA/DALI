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
#include <type_traits>
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename OutputType, typename InputType, int Dims, bool OutOfBounds>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const TensorShape<Dims> &in_strides,
                     const TensorShape<Dims> &out_strides,
                     const TensorShape<Dims> &anchor,
                     const TensorShape<Dims> &in_shape,
                     const TensorShape<Dims> &out_shape,
                     std::integral_constant<int, 1>,
                     std::integral_constant<bool, OutOfBounds>) {
  constexpr auto d = Dims - 1;  // NOLINT
  if (OutOfBounds) {
    for (int i = 0; i < out_shape[d]; i++) {
      output[i] = OutputType(0);  // TODO(janton): allow multichannel padding
    }
  } else {
    int in_idx = anchor[d];
    int out_idx = 0;

    // out of bounds (left side)
    for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
      output[out_idx] = OutputType(0);  // TODO(janton): allow multichannel padding
    }

    // within input bounds
    for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
      output[out_idx] = clamp<OutputType>(input[in_idx]);
    }

    // out of bounds (right side)
    for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
      output[out_idx] = OutputType(0);  // TODO(janton): allow multichannel padding
    }
  }
}

template <typename OutputType, typename InputType, int Dims, bool OutOfBounds, int DimsLeft>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const TensorShape<Dims> &in_strides,
                     const TensorShape<Dims> &out_strides,
                     const TensorShape<Dims> &anchor,
                     const TensorShape<Dims> &in_shape,
                     const TensorShape<Dims> &out_shape,
                     std::integral_constant<int, DimsLeft>,
                     std::integral_constant<bool, OutOfBounds>) {
  constexpr auto d = Dims - DimsLeft;  // NOLINT
  int in_idx = anchor[d];
  int out_idx = 0;

  if (anchor[d] > 0 && anchor[d] < in_shape[d])
    input += anchor[d] * in_strides[d];

  // out of bounds (left side)
  for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
    SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                    std::integral_constant<int, DimsLeft - 1>(),
                    std::integral_constant<bool, true>());
    output += out_strides[d];
  }

  // within input bounds
  for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
    SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                    std::integral_constant<int, DimsLeft - 1>(),
                    std::integral_constant<bool, OutOfBounds>());
    output += out_strides[d];
    if (!OutOfBounds)
      input += in_strides[d];
  }

  // out of bounds (right side)
  for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
    SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                    std::integral_constant<int, DimsLeft - 1>(),
                    std::integral_constant<bool, true>());
    output += out_strides[d];
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
                 const TensorShape<Dims> &out_shape) {
  detail::SliceKernelImpl(output, input, in_strides, out_strides, anchor, in_shape, out_shape,
                          std::integral_constant<int, Dims>(),
                          std::integral_constant<bool, false>());
}

template <typename OutputType, typename InputType, int Dims>
class SliceCPU {
 public:
  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const SliceArgs<Dims> &slice_args) {
    KernelRequirements req;
    auto shape = GetOutputShape(in.shape, slice_args);
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> &out,
           const InTensorCPU<InputType, Dims> &in,
           const SliceArgs<Dims> &slice_args) {
    const auto &in_shape = in.shape;
    const auto &out_shape = out.shape;
    const auto &anchor = slice_args.anchor;
    auto in_strides = GetStrides(in_shape);
    auto out_strides = GetStrides(out_shape);
    const InputType *in_ptr = in.data;
    OutputType *out_ptr = out.data;

    auto inptr2 = in_ptr;
    for (int i = 0; i < in_shape[0]; i++) {
      for (int j = 0; j < in_shape[1]; j++) {
        std::cout << " " << *(inptr2);
        inptr2++; 
      }
      std::cout << "\n";
    }

    std::cout << "anchor: ";
    for (int d = 0; d < Dims; d++)
      std::cout << " " << anchor[d];
    std::cout << "\n";

    std::cout << "shape: ";
    for (int d = 0; d < Dims; d++)
      std::cout << " " << out_shape[d];
    std::cout << "\n";

    SliceKernel(out_ptr, in_ptr, in_strides, out_strides, anchor, in_shape, out_shape);

    std::cout << "slice: \n";
    auto ptr = out_ptr;
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        std::cout << " " << *(ptr + i * out_strides[0] + j * out_strides[1]);
      }
      std::cout << "\n";
    }

  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_CPU_H_
