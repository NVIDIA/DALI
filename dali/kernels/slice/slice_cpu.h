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

template <typename OutputType, typename InputType, int Dims>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const TensorShape<Dims> &in_strides,
                     const TensorShape<Dims> &out_strides,
                     const TensorShape<Dims> &out_shape,
                     std::integral_constant<int, 1>) {
  for (int i = 0; i < out_shape[Dims - 1]; i++) {
    output[i] = clamp<OutputType>(input[i]);
  }
}

template <typename OutputType, typename InputType, int Dims, int DimsLeft>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const TensorShape<Dims> &in_strides,
                     const TensorShape<Dims> &out_strides,
                     const TensorShape<Dims> &out_shape,
                     std::integral_constant<int, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;  // NOLINT
  for (int i = 0; i < out_shape[d]; i++) {
    SliceKernelImpl(output, input, in_strides, out_strides, out_shape,
                    std::integral_constant<int, DimsLeft - 1>());
    input += in_strides[d];
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
                 const TensorShape<Dims> &out_shape) {
  for (int d = 0; d < Dims; d++) {
    input += in_strides[d] * anchor[d];
  }
  detail::SliceKernelImpl(output, input, in_strides, out_strides, out_shape,
                          std::integral_constant<int, Dims>());
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

    SliceKernel(out_ptr, in_ptr, in_strides, out_strides, anchor, out_shape);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_CPU_H_
