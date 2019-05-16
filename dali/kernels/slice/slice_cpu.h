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

template <typename OutputType, typename InputType, std::size_t Dims>
void SliceKernel(OutputType *output,
                 const InputType *input,
                 const std::array<int64_t, Dims>& in_strides,
                 const std::array<int64_t, Dims>& out_strides,
                 const std::array<int64_t, Dims>& anchor,
                 unsigned int total_pixels) {
  for (unsigned int i = 0; i < total_pixels; i++) {
    unsigned int idx = i;
    unsigned int out_idx = idx;
    unsigned int in_idx = 0;
    for (std::size_t d = 0; d < Dims; d++) {
        unsigned int i_d = idx / out_strides[d];
        idx = idx % out_strides[d];
        in_idx += (anchor[d] + i_d) * in_strides[d];
    }
    output[out_idx] = clamp<OutputType>(input[in_idx]);
  }
}

template <typename OutputType, typename InputType, std::size_t Dims>
class DLL_PUBLIC SliceCPU {
 public:
  SliceCPU() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, Dims> &in,
                                      const SliceArgs<Dims> &slice_args) {
    KernelRequirements req;
    auto shape = GetOutputShape<Dims>(in.shape, slice_args);
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, shape));
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context,
                      OutTensorCPU<OutputType, Dims> &out,
                      const InTensorCPU<InputType, Dims> &in,
                      const SliceArgs<Dims> &slice_args) {
    const auto &in_shape = in.shape;
    const auto &out_shape = out.shape;
    const unsigned int total_size = volume(out_shape);
    const auto &anchor = slice_args.anchor;
    auto in_strides = GetStrides<Dims>(in_shape);
    auto out_strides = GetStrides<Dims>(out_shape);
    const InputType *in_ptr = in.data;
    OutputType *out_ptr = out.data;

    SliceKernel(out_ptr, in_ptr, in_strides, out_strides, anchor, total_size);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_CPU_H_
