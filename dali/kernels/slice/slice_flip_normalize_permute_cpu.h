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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_CPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_CPU_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_common.h"
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/util/half.hpp"

namespace dali {
namespace kernels {

namespace detail {

struct ClampPolicy {
  template <typename OutputType, typename InputType>
  static inline void Fill(OutputType &destination, InputType element,
                          const float *mean, const float *inv_stddev) {
    (void) mean;
    (void) inv_stddev;
    destination = clamp<OutputType>(element);
  }
};

struct NormalizePolicy {
  template <typename OutputType, typename InputType>
  static inline void Fill(OutputType &destination, InputType element,
                          const float *mean, const float *inv_stddev) {
    destination = clamp<OutputType>(
      (static_cast<float>(element) - (*mean)) * (*inv_stddev));
  }
};

template <typename Policy, typename OutputType, typename InputType, size_t Dims>
void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                   const std::array<int64_t, Dims> &in_strides,
                                   const std::array<int64_t, Dims> &out_strides,
                                   const std::array<int64_t, Dims> &out_shape,
                                   const std::array<int64_t, Dims> &padded_out_shape,
                                   const float *mean, const float *inv_stddev,
                                   size_t normalization_dim,
                                   std::integral_constant<size_t, 1>) {
  constexpr auto d = Dims - 1;
  int i = 0;
  if (input) {
    for (; i < out_shape[d]; i++) {
      size_t norm_idx = normalization_dim == d ? i : 0;
      Policy::Fill(*output, *input, mean + norm_idx, inv_stddev + norm_idx);
      input += in_strides[d];
      output += out_strides[d];
    }
  }

  // zero pad
  for (; i < padded_out_shape[d]; i++) {
    *output = 0;
    output += out_strides[d];
  }
}

template <typename Policy, typename OutputType, typename InputType, size_t Dims, size_t DimsLeft>
void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                   const std::array<int64_t, Dims> &in_strides,
                                   const std::array<int64_t, Dims> &out_strides,
                                   const std::array<int64_t, Dims> &out_shape,
                                   const std::array<int64_t, Dims> &padded_out_shape,
                                   const float *mean, const float *inv_stddev,
                                   size_t normalization_dim,
                                   std::integral_constant<size_t, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;

  int i = 0;
  if (input) {
    for (; i < out_shape[d]; i++) {
      size_t norm_idx = normalization_dim == d ? i : 0;
      SliceFlipNormalizePermuteImpl<Policy>(
          output, input, in_strides, out_strides, out_shape, padded_out_shape, mean + norm_idx,
          inv_stddev + norm_idx, normalization_dim, std::integral_constant<size_t, DimsLeft - 1>());
      input += in_strides[d];
      output += out_strides[d];
    }
  }

  // zero pad
  for (; i < padded_out_shape[d]; i++) {
    SliceFlipNormalizePermuteImpl<Policy, OutputType, InputType>(
        output, nullptr, in_strides, out_strides, out_shape, padded_out_shape, nullptr, nullptr, 0,
        std::integral_constant<size_t, DimsLeft - 1>());
    output += out_strides[d];
  }
}

template <typename OutputType, typename InputType, size_t Dims>
void SliceFlipNormalizePermute(OutputType *output, const InputType *input,
                               const std::array<int64_t, Dims> &in_strides,
                               const std::array<int64_t, Dims> &out_strides,
                               const std::array<int64_t, Dims> &out_shape,
                               const std::array<int64_t, Dims> &padded_out_shape,
                               const std::vector<float> &mean,
                               const std::vector<float> &inv_stddev,
                               size_t normalization_dim) {
  const bool should_normalize = !mean.empty() && !inv_stddev.empty();
  if (should_normalize) {
    detail::SliceFlipNormalizePermuteImpl<NormalizePolicy>(
        output, input, in_strides, out_strides, out_shape, padded_out_shape,
        mean.data(), inv_stddev.data(), normalization_dim,
        std::integral_constant<size_t, Dims>());
  } else {
    detail::SliceFlipNormalizePermuteImpl<ClampPolicy>(
        output, input, in_strides, out_strides, out_shape, padded_out_shape,
        nullptr, nullptr, normalization_dim,
        std::integral_constant<size_t, Dims>());
  }
}

}  // namespace detail

template <typename OutputType, typename InputType, size_t Dims>
class SliceFlipNormalizePermuteCPU {
 public:
  using Args = SliceFlipNormalizePermuteArgs<OutputType, Dims>;

  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const Args &args) {
    KernelRequirements req;
    TensorShape<Dims> out_shape(args.shape);
    CheckValidOutputShape<Dims>(in.shape, out_shape, args);

    if (args.should_pad) {
      out_shape = TensorShape<Dims>(args.padded_shape);
    }

    if (args.should_permute) {
      out_shape = detail::permute<Dims>(out_shape, args.permuted_dims);
    }
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, out_shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> &out,
           const InTensorCPU<InputType, Dims> &in,
           const Args &args) {
    auto processed_args = detail::ProcessArgs<OutputType, Dims>(args, in.shape);
    detail::SliceFlipNormalizePermute(
        out.data, in.data + processed_args.input_offset, processed_args.in_strides, processed_args.out_strides,
        processed_args.out_shape, processed_args.padded_out_shape,
        processed_args.mean, processed_args.inv_stddev,
        processed_args.normalization_dim);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_CPU_H_
