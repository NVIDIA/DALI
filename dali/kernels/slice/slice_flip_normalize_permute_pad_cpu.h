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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
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
    if (std::is_integral<OutputType>::value && std::is_floating_point<InputType>::value) {
      destination = clamp<OutputType>(std::roundf(element));
    } else {
      destination = clamp<OutputType>(element);
    }
  }
};

struct NormalizePolicy {
  template <typename OutputType, typename InputType>
  static inline void Fill(OutputType &destination, InputType element,
                          const float *mean, const float *inv_stddev) {
    float fpout = (static_cast<float>(element) - (*mean)) * (*inv_stddev);
    if (std::is_integral<OutputType>::value) {
      destination = clamp<OutputType>(std::roundf(fpout));
    } else {
      destination = clamp<OutputType>(fpout);
    }
  }
};

template <typename OutputType, int Dims>
inline void Pad(OutputType *output,
                TensorShape<Dims> out_strides,
                TensorShape<Dims> padded_out_shape,
                float fill_value,
                std::integral_constant<int, 1>) {
  constexpr auto d = Dims - 1;
  for (int64_t i = 0; i < padded_out_shape[d]; i++) {
    *output = Convert<OutputType>(fill_value);
    output += out_strides[d];
  }
}

template <typename OutputType, int Dims, int DimsLeft>
inline void Pad(OutputType *output,
                TensorShape<Dims> out_strides,
                TensorShape<Dims> padded_out_shape,
                float fill_value,
                std::integral_constant<int, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;
  for (int64_t i = 0; i < padded_out_shape[d]; i++) {
    Pad(output, out_strides, padded_out_shape, fill_value,
        std::integral_constant<int, DimsLeft - 1>());
    output += out_strides[d];
  }
}

template <typename Policy, bool IsNormalizationDim, typename OutputType, typename InputType,
          int Dims>
inline void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                          TensorShape<Dims> in_strides,
                                          TensorShape<Dims> out_strides,
                                          TensorShape<Dims> out_shape,
                                          TensorShape<Dims> padded_out_shape,
                                          const float *mean, const float *inv_stddev,
                                          int normalization_dim,
                                          float fill_value,
                                          std::integral_constant<int, 1>) {
  constexpr auto d = Dims - 1;
  int64_t i = 0;
  for (; i < out_shape[d]; i++) {
    const int norm_idx = IsNormalizationDim ? i : 0;
    Policy::Fill(*output, *input, mean + norm_idx, inv_stddev + norm_idx);
    input += in_strides[d];
    output += out_strides[d];
  }

  // pad
  for (; i < padded_out_shape[d]; i++) {
    *output = Convert<OutputType>(fill_value);
    output += out_strides[d];
  }
}

template <typename Policy, bool IsNormalizationDim, typename OutputType, typename InputType,
          int Dims, int DimsLeft>
inline void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                          TensorShape<Dims> in_strides,
                                          TensorShape<Dims> out_strides,
                                          TensorShape<Dims> out_shape,
                                          TensorShape<Dims> padded_out_shape,
                                          const float *mean, const float *inv_stddev,
                                          int normalization_dim,
                                          float fill_value,
                                          std::integral_constant<int, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;
  const bool IsNextNormalizationDim = (d + 1 == normalization_dim);
  int64_t i = 0;
  if (IsNextNormalizationDim) {
    for (; i < out_shape[d]; i++) {
      SliceFlipNormalizePermuteImpl<Policy, true>(
          output, input, in_strides, out_strides, out_shape, padded_out_shape, mean, inv_stddev,
          normalization_dim, fill_value,
          std::integral_constant<int, DimsLeft - 1>());
      input += in_strides[d];
      output += out_strides[d];
    }
  } else {
    for (; i < out_shape[d]; i++) {
      const int norm_idx = IsNormalizationDim ? i : 0;
      SliceFlipNormalizePermuteImpl<Policy, false>(
          output, input, in_strides, out_strides, out_shape, padded_out_shape, mean + norm_idx,
          inv_stddev + norm_idx, normalization_dim, fill_value,
          std::integral_constant<int, DimsLeft - 1>());
      input += in_strides[d];
      output += out_strides[d];
    }
  }

  for (; i < padded_out_shape[d]; i++) {
    Pad(output, out_strides, padded_out_shape, fill_value,
        std::integral_constant<int, DimsLeft - 1>());
    output += out_strides[d];
  }
}

template <typename OutputType, typename InputType, int Dims>
void SliceFlipNormalizePermute(OutputType *output, const InputType *input,
                               const TensorShape<Dims> &in_strides,
                               const TensorShape<Dims> &out_strides,
                               const TensorShape<Dims> &out_shape,
                               const TensorShape<Dims> &padded_out_shape,
                               const std::vector<float> &mean,
                               const std::vector<float> &inv_stddev,
                               int normalization_dim,
                               float fill_value) {
  DALI_ENFORCE(mean.size() == inv_stddev.size());
  DALI_ENFORCE(mean.size() <= 1 || normalization_dim < Dims);
  const bool should_normalize = !mean.empty();
  const bool IsNextNormalizationDim = (0 == normalization_dim);
  if (should_normalize) {
    if (IsNextNormalizationDim) {
      detail::SliceFlipNormalizePermuteImpl<NormalizePolicy, true>(
          output, input, in_strides, out_strides, out_shape, padded_out_shape,
          mean.data(), inv_stddev.data(), normalization_dim, fill_value,
          std::integral_constant<int, Dims>());
    } else {
      detail::SliceFlipNormalizePermuteImpl<NormalizePolicy, false>(
          output, input, in_strides, out_strides, out_shape, padded_out_shape,
          mean.data(), inv_stddev.data(), normalization_dim, fill_value,
          std::integral_constant<int, Dims>());
    }
  } else {
    detail::SliceFlipNormalizePermuteImpl<ClampPolicy, false>(
        output, input, in_strides, out_strides, out_shape, padded_out_shape,
        nullptr, nullptr, 0, fill_value,
        std::integral_constant<int, Dims>());
  }
}

}  // namespace detail

template <typename OutputType, typename InputType, int Dims>
class SliceFlipNormalizePermutePadCpu {
 public:
  using Args = SliceFlipNormalizePermutePadArgs<Dims>;

  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const Args &args) {
    KernelRequirements req;
    TensorShape<Dims> out_shape(args.padded_shape);
    CheckValidOutputShape(in.shape, out_shape, args);
    out_shape = detail::permute(out_shape, args.permuted_dims);
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, out_shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> &out,
           const InTensorCPU<InputType, Dims> &in,
           const Args &orig_args) {
    auto args = detail::ProcessArgs(orig_args, in.shape);
    detail::SliceFlipNormalizePermute(
        out.data, in.data + args.input_offset, args.in_strides,
        args.out_strides, args.out_shape, args.padded_out_shape,
        args.mean, args.inv_stddev, args.normalization_dim, args.padding_val);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_
