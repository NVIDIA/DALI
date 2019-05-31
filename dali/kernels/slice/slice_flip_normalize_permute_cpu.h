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
#include "dali/kernels/slice/slice_kernel_utils.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename NormalizePolicy, typename OutputType, typename InputType, size_t Dims>
void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                   const std::array<int64_t, Dims> &in_strides,
                                   const std::array<int64_t, Dims> &out_strides,
                                   const TensorShape<static_cast<int>(Dims)> &out_shape,
                                   const OutputType *mean, const OutputType *inv_stddev,
                                   const int64_t *permuted_dims,
                                   std::integral_constant<size_t, 1>) {
  constexpr auto d = Dims - 1;  // NOLINT
  for (int i = 0; i < out_shape[Dims - 1]; i++) {
    *output = NormalizePolicy::Normalize(*input, mean, inv_stddev, i);
    input += in_strides[d];
    output += out_strides[d];
  }
}

template <typename NormalizePolicy, typename OutputType, typename InputType, size_t Dims,
          size_t DimsLeft>
void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                   const std::array<int64_t, Dims> &in_strides,
                                   const std::array<int64_t, Dims> &out_strides,
                                   const TensorShape<static_cast<int>(Dims)> &out_shape,
                                   const OutputType *mean, const OutputType *inv_stddev,
                                   const int64_t *permuted_dims,
                                   std::integral_constant<size_t, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;  // NOLINT
  for (int i = 0; i < out_shape[d]; i++) {
    SliceFlipNormalizePermuteImpl<NormalizePolicy>(
      output, input, in_strides, out_strides,
      out_shape, mean, inv_stddev, permuted_dims,
      std::integral_constant<size_t, DimsLeft - 1>());
    input += in_strides[d];
    output += out_strides[d];
  }
}

template <typename NormalizePolicy, typename OutputType, typename InputType, size_t Dims>
void SliceFlipNormalizePermute(OutputType *output, const InputType *input,
                               const std::array<int64_t, Dims> &in_strides,
                               const std::array<int64_t, Dims> &out_strides,
                               const TensorShape<static_cast<int>(Dims)> &out_shape,
                               const OutputType *mean, const OutputType *inv_stddev,
                               const int64_t *permuted_dims) {
  detail::SliceFlipNormalizePermuteImpl<NormalizePolicy>(
    output, input, in_strides, out_strides,
    out_shape, mean, inv_stddev, permuted_dims,
    std::integral_constant<size_t, Dims>());
}

struct NoNormalizePolicy {
  template <typename OutputType, typename InputType>
  static inline OutputType Normalize(InputType element,
                                     const OutputType *mean,
                                     const OutputType *inv_stddev,
                                     size_t i) {
    (void)mean;
    (void)inv_stddev;
    (void)i;
    return clamp<OutputType>(element);
  }
};

struct NormalizePolicy {
  template <typename OutputType, typename InputType>
  static inline OutputType Normalize(InputType element,
                                     const OutputType *mean,
                                     const OutputType *inv_stddev,
                                     size_t i) {
    return (clamp<OutputType>(element) - mean[i]) * inv_stddev[i];
  }
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

}  // namespace detail

template <typename OutputType, size_t Dims>
struct SliceFlipNormalizePermuteArgs {
  std::array<int64_t, Dims> anchor;
  std::array<int64_t, Dims> shape;

  bool should_flip = false;
  std::array<bool, Dims> flip;

  bool should_permute = false;
  std::array<int64_t, Dims> permuted_dims;

  bool should_normalize = false;
  std::vector<OutputType> mean;
  std::vector<OutputType> inv_stddev;
};

template <typename OutputType, typename InputType, size_t Dims>
class SliceFlipNormalizePermuteCPU {
 public:
  using Args = SliceFlipNormalizePermuteArgs<OutputType, Dims>;

  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const Args &args) {
    KernelRequirements req;
    auto shape = GetOutputShape<Dims>(in.shape, args);
    if (args.should_permute) {
      shape = detail::permute<Dims>(shape, args.permuted_dims);
    }
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> &out,
           const InTensorCPU<InputType, Dims> &in,
           const Args &args) {
    const auto *input = in.data;
    auto in_strides = GetStrides<Dims>(in.shape);
    auto out_strides = GetStrides<Dims>(out.shape);

    // Flip operation is implemented by manipulating the anchor
    // and the sign of the input strides
    if (args.should_flip) {
      const auto slice_shape = args.should_permute ?
        detail::permute<Dims>(out.shape, detail::inverse_permutation(args.permuted_dims)) :
        out.shape;
      for (size_t d = 0; d < Dims; d++) {
        if (args.flip[d]) {
          input += (args.anchor[d] + slice_shape[d] - 1) * in_strides[d];
          in_strides[d] = -in_strides[d];
        } else {
          input += args.anchor[d] * in_strides[d];
        }
      }
    } else {
      for (size_t d = 0; d < Dims; d++) {
        input += args.anchor[d] * in_strides[d];
      }
    }

    if (args.should_permute) {
      in_strides = detail::permute(in_strides, args.permuted_dims);
    }

    if (args.should_normalize) {
      auto mean = args.mean;
      auto inv_stddev = args.inv_stddev;
      // Detect the case where the last dimension is flipped
      if (args.should_flip && args.flip[Dims - 1]) {
        std::reverse(mean.begin(), mean.end());
        std::reverse(inv_stddev.begin(), inv_stddev.end());
      }

      detail::SliceFlipNormalizePermute<detail::NormalizePolicy>(
          out.data, input, in_strides, out_strides, out.shape,
          mean.data(), inv_stddev.data(), nullptr);
    } else {
      const OutputType *mean = nullptr, *inv_stddev = nullptr;
      detail::SliceFlipNormalizePermute<detail::NoNormalizePolicy>(
        out.data, input, in_strides, out_strides, out.shape,
        mean, inv_stddev, nullptr);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_CPU_H_
