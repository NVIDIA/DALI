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
#include "dali/util/half.hpp"

namespace dali {
namespace kernels {

namespace detail {

struct ClampPolicy {
  template <typename OutputType, typename InputType>
  static inline OutputType Value(InputType element, const std::vector<OutputType> &mean,
                                 const std::vector<OutputType> &inv_stddev, size_t idx) {
    (void) mean;
    (void) inv_stddev;
    (void) idx;
    return clamp<OutputType>(element);
  }
};

struct NormalizePolicy {
  template <typename OutputType, typename InputType>
  static inline OutputType Value(InputType element, const std::vector<OutputType> &mean,
                                 const std::vector<OutputType> &inv_stddev, size_t idx) {
    return (clamp<OutputType>(element) - mean[idx]) * inv_stddev[idx];
  }
};

struct ZeroPadPolicy {
  template <typename OutputType, typename InputType>
  static inline OutputType Value(InputType element, const std::vector<OutputType> &mean,
                                 const std::vector<OutputType> &inv_stddev, size_t idx) {
    (void) element;
    (void) mean;
    (void) inv_stddev;
    (void) idx;
    return static_cast<OutputType>(0);
  }
};

template <typename Policy, typename OutputType, typename InputType, size_t Dims>
void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                   const std::array<int64_t, Dims> &in_strides,
                                   const std::array<int64_t, Dims> &out_strides,
                                   const TensorShape<static_cast<int>(Dims)> &out_shape,
                                   const TensorShape<static_cast<int>(Dims)> &padded_out_shape,
                                   const std::vector<OutputType> &mean,
                                   const std::vector<OutputType> &inv_stddev,
                                   size_t normalization_dim,
                                   size_t normalization_index,
                                   const int64_t *permuted_dims,
                                   std::integral_constant<size_t, 1>) {
  constexpr auto d = Dims - 1;
  for (int i = 0; i < out_shape[d]; i++) {
    if (normalization_dim == d)
      normalization_index = i;
    *output = Policy::Value(*input, mean, inv_stddev, normalization_index);
    input += in_strides[d];
    output += out_strides[d];
  }

  for (int i = out_shape[d]; i < padded_out_shape[d]; i++) {
    *output = ZeroPadPolicy::Value(*input, mean, inv_stddev, normalization_index);
    input += in_strides[d];
    output += out_strides[d];
  }
}

template <typename Policy, typename OutputType, typename InputType, size_t Dims, size_t DimsLeft>
void SliceFlipNormalizePermuteImpl(OutputType *output, const InputType *input,
                                   const std::array<int64_t, Dims> &in_strides,
                                   const std::array<int64_t, Dims> &out_strides,
                                   const TensorShape<static_cast<int>(Dims)> &out_shape,
                                   const TensorShape<static_cast<int>(Dims)> &padded_out_shape,
                                   const std::vector<OutputType> &mean,
                                   const std::vector<OutputType> &inv_stddev,
                                   size_t normalization_dim,
                                   size_t normalization_index,
                                   const int64_t *permuted_dims,
                                   std::integral_constant<size_t, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;
  for (int i = 0; i < out_shape[d]; i++) {
    if (normalization_dim == d)
      normalization_index = i;
    SliceFlipNormalizePermuteImpl<Policy>(output, input, in_strides, out_strides, out_shape,
                                          padded_out_shape, mean, inv_stddev, normalization_dim,
                                          normalization_index, permuted_dims,
                                          std::integral_constant<size_t, DimsLeft - 1>());
    input += in_strides[d];
    output += out_strides[d];
  }

  for (int i = out_shape[d]; i < padded_out_shape[d]; i++) {
    SliceFlipNormalizePermuteImpl<ZeroPadPolicy>(
        output, input, in_strides, out_strides, out_shape, padded_out_shape, mean, inv_stddev,
        normalization_dim, normalization_index, permuted_dims,
        std::integral_constant<size_t, DimsLeft - 1>());
    input += in_strides[d];
    output += out_strides[d];
  }
}

template <typename OutputType, typename InputType, size_t Dims>
void SliceFlipNormalizePermute(OutputType *output, const InputType *input,
                               const std::array<int64_t, Dims> &in_strides,
                               const std::array<int64_t, Dims> &out_strides,
                               const TensorShape<static_cast<int>(Dims)> &out_shape,
                               const TensorShape<static_cast<int>(Dims)> &padded_out_shape,
                               const std::vector<OutputType> &mean,
                               const std::vector<OutputType> &inv_stddev,
                               size_t normalization_dim,
                               const int64_t *permuted_dims) {
  const bool should_normalize = !mean.empty() && !inv_stddev.empty();
  if (should_normalize) {
    detail::SliceFlipNormalizePermuteImpl<NormalizePolicy>(
        output, input, in_strides, out_strides, out_shape, padded_out_shape, mean, inv_stddev,
        normalization_dim, 0, permuted_dims, std::integral_constant<size_t, Dims>());
  } else {
    detail::SliceFlipNormalizePermuteImpl<ClampPolicy>(
        output, input, in_strides, out_strides, out_shape, padded_out_shape, mean, inv_stddev,
        normalization_dim, 0, permuted_dims, std::integral_constant<size_t, Dims>());
  }
}

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

  bool should_pad = false;
  std::array<int64_t, Dims> padded_shape;

  bool should_flip = false;
  std::array<bool, Dims> flip;

  bool should_permute = false;
  std::array<int64_t, Dims> permuted_dims;

  bool should_normalize = false;
  size_t normalization_dim = Dims-1;
  size_t normalization_index = 0;
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
    const auto *input = in.data;
    auto in_strides = GetStrides<Dims>(in.shape);

    auto slice_shape = args.shape;
    auto non_padded_out_shape = args.should_permute ?
      detail::permute<Dims>(slice_shape, args.permuted_dims) :
      slice_shape;
    auto padded_out_shape = out.shape;
    auto out_strides = GetStrides<Dims>(padded_out_shape);

    // Flip operation is implemented by manipulating the anchor
    // and the sign of the input strides
    if (args.should_flip) {
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

    std::vector<OutputType> mean, inv_stddev;
    size_t normalization_dim = args.normalization_dim;
    if (args.should_normalize) {
      mean = args.mean;
      DALI_ENFORCE(!mean.empty());
      inv_stddev = args.inv_stddev;
      DALI_ENFORCE(!inv_stddev.empty());

      if (args.should_permute) {
        normalization_dim = detail::inverse_permutation(args.permuted_dims)[normalization_dim];
      }

      // Detect the case where the last dimension is flipped
      if (args.should_flip && args.flip[Dims - 1]) {
        std::reverse(mean.begin(), mean.end());
        std::reverse(inv_stddev.begin(), inv_stddev.end());
      }
    }
    detail::SliceFlipNormalizePermute(out.data, input, in_strides, out_strides,
                                      non_padded_out_shape, padded_out_shape, mean, inv_stddev,
                                      normalization_dim, nullptr);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_CPU_H_
