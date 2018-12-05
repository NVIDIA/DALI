// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_CROP_KERNEL_CROP_KERNEL_H_
#define DALI_PIPELINE_OPERATORS_CROP_KERNEL_CROP_KERNEL_H_

#include <tuple>
#include <vector>

#include "dali/common.h"
#include "dali/pipeline/operators/crop/kernel/coords.h"
#include "dali/pipeline/operators/crop/kernel/sequence.h"

namespace dali {
namespace detail {

template <size_t N>
std::array<Index, N> ToStaticShape(const std::vector<Index> &shape) {
  std::array<Index, N> result;
  for (size_t i = 0; i < N; i++) {
    result[i] = shape[i];
  }
  return result;
}

template <size_t N>
std::vector<Index> ToDynamicShape(const std::array<Index, N> &shape) {
  return {shape.begin(), shape.end()};
}

template <typename InType, typename OutType, typename OutLayout>
class CropKernel {
 public:
  CropKernel() = delete;
  static constexpr size_t input_dim = 3;
  static constexpr size_t output_dim = 3;
  // TODO(klecki) - should be converted to some kind of std::tuple for multiple input and outputs
  using InputType = InType;
  using OutputType = OutType;
  using InputShape = std::array<Index, input_dim>;
  using OutputShape = std::array<Index, output_dim>;

  static constexpr bool can_calculate_output_size = true;

  // TODO(klecki): One of the issues is if this type can depend on T and U or if we can have
  //               the same KernelAttributes for all concrete CropKernel<T, U> types.
  struct KernelAttributes {
    const Index h_start;
    const Index w_start;
    const Index H_out;
    const Index W_out;
  };

  static std::array<Index, output_dim> CalcOutputSize(const std::array<Index, input_dim> &in,
                                                      KernelAttributes attr) {
    return permuteShape({attr.H_out, attr.W_out, in[2]}, OutLayout{});
  }

  static void Run(const InputType *in, const InputShape &in_shape, KernelAttributes attr,
                  OutputType *out, const OutputShape &out_shape) {
    const auto *in_ptr = in + getOffset<0, 1, 2>(in_shape, {attr.h_start, attr.w_start, 0});
    for (Index h = 0; h < attr.H_out; h++) {
      for (Index w = 0; w < attr.W_out; w++) {
        for (Index c = 0; c < in_shape[2]; c++) {
          // From HWC
          const auto in_idx = getOffset<0, 1, 2>(in_shape, {h, w, c});
          // To HWC or CHW
          const auto out_idx = getOffset(out_shape, {h, w, c}, OutLayout{});
          out[out_idx] = static_cast<OutputType>(in_ptr[in_idx]);
        }
      }
    }
  }
};

template <typename InType, typename OutType, typename OutLayout>
using SequenceCropKernel = SequenceAdapter<CropKernel<InType, OutType, OutLayout>>;

}  // namespace detail
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_KERNEL_CROP_KERNEL_H_
