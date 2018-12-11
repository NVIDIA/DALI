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

#ifndef DALI_PIPELINE_OPERATORS_CROP_KERNEL_SEQUENCE_H_
#define DALI_PIPELINE_OPERATORS_CROP_KERNEL_SEQUENCE_H_

#include <array>
#include <cstdint>
#include <type_traits>

namespace dali {
namespace detail {

template <size_t N>
int64_t CalcOffsetToSubspace(const std::array<int64_t, N> &shape) {
  static_assert(N > 1, "To get subspace there must be at least two dimensions");
  int64_t offset = shape[1];
  for (size_t i = 2; i < N; i++) {
    offset *= shape[i];
  }
  return offset;
}

template <size_t N>
std::array<int64_t, N - 1> GetSubspaceShape(const std::array<int64_t, N> &shape) {
  static_assert(N > 1, "To get subspace there must be at least two dimensions");
  std::array<int64_t, N - 1> result;
  for (size_t i = 1; i < N; i++) {
    result[i - 1] = shape[i];
  }
  return result;
}

template <typename Adapted>
struct SequenceAdapter {
 public:
  SequenceAdapter() = delete;
  static constexpr size_t input_dim = Adapted::input_dim + 1;
  static constexpr size_t output_dim = Adapted::output_dim + 1;
  using InputType = typename Adapted::InputType;
  using OutputType = typename Adapted::OutputType;
  using InputShape = std::array<int64_t, input_dim>;
  using OutputShape = std::array<int64_t, output_dim>;

  static constexpr bool can_calculate_output_size = Adapted::can_calculate_output_size;

  using KernelAttributes = typename Adapted::KernelAttributes;

  static typename std::enable_if<Adapted::can_calculate_output_size,
                                 std::array<int64_t, output_dim>>::type
  CalcOutputSize(const std::array<int64_t, input_dim> &in, KernelAttributes attr) {
    // Forward length of sequence
    std::array<int64_t, output_dim> out;
    out[0] = in[0];
    std::array<int64_t, Adapted::input_dim> adapted_in;
    for (size_t i = 1; i < input_dim; i++) {
      adapted_in[i - 1] = in[i];
    }
    std::array<int64_t, Adapted::output_dim> adapted_out =
        Adapted::CalcOutputSize(adapted_in, attr);
    for (size_t i = 1; i < output_dim; i++) {
      out[i] = adapted_out[i - 1];
    }
    return out;
  }

  static void Run(const InputType *in, const InputShape &in_shape, KernelAttributes attr,
                  OutputType *out, const OutputShape &out_shape) {
    const auto sequence_length = in_shape[0];
    const auto in_offset = CalcOffsetToSubspace(in_shape);
    auto in_subspace_shape = GetSubspaceShape(in_shape);
    const auto out_offset = CalcOffsetToSubspace(out_shape);
    auto out_subspace_shape = GetSubspaceShape(out_shape);
    for (int64_t i = 0; i < sequence_length; i++) {
      auto *in_elem = in + i * in_offset;
      auto *out_elem = out + i * out_offset;
      Adapted::Run(in_elem, in_subspace_shape, attr, out_elem, out_subspace_shape);
    }
  }
};

}  // namespace detail
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_KERNEL_SEQUENCE_H_
