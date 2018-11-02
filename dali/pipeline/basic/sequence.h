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

#ifndef DALI_PIPELINE_BASIC_SEQUENCE_H_
#define DALI_PIPELINE_BASIC_SEQUENCE_H_

#include <type_traits>

#include "dali/pipeline/basic/tensor.h"

namespace dali {
namespace basic {

template <typename Adapted>
struct SequenceAdapter {
 public:
  SequenceAdapter() = delete;
  static constexpr size_t input_dim = Adapted::input_dim + 1;
  static constexpr size_t output_dim = Adapted::output_dim + 1;
  using InputType = add_dim_t<typename Adapted::InputType>;
  using OutputType = add_dim_t<typename Adapted::OutputType>;

  static constexpr bool can_calculate_output_size = Adapted::can_calculate_output_size;

  using KernelAttributes = typename Adapted::KernelAttributes;

  static typename std::enable_if<Adapted::can_calculate_output_size>::type CalcOutputSize(
      const std::array<Index, input_dim> &in, KernelAttributes attr,
      std::array<Index, output_dim> &out) {  // NOLINT
    // Forward length of sequence
    out[0] = in[0];
    std::array<Index, Adapted::input_dim> adapted_in;
    std::array<Index, Adapted::output_dim> adapted_out;
    for (size_t i = 1; i < input_dim; i++) {
      adapted_in[i - 1] = in[i];
    }
    Adapted::CalcOutputSize(adapted_in, attr, adapted_out);
    for (size_t i = 1; i < output_dim; i++) {
      out[i] = adapted_out[i - 1];
    }
  }

  static void Run(InputType in, KernelAttributes attr, OutputType out) {  // NOLINT
    auto in_seq = SequenceWrapper<typename InputType::type, input_dim>(in);
    auto out_seq = SequenceWrapper<typename OutputType::type, output_dim>(out);
    for (Index i = 0; i < in_seq.sequence_length; i++) {
      Adapted::Run(in_seq.Get(i), attr, out_seq.Get(i));
    }
  }
};

}  // namespace basic
}  // namespace dali

#endif  // DALI_PIPELINE_BASIC_SEQUENCE_H_
