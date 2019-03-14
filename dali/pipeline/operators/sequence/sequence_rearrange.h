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

#ifndef DALI_PIPELINE_OPERATORS_SEQUENCE_SEQUENCE_REARRANGE_H_
#define DALI_PIPELINE_OPERATORS_SEQUENCE_SEQUENCE_REARRANGE_H_

#include <tuple>
#include <vector>

#include "dali/pipeline/operators/operator.h"

namespace dali {

inline Index GetSeqLength(const std::vector<Index>& seq_shape) {
  return seq_shape[0];
}

inline std::tuple<std::vector<Index>, Index> GetNewShapeAndElementSize(
    const std::vector<Index>& in_sample_shape, const std::vector<int>& new_order) {
  const int in_seq_length = GetSeqLength(in_sample_shape);
  for (const auto& src_idx : new_order) {
    DALI_ENFORCE(0 <= src_idx && src_idx < in_seq_length,
                 "Source element src_idx must be in [0, input_sequence_length = " +
                     std::to_string(in_seq_length) +
                     ") for new_order argument, but it is: " + std::to_string(src_idx));
  }
  Index element_size = volume(in_sample_shape) / GetSeqLength(in_sample_shape);

  const int out_seq_length = new_order.size();
  auto new_sample_shape = in_sample_shape;
  new_sample_shape[0] = out_seq_length;
  return std::tuple<std::vector<Index>, Index>{new_sample_shape, element_size};
}

template <typename Backend>
class SequenceRearrange : public Operator<Backend> {
 public:
  inline explicit SequenceRearrange(const OpSpec& spec)
      : Operator<Backend>(spec), new_order_(spec.GetRepeatedArgument<int>("new_order")) {
    DALI_ENFORCE(!new_order_.empty(), "Empty result sequence not allowed");
  }

  DISABLE_COPY_MOVE_ASSIGN(SequenceRearrange);

 protected:
  void RunImpl(Workspace<Backend>* ws, const int idx) override;

 private:
  std::vector<int> new_order_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_SEQUENCE_SEQUENCE_REARRANGE_H_
