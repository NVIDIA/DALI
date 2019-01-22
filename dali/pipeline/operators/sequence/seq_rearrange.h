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

#ifndef DALI_PIPELINE_OPERATORS_SEQUENCE_SEQ_REARRANGE_H_
#define DALI_PIPELINE_OPERATORS_SEQUENCE_SEQ_REARRANGE_H_

#include <cstring>

#include "dali/pipeline/operators/operator.h"

namespace dali {

static std::vector<Index> GetElementShape(const std::vector<Index>& in_sample_shape) {
  std::vector<Index> element_shape;
  element_shape.insert(element_shape.end(), in_sample_shape.begin() + 1, in_sample_shape.end());
  return element_shape;
}

static Index GetSeqLength(const std::vector<Index>& seq_shape) {
  return seq_shape[0];
}

static std::tuple<std::vector<Index>, Index> GetNewShapeAndElementSize(
    const std::vector<Index>& in_sample_shape, const std::vector<int>& new_order) {
  auto element_shape = GetElementShape(in_sample_shape);
  const int in_seq_length = GetSeqLength(in_sample_shape);
  for (const auto& src_idx : new_order) {
    DALI_ENFORCE(
        0 <= src_idx && src_idx < in_seq_length,
        "Source element src_idx must be in [0, input_sequence_length) for new_order argument");
  }
  Index element_size = Product(element_shape);

  const int out_seq_length = new_order.size();
  std::vector<Index> new_sample_shape;
  new_sample_shape.push_back(out_seq_length);
  new_sample_shape.insert(new_sample_shape.end(), element_shape.begin(), element_shape.end());
  return std::tuple<std::vector<Index>, Index>{new_sample_shape, element_size};
}

template <typename Backend>
class SequenceRearrange : public Operator<Backend> {
 public:
  inline explicit SequenceRearrange(const OpSpec& spec)
      : Operator<Backend>(spec), new_order_(spec.GetRepeatedArgument<int>("new_order")) {
    DALI_ENFORCE(new_order_.size() > 0, "Empty result sequence not allowed");
  }

  inline ~SequenceRearrange() override = default;

  DISABLE_COPY_MOVE_ASSIGN(SequenceRearrange);

 protected:
  void RunImpl(Workspace<Backend>* ws, const int idx) override;

 private:
  std::vector<int> new_order_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_SEQUENCE_SEQ_REARRANGE_H_
