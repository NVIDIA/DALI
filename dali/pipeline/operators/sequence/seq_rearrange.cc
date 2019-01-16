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

#include "dali/pipeline/operators/sequence/seq_rearrange.h"

namespace dali {

// TODO(klecki) ingnoring idx and all stuff from MIS - check if this is the right idx to ignore
template <>
void SequenceRarrange<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(0);
  DALI_ENFORCE(input.ndim() > 1, "Sequence elements must have at least 1 dim");
  const auto input_shape = input.shape();
  const int in_seq_lenght = input_shape[0];
  for (const auto &idx : new_order_) {
    DALI_ENFORCE(0 <= idx && idx < in_seq_lenght,
                 "Source element idx must be in [0, input_sequence_lenght) for new_order argument");
  }
  const int out_seq_lenght = new_order_.size();
  std::vector<Index> element_shape;
  element_shape.insert(element_shape.end(), input_shape.begin() + 1, input_shape.end());

  std::vector<Index> new_sample_shape;
  new_sample_shape.push_back(out_seq_lenght);
  new_sample_shape.insert(new_sample_shape.end(), element_shape.begin(), element_shape.end());

  auto *output = ws->Output<CPUBackend>(0);
  output->set_type(input.type());
  output->Resize(new_sample_shape);

  Index element_size = Product(element_shape);
  Index element_bytes = element_size * input.type().size();

  for (int i = 0; i < out_seq_lenght; i++) {
    TypeInfo type = input.type();
    type.Copy<CPUBackend, CPUBackend>(output->mutable_data<char>() + i * element_bytes,
                                      input.data<char>() + new_order_[i] * element_bytes,
                                      element_size, 0);
  }
}

DALI_REGISTER_OPERATOR(SequenceRarrange, SequenceRarrange<CPUBackend>, CPU);

DALI_SCHEMA(SequenceRarrange)
    .DocStr("Rearrange the sequence stored as tensor.")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("new_order", R"code(List describing new order for elements of each sample.
Output sequence at position `i` will contain element `new_order[i]` from input sequence.
Elements can be repeated or dropped, only indices in [0, input_outermost_dimension) are allowed
to be used in `new_order`.)code",
            DALI_INT_VEC);

}  // namespace dali