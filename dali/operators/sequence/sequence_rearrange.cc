// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>

#include "dali/operators/sequence/sequence_rearrange.h"

namespace dali {

DALI_SCHEMA(SequenceRearrange)
    .DocStr(R"code(Rearranges frames in a sequence.

Assumes that the outermost dimension represents the frame index in the sequence.
If the input has a non-empty layout description, it must start with ``F`` (frame).)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .AddArg("new_order", R"code(List that describes the new order for the elements in each sample.

Output sequence at position ``i`` will contain element ``new_order[i]`` from input sequence::

  out[i, ...] = in[new_order[i], ...]

Elements can be repeated or dropped, but empty output sequences are not allowed. Only indices in
``[0, input_outermost_extent)`` are allowed to be used in ``new_order``. Can be specified
per sample as 1D tensors.)code",
            DALI_INT_VEC, true);

void ValidateSeqRearrange(const TensorShape<> &in_sample_shape,
                          const TensorView<StorageCPU, const int, 1> &new_order, int sample_idx) {
  const int in_seq_length = GetSeqLength(in_sample_shape);
  const int out_seq_length = new_order.num_elements();
  DALI_ENFORCE(out_seq_length > 0,
               make_string("Empty `new_order` for sample ", sample_idx, " is not allowed."));
  for (int i = 0; i < out_seq_length; i++) {
    int src_idx = new_order.data[i];
    DALI_ENFORCE(
        0 <= src_idx && src_idx < in_seq_length,
        make_string("new_order[", i, "] must be between 0 and input_sequence_length = ",
                    in_seq_length, " for sample ", sample_idx, ", but it is: ", src_idx, "."));
  }
}

TensorShape<> GetSeqRearrangedShape(const TensorShape<> &in_sample_shape,
                                    const TensorView<StorageCPU, const int, 1> &new_order) {
  TensorShape<> result = in_sample_shape;
  result[0] = new_order.num_elements();
  return result;
}

copy_desc GetCopyDesc(char *output_sample, const char *input_sample, int out_elem_idx,
                      int in_elem_idx, int64_t element_sizeof) {
  copy_desc result;
  result.from = input_sample + in_elem_idx * element_sizeof;
  result.to = output_sample + out_elem_idx * element_sizeof;
  result.size = element_sizeof;
  return result;
}

template <>
void SequenceRearrange<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &thread_pool = ws.GetThreadPool();
  auto out_shape = output.shape();
  auto curr_batch_size = ws.GetInputBatchSize(0);

  for (int sample_idx = 0; sample_idx < curr_batch_size; ++sample_idx) {
    thread_pool.AddWork([this, &ws, &input, &output, sample_idx](int tid) {
      TypeInfo type = input.type();
      const auto *in_sample = reinterpret_cast<const char *>(input[sample_idx].raw_data());
      auto *out_sample = reinterpret_cast<char *>(output[sample_idx].raw_mutable_data());
      const auto &in_shape = input[sample_idx].shape();
      auto element_sizeof = volume(in_shape.last(in_shape.sample_dim() - 1)) * type.size();

      TensorView<StorageCPU, const int, 1> new_order = {};

      if (single_order_) {
        new_order = TensorView<StorageCPU, const int, 1>(new_order_.data(),
                                                         TensorShape<1>(new_order_.size()));
      } else {
        auto &new_orders = ws.ArgumentInput("new_order");
        new_order = view<const int, 1>(new_orders[sample_idx]);
      }
      for (int i = 0; i < new_order.shape.num_elements(); i++) {
        auto copy_desc = GetCopyDesc(out_sample, in_sample, i, new_order.data[i], element_sizeof);
        memcpy(copy_desc.to, copy_desc.from, copy_desc.size);
      }
    }, out_shape.tensor_size(sample_idx));
  }
  thread_pool.RunAll();

  output.SetLayout(input.GetLayout());
}

DALI_REGISTER_OPERATOR(SequenceRearrange, SequenceRearrange<CPUBackend>, CPU);

}  // namespace dali
