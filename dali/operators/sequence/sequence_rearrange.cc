// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

template <>
void SequenceRearrange<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  auto &thread_pool = ws.GetThreadPool();

  for (int sample_idx = 0; sample_idx < batch_size_; ++sample_idx) {
    thread_pool.DoWorkWithID([this, &ws, sample_idx](int tid) {
      const auto &input = ws.InputRef<CPUBackend>(0);
      auto &output = ws.OutputRef<CPUBackend>(0);
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
    });
  }
  thread_pool.WaitForWork();
}

DALI_REGISTER_OPERATOR(SequenceRearrange, SequenceRearrange<CPUBackend>, CPU);

DALI_SCHEMA(SequenceRearrange)
    .DocStr("Rearrange the sequence stored as tensor.")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("new_order", R"code(List describing new order for elements of each sample.
Output sequence at position ``i`` will contain element ``new_order[i]`` from input sequence.
Elements can be repeated or dropped, only indices in [0, input_outermost_dimension) are allowed
to be used in `new_order`. Can be specified per sample as 1D tensors.)code",
            DALI_INT_VEC, true);

}  // namespace dali
