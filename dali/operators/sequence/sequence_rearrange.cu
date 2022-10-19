// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>

#include "dali/kernels/common/scatter_gather.h"
#include "dali/operators/sequence/sequence_rearrange.h"

namespace dali {

template <>
void SequenceRearrange<GPUBackend>::RunImpl(Workspace &ws) {
  scatter_gather_.Reset();
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  const TypeInfo &type = input.type_info();
  auto curr_batch_size = ws.GetInputBatchSize(0);

  for (int sample_idx = 0; sample_idx < curr_batch_size; ++sample_idx) {
    auto *out_sample = reinterpret_cast<char *>(output.raw_mutable_tensor(sample_idx));
    const auto *in_sample = reinterpret_cast<const char *>(input.raw_tensor(sample_idx));
    const auto &in_shape = input.shape()[sample_idx];
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
      scatter_gather_.AddCopy(copy_desc.to, copy_desc.from, copy_desc.size);
    }
  }
  scatter_gather_.Run(ws.stream());
  output.SetLayout(input.GetLayout());
}

DALI_REGISTER_OPERATOR(SequenceRearrange, SequenceRearrange<GPUBackend>, GPU);

}  // namespace dali
