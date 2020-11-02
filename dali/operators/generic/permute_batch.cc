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
#include "dali/operators/generic/permute_batch.h"

namespace dali {

DALI_SCHEMA(PermuteBatch)
  .DocStr(R"(Returns a batch of tensors constructed by selecting tensors from the input based
on indices given in ``indices`` argument::

  out_tensor[i] = in_tensor[indices[i]]

)")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("indices", R"(List of indices, matching current batch size, or a batch
of scalars representing indices of the tensors in the input batch.

The indices must be within ``[0..batch_size)`` range. Repetitions and omissions are allowed.)",
    DALI_INT_VEC, true);

void PermuteBatch<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  const auto &output_shape = output.shape();
  output.SetLayout(input.GetLayout());

  auto &tp = ws.GetThreadPool();
  int N = indices_.size();
  for (int i = 0; i < N; i++) {
    auto size = output_shape.tensor_size(i);
    int src = indices_[i];
    tp.AddWork([&, i, src](int tid) {
      output.SetMeta(i, input.GetMeta(i));
      output[i].Copy(input[src], 0);
    }, size);
  }
  tp.RunAll();
}

void PermuteBatch<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  auto &input = ws.InputRef<GPUBackend>(0);
  auto &output = ws.OutputRef<GPUBackend>(0);

  output.SetLayout(input.GetLayout());
  int N = indices_.size();
  for (int i = 0; i < N; i++)
    output.SetMeta(i, input.GetMeta(indices_[i]));

  const auto &out_shape = output.shape();
  int element_size = output.type().size();

  for (int i = 0; i < N; i++) {
    auto size = out_shape.tensor_size(i) * element_size;
    sg_.AddCopy(output.raw_mutable_tensor(i), input.raw_tensor(indices_[i]), size);
  }
  sg_.Run(ws.stream());
}

DALI_REGISTER_OPERATOR(PermuteBatch, PermuteBatch<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(PermuteBatch, PermuteBatch<GPUBackend>, GPU);

}  // namespace dali
