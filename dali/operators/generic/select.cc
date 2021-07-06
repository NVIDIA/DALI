// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include "dali/operators/generic/select.h"

namespace dali {

DALI_SCHEMA(Select)
  .DocStr(R"(Builds a batch by selecting each sample from one of the input batches.

This operator is useful for conditionally selecting results of different operations.
The shapes of the corresponding samples in the inputs may differ, but the number of dimensions
and data type of all the inputs must be the same.)")
  .NumInput(1, 99)
  .NumOutput(1)
  .AllowSequences()
  .SupportVolumetric()
  .AddArg("input_idx", R"(Index of the input to take the sample from.

This argument contains (per-sample) indices that indicate from which input each
sample is taken.
Using a negative index will produce an empty tensor with the same number of dimensions as
the inputs. Negative indices cannot be used with batches of scalars (0D tensors) since they
can never be empty.
)", DALI_INT32, true)
  .AddOptionalArg<TensorLayout>("layout", R"(Layot string for the output.

If not specified, the input layouts are checked and the first non-empty is used.)", nullptr);

template <>
void Select<CPUBackend>::RunImpl(HostWorkspace &ws) {
  SetOutputLayout(ws);
  TensorVector<CPUBackend> &out = ws.OutputRef<CPUBackend>(0);
  const auto &out_shape = out.shape();
  int N = out_shape.num_samples();
  int64_t element_size = out.type().size();
  int64_t total_size = out_shape.num_elements() * element_size;
  const int64_t min_size = 16 << 10;

  ThreadPool &tp = ws.GetThreadPool();
  int min_blocks = tp.NumThreads() * 10;
  int64_t block_size = std::max(total_size / min_size, min_size);
  for (int i = 0; i < N; i++) {
    auto sample_size = out_shape.tensor_size(i) * element_size;
    if (sample_size == 0)
      continue;
    int idx = *input_idx_[i].data;
    auto &inp = ws.InputRef<CPUBackend>(idx);
    assert(idx >= 0 && idx < spec_.NumRegularInput());
    int blocks = div_ceil(sample_size, block_size);
    char *out_ptr = static_cast<char*>(out.raw_mutable_tensor(i));
    const char *in_ptr = static_cast<const char*>(inp.raw_tensor(i));
    ptrdiff_t start = 0;
    for (int block = 0; block < blocks; block++) {
      ptrdiff_t end = sample_size * (block + 1) / blocks;
      tp.AddWork([in_ptr, out_ptr, start, end](int) {
        memcpy(out_ptr + start, in_ptr + start, end - start);
      }, end - start);
      start = end;
    }
  }
  tp.RunAll();
}

template <>
void Select<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  SetOutputLayout(ws);
  TensorList<GPUBackend> &out = ws.OutputRef<GPUBackend>(0);
  const auto &out_shape = out.shape();
  int N = out_shape.num_samples();
  int64_t element_size = out.type().size();
  for (int i = 0; i < N; i++) {
    auto sample_size = out_shape.tensor_size(i) * element_size;
    if (sample_size == 0)
      continue;
    int idx = *input_idx_[i].data;
    auto &inp = ws.InputRef<GPUBackend>(idx);
    assert(idx >= 0 && idx < spec_.NumRegularInput());
    sg_.AddCopy(out.raw_mutable_tensor(i), inp.raw_tensor(i), sample_size);
  }
  sg_.Run(ws.stream());
}

DALI_REGISTER_OPERATOR(Select, Select<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Select, Select<GPUBackend>, GPU);

}  // namespace namespace dali
