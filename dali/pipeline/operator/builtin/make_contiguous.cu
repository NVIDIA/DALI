// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operator/builtin/make_contiguous.h"

namespace dali {

void MakeContiguousMixed::Run(MixedWorkspace &ws) {
  const auto& input = ws.Input<CPUBackend>(0, 0);
  int sample_dim = input.shape().sample_dim();
  TensorListShape<> output_shape(batch_size_, sample_dim);
  TensorLayout layout = input.GetLayout();
  TypeInfo type = input.type();
  size_t total_bytes = 0;
  for (int i = 0; i < batch_size_; ++i) {
    auto &sample = ws.Input<CPUBackend>(0, i);
    output_shape.set_tensor_shape(i, sample.shape());
    size_t sample_bytes = sample.nbytes();
    if (coalesced && sample_bytes > COALESCE_TRESHOLD)
      coalesced = false;
    total_bytes += sample_bytes;
    DALI_ENFORCE(type == sample.type(), "Inconsistent types in "
        "input batch. Cannot copy to contiguous device buffer.");
    DALI_ENFORCE(sample_dim == sample.shape().sample_dim(), "Inconsistent sample dimensions "
        "in input batch. Cannot copy to contiguous device buffer.");
  }

  auto &output = ws.Output<GPUBackend>(0);
  output.Resize(output_shape);
  output.SetLayout(layout);
  output.set_type(type);

  if (coalesced) {
    TimeRange tm("coalesced", TimeRange::kBlue);

    if (!cpu_output_buff.capacity()) {
      size_t alloc_size = std::max<size_t>(total_bytes, batch_size_*bytes_per_sample_hint);
      cpu_output_buff.reserve(total_bytes);
    }

    cpu_output_buff.ResizeLike(output);
    cpu_output_buff.SetLayout(layout);
    cpu_output_buff.set_type(type);

    for (int i = 0; i < batch_size_; ++i) {
      auto &input = ws.Input<CPUBackend>(0, i);
      memcpy(cpu_output_buff.raw_mutable_tensor(i), input.raw_data(), input.nbytes());
    }
    CUDA_CALL(cudaMemcpyAsync(
          output.raw_mutable_data(),
          cpu_output_buff.raw_mutable_data(),
          cpu_output_buff.nbytes(),
          cudaMemcpyHostToDevice,
          ws.stream()));
  } else {
    TimeRange tm("non coalesced", TimeRange::kGreen);
    for (int i = 0; i < batch_size_; ++i) {
      auto &input = ws.Input<CPUBackend>(0, i);
      CUDA_CALL(cudaMemcpyAsync(
              output.raw_mutable_tensor(i),
              input.raw_data(),
              input.nbytes(),
              cudaMemcpyHostToDevice,
              ws.stream()));
    }
  }
  coalesced = true;
}

DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguousMixed, Mixed);

}  // namespace dali
