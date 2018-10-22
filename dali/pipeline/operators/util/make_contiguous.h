// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_MAKE_CONTIGUOUS_H_
#define DALI_PIPELINE_OPERATORS_UTIL_MAKE_CONTIGUOUS_H_

#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/common.h"

// Found by benchmarking coalesced vs non coalesced on diff size images
#define COALESCE_TRESHOLD 8192

namespace dali {

class MakeContiguous : public Operator<MixedBackend> {
 public:
  inline explicit MakeContiguous(const OpSpec &spec) :
    Operator<MixedBackend>(spec),
    coalesced(true)
    {}

  virtual inline ~MakeContiguous() = default;

  using Operator<MixedBackend>::Run;
  void Run(MixedWorkspace *ws) override {
    vector<Dims> output_shape(batch_size_);
    TypeInfo type = ws->Input<CPUBackend>(0, 0).type();
    for (int i = 0; i < batch_size_; ++i) {
      const auto &input = ws->Input<CPUBackend>(0, i);
      output_shape[i] = input.shape();
      if (coalesced && input.nbytes() > COALESCE_TRESHOLD)
        coalesced = false;
      DALI_ENFORCE(type == input.type(), "Inconsistent types in "
          "input batch. Cannot copy to contiguous device buffer.");
    }

    if (ws->OutputIsType<CPUBackend>(0)) {
      auto output = ws->Output<CPUBackend>(0);
      output->Resize(output_shape);
      output->set_type(type);

      for (int i = 0; i < batch_size_; ++i) {
        const auto &input = ws->Input<CPUBackend>(0, i);
        if (!i)
          output->SetLayout(input.GetLayout());

        // Note: We know that this will translate into
        // a std::memcpy, so it is safe to pass stream 0
        type.Copy<CPUBackend, CPUBackend>(
            output->raw_mutable_tensor(i),
            input.raw_data(), input.size(), 0);
      }
    } else {
      auto output = ws->Output<GPUBackend>(0);
      output->Resize(output_shape);
      output->set_type(type);

      if (coalesced) {
        TimeRange tm("coalesced", TimeRange::kBlue);
        cpu_output_buff.CopyAttributes(*output);
        for (int i = 0; i < batch_size_; ++i) {
          auto &input = ws->Input<CPUBackend>(0, i);
          memcpy(cpu_output_buff.raw_mutable_tensor(i), input.raw_data(), input.nbytes());
        }
        CUDA_CALL(cudaMemcpyAsync(
              output->raw_mutable_data(),
              cpu_output_buff.raw_mutable_data(),
              cpu_output_buff.nbytes(),
              cudaMemcpyHostToDevice,
              ws->stream()));
      } else {
        TimeRange tm("non coalesced", TimeRange::kGreen);
        for (int i = 0; i < batch_size_; ++i) {
          auto &input = ws->Input<CPUBackend>(0, i);
          CUDA_CALL(cudaMemcpyAsync(
                  output->raw_mutable_tensor(i),
                  input.raw_data(),
                  input.nbytes(),
                  cudaMemcpyHostToDevice,
                  ws->stream()));
        }
      }
    }
    coalesced = true;
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguous);

 protected:
  USE_OPERATOR_MEMBERS();
  TensorList<CPUBackend> cpu_output_buff;
  bool coalesced;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_MAKE_CONTIGUOUS_H_
