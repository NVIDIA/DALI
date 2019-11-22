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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_

#include <algorithm>
#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/core/common.h"

// Found by benchmarking coalesced vs non coalesced on diff size images
#define COALESCE_TRESHOLD 8192

namespace dali {

class MakeContiguous : public Operator<MixedBackend> {
 public:
  inline explicit MakeContiguous(const OpSpec &spec) :
      Operator<MixedBackend>(spec),
      coalesced(true) {
    std::vector<int> hints;
    GetSingleOrRepeatedArg(spec, hints, "bytes_per_sample_hint", spec.NumOutput());
    if (!hints.empty())
      bytes_per_sample_hint = hints[0];
    CUDA_CALL(cudaGetDevice(&event_dev_));
    CUDA_CALL(cudaEventCreateWithFlags(&coalesce_event_, cudaEventDisableTiming));
  }

  virtual inline ~MakeContiguous() {
    DeviceGuard g(event_dev_);
    CUDA_CALL(cudaEventSynchronize(coalesce_event_));
    CUDA_CALL(cudaEventDestroy(coalesce_event_));
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const MixedWorkspace &ws) override {
    return false;
  }

  using Operator<MixedBackend>::Run;
  void Run(MixedWorkspace &ws) override {
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

    if (ws.OutputIsType<CPUBackend>(0)) {
      auto &output = ws.Output<CPUBackend>(0);
      output.Resize(output_shape);
      output.SetLayout(layout);
      output.set_type(type);

      for (int i = 0; i < batch_size_; ++i) {
        auto &input = ws.Input<CPUBackend>(0, i);

        // Note: We know that this will translate into
        // a std::memcpy, so it is safe to pass stream 0
        type.Copy<CPUBackend, CPUBackend>(
            output.raw_mutable_tensor(i),
            input.raw_data(), input.size(), 0);
      }
    } else {
      auto &output = ws.Output<GPUBackend>(0);
      output.Resize(output_shape);
      output.SetLayout(layout);
      output.set_type(type);

      if (coalesced) {
        TimeRange tm("coalesced", TimeRange::kBlue);
        // We need to wait for the async memcpy on the cpu_output_buff to finish
        // before we can start manipulating it again
        CUDA_CALL(cudaEventSynchronize(coalesce_event_));

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
        // In case we will want to modify the coalesced buffer, we will wait for this event
        // for async work to finish.
        CUDA_CALL(cudaEventRecord(coalesce_event_, ws.stream()));
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
    }
    coalesced = true;
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguous);

 protected:
  USE_OPERATOR_MEMBERS();
  TensorList<CPUBackend> cpu_output_buff;
  bool coalesced;
  int bytes_per_sample_hint;
  int event_dev_;
  cudaEvent_t coalesce_event_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_
