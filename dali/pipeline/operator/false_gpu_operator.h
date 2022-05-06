// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_FALSE_GPU_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_FALSE_GPU_OPERATOR_H_

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_shape.h"

namespace dali {

/**
 * @brief Produces a false GPU version of a given CPU operator.
 *        The resulting operator copies GPU inputs to host memory before setup, and copies
 *        the CPU outputs to device memory after Run.
 *
 * @tparam CPUOperator CPU operator class
 */
template <typename CPUOperator>
class FalseGPUOperator : public Operator<GPUBackend> {
 public:
  FalseGPUOperator(const OpSpec &spec) : Operator<GPUBackend>(spec), cpu_impl_(spec), thread_pool_(3, 0, false) {
    cpu_ws_.SetThreadPool(&thread_pool_);
  }
  ~FalseGPUOperator() override = default;

 protected:
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<GPUBackend> &ws) override {
    if (cpu_ws_.NumInput() == 0 && cpu_ws_.NumOutput() == 0) {
      cpu_inputs_.resize(ws.NumInput());
      for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
        cpu_inputs_[input_idx] = std::make_shared<TensorVector<CPUBackend>>();
        cpu_ws_.AddInput(cpu_inputs_[input_idx]);
      }
      for (int output_idx = 0; output_idx < ws.NumOutput(); output_idx++) {
        cpu_ws_.AddOutput(std::make_shared<TensorVector<CPUBackend>>());
      }
    } else {
      // Number of inputs/outputs should not change after first iteration
      assert(ws.NumInput() == cpu_ws_.NumInput());
      assert(ws.NumOutput() == cpu_ws_.NumOutput());
    }

    for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
      if (ws.InputIsType<GPUBackend>(0)) {
        auto& gpu_input = ws.Input<GPUBackend>(input_idx);
        cpu_inputs_[input_idx]->Copy(gpu_input, AccessOrder(ws.stream()));
      } else {
        // Some GPU operators might accept CPU inputs (e.g. Slice)
        auto& cpu_input = ws.Input<CPUBackend>(input_idx);
        cpu_inputs_[input_idx]->Copy(cpu_input);
      }
    }

    CUDA_CALL(cudaStreamSynchronize(ws.stream()));

    auto ret = cpu_impl_.Setup(output_desc, cpu_ws_);

    for (int output_idx = 0; output_idx < cpu_ws_.NumOutput(); output_idx++) {
      auto &desc = output_desc[output_idx];
      cpu_ws_.template Output<CPUBackend>(output_idx).Resize(desc.shape, desc.type);
    }
    return ret;
  }

  void RunImpl(workspace_t<GPUBackend> &ws) override {
    cpu_impl_.Run(cpu_ws_);

    for (int output_idx = 0; output_idx < ws.NumOutput(); output_idx++) {
      const auto& cpu_output = cpu_ws_.Output<CPUBackend>(output_idx);
      ws.Output<GPUBackend>(output_idx).Copy(cpu_output, ws.stream());
    }
  }

  USE_OPERATOR_MEMBERS();
  using Operator<GPUBackend>::RunImpl;

 private:
  CPUOperator cpu_impl_;
  ThreadPool thread_pool_;
  HostWorkspace cpu_ws_;

  // Keep it here so that we can modify (ws gives only const ref to inputs)
  std::vector<std::shared_ptr<TensorVector<CPUBackend>>> cpu_inputs_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_FALSE_GPU_OPERATOR_H_
