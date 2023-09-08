// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_DALI_TEST_CHECKPOINTING_H_
#define DALI_TEST_DALI_TEST_CHECKPOINTING_H_

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "dali/test/dali_test.h"
#include "dali/pipeline/operator/checkpointing/checkpoint.h"

namespace dali {

/**
 * @brief Pipeline wrapper that allows deep copying in checkpointing tests.
*/
class PipelineWrapper {
 public:
  using OutputsType = std::vector<std::pair<std::string, std::string>>;

  inline PipelineWrapper(int batch_size, OutputsType outputs)
    : batch_size_(batch_size),
      pipe_(new Pipeline(batch_size, 1, 0)),
      outputs_(outputs) {}

  inline void AddOperator(const OpSpec &spec) {
    std::string identifier = std::to_string(next_identifier_++);
    AddOperator(spec, identifier);
  }

  inline void AddOperator(const OpSpec &spec, const std::string &identifier) {
    pipe_->AddOperator(spec, identifier);
    ops_.emplace_back(spec, identifier);
  }

  inline void Build() {
    pipe_->Build(outputs_);
  }

  /**
   * @brief Make a deep copy of a pipeline, transferring state by checkpointing.
   */
  inline PipelineWrapper CopyByCheckpointing() {
    PipelineWrapper clone(batch_size_, outputs_);
    for (const auto &[spec, id] : ops_)
      clone.AddOperator(spec, id);
    clone.Build();

    for (const auto &[spec, id] : ops_) {
      OpCheckpoint cpt(spec);
      // TODO(mstaniewski): provide a stream, so operators with state kept
      // in device memory can be tested.
      GetOperator(id)->SaveState(cpt, std::nullopt);
      clone.GetOperator(id)->RestoreState(cpt);
    }

    return clone;
  }

  template<typename OutputType>
  std::vector<OutputType> RunIteration() {
    pipe_->RunCPU();
    pipe_->RunGPU();
    pipe_->Outputs(&ws_);

    auto collect_value_from_each_sample = [](const TensorList<CPUBackend> &data) {
      std::vector<OutputType> result;
      for (int i = 0; i < data.num_samples(); i++)
        result.push_back(data.tensor<OutputType>(i)[0]);
      return result;
    };

    if (ws_.OutputIsType<CPUBackend>(0)) {
      return collect_value_from_each_sample(ws_.Output<CPUBackend>(0));
    } else {
      TensorList<CPUBackend> cpu;
      cpu.Copy(ws_.Output<GPUBackend>(0));
      return collect_value_from_each_sample(cpu);
    }
  }

 protected:
  OperatorBase *GetOperator(const std::string &name) {
    return pipe_->GetOperatorNode(name)->op.get();
  }

  const int batch_size_;
  int next_identifier_ = 0;
  std::unique_ptr<Pipeline> pipe_;
  std::vector<std::pair<OpSpec, std::string>> ops_;
  OutputsType outputs_;
  Workspace ws_;
};


class CheckpointingTest : public DALITest {
 public:
  template<typename OutputType>
  void RunTest(PipelineWrapper &&original_pipeline, int iterations = 10) {
    for (int i = 0; i < iterations; i++)
      original_pipeline.RunIteration<OutputType>();

    auto restored_pipeline = original_pipeline.CopyByCheckpointing();

    for (int i = 0; i < iterations; i++)
      EXPECT_EQ(original_pipeline.RunIteration<OutputType>(),
                restored_pipeline.RunIteration<OutputType>());
  }
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_CHECKPOINTING_H_
