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

#include "dali/pipeline/operator/checkpointing/checkpoint.h"

#include <gtest/gtest.h>

#include "dali/test/dali_test.h"

namespace dali {

class PipelineWrapper {
 public:
  using OutputsType = std::vector<std::pair<std::string, std::string>>;

  inline PipelineWrapper(int batch_size, OutputsType outputs)
    : batch_size_(batch_size),
      pipe_(new Pipeline(batch_size, 1, 0)),
      outputs_(outputs) {}

  inline void AddOperator(OpSpec spec) {
    std::string identifier = std::to_string(next_identifier_++);
    AddOperator(spec, identifier);
  }

  inline void AddOperator(OpSpec spec, std::string identifier) {
    pipe_->AddOperator(spec, identifier);
    ops_.emplace_back(spec, identifier);
  }

  inline void Build() {
    pipe_->Build(outputs_);
  }

  inline PipelineWrapper CopyByCheckpointing() {
    PipelineWrapper clone(batch_size_, outputs_);
    for (const auto &[spec, id] : ops_)
      clone.AddOperator(spec, id);
    clone.Build();

    for (const auto &[spec, id] : ops_) {
      // Only save/restore state for operators with checkpointing enabled.
      bool checkpointing;
      if (!spec.TryGetArgument(checkpointing, "checkpointing") || !checkpointing)
        continue;

      OpCheckpoint cpt(spec);
      GetOperator(id)->SaveState(cpt, std::nullopt);
      clone.GetOperator(id)->RestoreState(cpt);
    }

    return clone;
  }

  template<class OutputType>
  std::vector<OutputType> RunIteration() {
    pipe_->RunCPU();
    pipe_->RunGPU();
    pipe_->Outputs(&ws_);

    // read a single value from each sample
    std::vector<OutputType> result;
    const auto batch_size = ws_.Output<CPUBackend>(0).AsTensor().shape()[0];
    for (int i = 0; i < batch_size; i++)
      result.push_back(ws_.Output<CPUBackend>(0).tensor<OutputType>(0)[i]);

    return result;
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
  void RunTest(PipelineWrapper original_pipeline, int iterations) {
    for (int i = 0; i < iterations; i++)
      original_pipeline.RunIteration<OutputType>();

    auto restored_pipeline = original_pipeline.CopyByCheckpointing();

    for (int i = 0; i < iterations; i++)
      EXPECT_EQ(original_pipeline.RunIteration<OutputType>(),
                restored_pipeline.RunIteration<OutputType>());
  }
};

}  // namespace dali
