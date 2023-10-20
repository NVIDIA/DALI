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

#ifndef DALI_OPERATORS_GENERIC_FLIP_H_
#define DALI_OPERATORS_GENERIC_FLIP_H_

#include <vector>
#include <string>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {

template <typename Backend>
class Flip: public StatelessOperator<Backend> {
 public:
  explicit Flip(const OpSpec &spec);

  ~Flip() override = default;
  DISABLE_COPY_MOVE_ASSIGN(Flip);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    auto &input = ws.Input<Backend>(0);
    output_desc[0].type =  input.type();
    output_desc[0].shape = input.shape();
    return true;
  }

  bool CanInferOutputs() const override {
    return true;
  }

  void RunImpl(legacy_workspace_t<Backend> &ws) override;

  int GetHorizontal(const ArgumentWorkspace &ws, int idx) {
    return this->spec_.template GetArgument<int>("horizontal", &ws, idx);
  }

  int GetVertical(const ArgumentWorkspace &ws, int idx) {
    return this->spec_.template GetArgument<int>("vertical", &ws, idx);
  }

  int GetDepthwise(const ArgumentWorkspace &ws, int idx) {
    return this->spec_.template GetArgument<int>("depthwise", &ws, idx);
  }

  std::vector<int> GetHorizontal(const Workspace &ws, int curr_batch_size) {
    std::vector<int> result;
    OperatorBase::GetPerSampleArgument(result, "horizontal", ws, curr_batch_size);
    return result;
  }

  std::vector<int> GetVertical(const Workspace &ws, int curr_batch_size) {
    std::vector<int> result;
    OperatorBase::GetPerSampleArgument(result, "vertical", ws, curr_batch_size);
    return result;
  }

  std::vector<int> GetDepthwise(const Workspace &ws, int curr_batch_size) {
    std::vector<int> result;
    OperatorBase::GetPerSampleArgument(result, "depthwise", ws, curr_batch_size);
    return result;
  }

 private:
  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_FLIP_H_
