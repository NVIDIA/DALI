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

#ifndef DALI_TEST_OPERATORS_PASSTHROUGH_INPUT_H_
#define DALI_TEST_OPERATORS_PASSTHROUGH_INPUT_H_

#include <vector>

#include "dali/pipeline/operator/builtin/input_operator.h"

namespace dali {


template <typename Backend>
class PassthroughInput : public InputOperator<Backend> {
  using OutBackend = std::conditional_t<
          std::is_same_v<Backend, CPUBackend>,
          CPUBackend /* CPUBackend */,
          GPUBackend /* GPUBackend or MixedBackend */
  >;

 public:
  explicit PassthroughInput(const OpSpec &spec) : InputOperator<Backend>(spec) {}

  DISABLE_COPY_MOVE_ASSIGN(PassthroughInput);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    InputOperator<Backend>::HandleDataAvailability();
    TensorListShape<> shape;
    output_desc.resize(1);
    output_desc[0].shape = InputOperator<Backend>::PeekCurrentData().shape();
    output_desc[0].type = InputOperator<Backend>::PeekCurrentData().type();
    return false;
  }


  void Run(Workspace &ws) override {
    auto& out = ws.Output<OutBackend>(0);
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      this->ForwardCurrentData(out, ws.GetThreadPool());
    } else {
      this->ForwardCurrentData(out, ws.stream());
    }
  }
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_PASSTHROUGH_INPUT_H_
