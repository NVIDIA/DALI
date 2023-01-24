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


template<typename Backend>
class PassthroughInput : public InputOperator<Backend> {
  using OutBackend = std::conditional_t<
          std::is_same_v<Backend, CPUBackend>,
          CPUBackend /* CPUBackend */,
          GPUBackend /* GPUBackend or MixedBackend */
  >;

 public:
  explicit PassthroughInput(const OpSpec &spec) :
          InputOperator<Backend>(spec),
          cpu_input_(spec.GetArgument<bool>("cpu_input")) {
    if constexpr (std::is_same_v<Backend, MixedBackend>) {
      tp_ = std::make_unique<ThreadPool>(this->num_threads_, this->device_id_, false,
                                         "PassthroughInput thread pool");
    }
  }

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
    DALI_ENFORCE(!cpu_input_ || ((cpu_input_) != (std::is_same_v<Backend, GPUBackend>)),
                 "Can't have CPU input in the GPU operator.");
    if (cpu_input_) {
      RunCpuInput(ws);
    } else {
      RunGpuInput(ws);
    }
  }


  void RunCpuInput(Workspace &ws) {
    auto &out = ws.Output<OutBackend>(0);
    TensorList<CPUBackend> intermediate;
    this->ForwardCurrentData(intermediate,
                             std::is_same_v<Backend, CPUBackend> ? ws.GetThreadPool() : *tp_);
    out.Copy(intermediate, ws.stream());
  }


  void RunGpuInput(Workspace &ws) {
    auto &out = ws.Output<OutBackend>(0);
    this->ForwardCurrentData(out, ws.stream());
  }


  bool cpu_input_;
  std::unique_ptr<ThreadPool> tp_;
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_PASSTHROUGH_INPUT_H_
