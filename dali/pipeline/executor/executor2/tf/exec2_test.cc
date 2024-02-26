// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include "dali/pipeline/operator/arg_helper.h"
#include "exec_graph.h"

namespace dali {
namespace exec2 {
namespace test {

inline OpSchema &CreateTestSchema(const std::string &name) {
  OpSchema &s = SchemaRegistry::RegisterSchema(name);
  s = OpSchema(name);  // reset
  return s;
}

class DummyOp : public Operator<CPUBackend> {
  DummyOp(const OpSpec &spec) : Operator<CPUBackend>(spec) {
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    outs[0].shape = uniform_list_shape(N, TensorShape<>{});
    outs[0].type = DALI_INT32;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    addend_.Acquire(spec_, ws, N);
    for (int s = 0; s < N; s++) {
      int sum = *addend_[s].data;
      for (int i = 0; i < ws.NumInput(); i++) {
        s += *ws.Input<CPUBackend>(i)[s].data<int>();
      }
      *ws.Output<CPUBackend>(0)[s].mutable_data<int>() = s;
    }
  }

  bool CanInferOutputs() const override { return true; }
  ArgValue<int> addend_{"addend", spec_};
};

TEST(Exec2Test, SimpleGraph) {
  CreateTestSchema("DummyOp")
    .NumInput(1, 99)
    .NumOutput(1)
    .AddArg("addend", "a value added to the sum of inputs", DALI_INT32);
}

}  // namespace test
}  // namespace exec2
}  // namespace dali