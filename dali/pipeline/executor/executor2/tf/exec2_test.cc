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
  const OpSchema *psc = SchemaRegistry::TryGetSchema(name);
  OpSchema &s = psc ? const_cast<OpSchema &>(*psc) : SchemaRegistry::RegisterSchema(name);
  s = OpSchema(name);  // reset
  return s;
}

class DummyOp : public Operator<CPUBackend> {
 public:
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

  static OpSchema &CreateSchema() {
    return CreateTestSchema("DummyOp")
      .NumInput(0, 99)
      .NumOutput(1)
      .AddArg("addend", "a value added to the sum of inputs", DALI_INT32);
  }
};

TEST(Exec2Test, SimpleGraph) {
  DummyOp::CreateSchema();
  OpSpec spec0("DummyOp");
  spec0.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op0o0", "cpu")
       .AddArg("name", "op0");
  DummyOp op0(spec0);

  OpSpec spec1("DummyOp");
  spec1.AddArg("addend", 200)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op1o0", "cpu")
       .AddArg("name", "op1");
  DummyOp op1(spec1);

  OpSpec spec2("DummyOp");
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("name", "op2");
  DummyOp op2(spec2);
  ExecGraph def;
  ExecNode *n0 = def.add_node(&op0);
  ExecNode *n1 = def.add_node(&op1);
  ExecNode *n2 = def.add_node(&op2);
  def.link(n0, 0, n2, 0);
  def.link(n1, 0, n2, 1);
  def.link(n2, 0, nullptr, 0);
  auto sched = SchedGraph::from_def(def);
  tf::Taskflow tf;
  sched->schedule(tf);
  tf::Executor ex;
  ex.run(tf);
  ex.wait_for_all();
}


}  // namespace test
}  // namespace exec2
}  // namespace dali