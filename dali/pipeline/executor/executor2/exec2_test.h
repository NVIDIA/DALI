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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_TEST_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_TEST_H_

#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace exec2 {
namespace test {

constexpr char kTestOpName[] = "Exec2TestOp";

class DummyOpCPU : public Operator<CPUBackend> {
 public:
  explicit DummyOpCPU(const OpSpec &spec) : Operator<CPUBackend>(spec) {
    instance_name_ = spec_.GetArgument<string>("name");
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
      int sum = *addend_[s].data + s;
      for (int i = 0; i < ws.NumInput(); i++) {
        sum += *ws.Input<CPUBackend>(i)[s].data<int>();
      }
      *ws.Output<CPUBackend>(0)[s].mutable_data<int>() = sum;
    }
  }

  bool CanInferOutputs() const override { return true; }
  ArgValue<int> addend_{"addend", spec_};

  std::string instance_name_;
};

constexpr char kCounterOpName[] = "Exec2Counter";

class CounterOp : public Operator<CPUBackend> {
 public:
  explicit CounterOp(const OpSpec &spec) : Operator<CPUBackend>(spec) {
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    outs[0].shape = uniform_list_shape(N, TensorShape<>{});
    outs[0].type = DALI_INT32;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    for (int s = 0; s < N; s++) {
      *ws.Output<CPUBackend>(0)[s].mutable_data<int>() = counter++;
    }
  }

  bool CanInferOutputs() const override { return true; }

  int counter = 0;
};

inline auto GetTestGraph1() {
  auto spec0 = OpSpec(kTestOpName)
    .AddArg("max_batch_size", 32)
    .AddArg("device", "cpu")
    .AddArg("num_threads", 1)
    .AddArg("name", "op0")
    .AddOutput("op0_0", "cpu")
    .AddArg("addend", 10);
  auto spec1 = OpSpec(kTestOpName)
    .AddArg("max_batch_size", 32)
    .AddArg("device", "cpu")
    .AddArg("num_threads", 1)
    .AddArg("name", "op1")
    .AddArg("addend", 20)
    .AddOutput("op1_0", "cpu");
  auto spec2 = OpSpec(kTestOpName)
    .AddArg("max_batch_size", 32)
    .AddArg("device", "cpu")
    .AddInput("op0_0", "cpu")
    .AddArg("num_threads", 1)
    .AddArg("name", "op2")
    .AddArgumentInput("addend", "op1_0")
    .AddOutput("op2_0", "cpu");
  auto spec3 = OpSpec(kTestOpName)
    .AddArg("max_batch_size", 32)
    .AddArg("device", "cpu")
    .AddArg("num_threads", 1)
    .AddArg("name", "op3")
    .AddInput("op0_0", "cpu")
    .AddInput("op1_0", "cpu")
    .AddArg("addend", 1)
    .AddOutput("op3_0", "cpu");
  graph::OpGraph::Builder b;
  b.Add("op0", std::move(spec0));
  b.Add("op1", std::move(spec1));
  b.Add("op2", std::move(spec2));
  b.Add("op3", std::move(spec3));
  b.AddOutput("op3_0_cpu");
  b.AddOutput("op2_0_cpu");
  return std::move(b).GetGraph(true);
}

inline size_t CountOutgoingEdges(const graph::OpNode &op, bool include_outputs = true) {
  size_t n = 0;
  for (auto &out : op.outputs) {
    n += out->consumers.size();
    if (out->pipeline_output && include_outputs)
      n++;
  }
  return n;
}

}  // namespace test
}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_TEST_H_
