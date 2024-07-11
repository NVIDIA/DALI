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

class DummyOpGPU : public Operator<GPUBackend> {
 public:
  explicit DummyOpGPU(const OpSpec &spec) : Operator<GPUBackend>(spec) {
    instance_name_ = spec_.GetArgument<string>("name");
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    int N = ws.GetRequestedBatchSize(0);
    outs[0].shape = uniform_list_shape(N, TensorShape<>{});
    outs[0].type = DALI_INT32;
    return true;
  }

  void RunImpl(Workspace &ws) override;

  bool CanInferOutputs() const override { return true; }

 private:
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

inline auto &AddCommonArgs(
    OpSpec &spec, int max_batch_size, const std::string &device = "cpu", int num_threads = 1) {
  spec.AddArg("max_batch_size", max_batch_size);
  spec.AddArg("device", device);
  spec.AddArg("num_threads", num_threads);
  return spec;
}

inline auto GetTestGraph1() {
  auto spec0 = OpSpec(kTestOpName)
    .AddArg("name", "op0")
    .AddOutput("op0_0", "cpu")
    .AddArg("addend", 10);
  auto spec1 = OpSpec(kTestOpName)
    .AddArg("name", "op1")
    .AddArg("addend", 20)
    .AddOutput("op1_0", "cpu");
  auto spec2 = OpSpec(kTestOpName)
    .AddInput("op0_0", "cpu")
    .AddArg("name", "op2")
    .AddArgumentInput("addend", "op1_0")
    .AddOutput("op2_0", "cpu");
  auto spec3 = OpSpec(kTestOpName)
    .AddArg("name", "op3")
    .AddInput("op0_0", "cpu")
    .AddInput("op1_0", "cpu")
    .AddArg("addend", 1)
    .AddOutput("op3_0", "cpu");
  graph::OpGraph::Builder b;
  b.Add("op0", std::move(AddCommonArgs(spec0, 32)));
  b.Add("op1", std::move(AddCommonArgs(spec1, 32)));
  b.Add("op2", std::move(AddCommonArgs(spec2, 32)));
  b.Add("op3", std::move(AddCommonArgs(spec3, 32)));
  b.AddOutput("op3_0_cpu");
  b.AddOutput("op2_0_cpu");
  return std::move(b).GetGraph(true);
}

inline void CheckTestGraph1Results(const Workspace &ws, int batch_size) {
  auto &o0 = ws.Output<CPUBackend>(0);
  auto &o1 = ws.Output<CPUBackend>(1);
  ASSERT_EQ(o0.num_samples(), batch_size);
  ASSERT_EQ(o1.num_samples(), batch_size);
  for (int i = 0; i < batch_size; i++) {
    // The pipeline:
    // op0 = DummyOp(addend=10)
    // op1 = DummyOp(addend=20)
    // op2 = DummyOp(op0, addend=op1)
    // op3 = DummyOp(op0, op1, addend=1)
    // return op3, op2  # swapped!

    // DummyOp adds its argumetns, the "addend" and the sample index - thus, we have
    // tripled sample index + the sum of addends at output
    EXPECT_EQ(*o0[i].data<int>(), 10 + 20 + 3 * i + 1);
    EXPECT_EQ(*o1[i].data<int>(), 10 + 20 + 3 * i);
  }
}

inline auto GetTestGraph2() {
  auto spec0 = OpSpec(kTestOpName)
    .AddArg("name", "op0")
    .AddOutput("op0_0", "cpu")
    .AddArg("addend", 10);
  auto spec0c = OpSpec("MakeContiguous")
    .AddArg("name", "op0_cont")
    .AddInput("op0_0", "cpu")
    .AddOutput("op0_0", "gpu");
  auto spec1 = OpSpec(kTestOpName)
    .AddArg("name", "op1")
    .AddArg("addend", 20)
    .AddOutput("op1_0", "cpu");
  auto spec1c = OpSpec("MakeContiguous")
    .AddArg("name", "op1_cont")
    .AddInput("op1_0", "cpu")
    .AddOutput("op1_0", "gpu");
  auto spec2 = OpSpec(kTestOpName)
    .AddInput("op0_0", "cpu")
    .AddArg("name", "op2")
    .AddArgumentInput("addend", "op1_0")
    .AddOutput("op2_0", "cpu");
  auto spec3 = OpSpec(kTestOpName)
    .AddArg("name", "op3")
    .AddInput("op0_0", "gpu")
    .AddInput("op1_0", "gpu")
    .AddArg("addend", 1)
    .AddOutput("op3_0", "gpu");
  auto spec2c = OpSpec("MakeContiguous")
    .AddArg("name", "op2_cont")
    .AddInput("op2_0", "cpu")
    .AddOutput("op2_0", "gpu");
  graph::OpGraph::Builder b;
  b.Add("op0",  std::move(AddCommonArgs(spec0,  32, "cpu", 1)));
  b.Add("op0c", std::move(AddCommonArgs(spec0c, 32, "mixed", 1)));
  b.Add("op1",  std::move(AddCommonArgs(spec1,  32, "cpu", 1)));
  b.Add("op1c", std::move(AddCommonArgs(spec1c, 32, "mixed", 1)));
  b.Add("op2",  std::move(AddCommonArgs(spec2,  32, "cpu", 1)));
  b.Add("op2c", std::move(AddCommonArgs(spec2c, 32, "mixed", 1)));
  b.Add("op3",  std::move(AddCommonArgs(spec3,  32, "gpu", 1)));
  b.AddOutput("op3_0_gpu");
  b.AddOutput("op2_0_gpu");
  return std::move(b).GetGraph(true);
}

inline void CheckTestGraph2Results(const Workspace &ws, int batch_size) {
  auto &o0g = ws.Output<GPUBackend>(0);
  auto &o1g = ws.Output<GPUBackend>(1);
  TensorList<CPUBackend> o0, o1;
  o0.Copy(o0g);
  o1.Copy(o1g);
  ASSERT_EQ(o0.num_samples(), batch_size);
  ASSERT_EQ(o1.num_samples(), batch_size);
  for (int i = 0; i < batch_size; i++) {
    // The pipeline:
    // op0 = DummyOp(addend=10)
    // op1 = DummyOp(addend=20)
    // op2 = DummyOp(op0, addend=op1)
    // op3 = DummyOp(op0.gpu(), op1.gpu(), addend=1)
    // return op3, op2.gpu()  # swapped!

    // DummyOp adds its argumetns, the "addend" and the sample index - thus, we have
    // tripled sample index + the sum of addends at output
    EXPECT_EQ(*o0[i].data<int>(), 10 + 20 + 3 * i + 1);
    EXPECT_EQ(*o1[i].data<int>(), 10 + 20 + 3 * i);
  }
}

}  // namespace test
}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_TEST_H_
