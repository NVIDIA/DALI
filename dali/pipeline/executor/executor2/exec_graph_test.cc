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
#include <string>
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/test/timing.h"

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

}  // namespace test
}  // namespace exec2

DALI_SCHEMA(Exec2TestOp)  // DALI_SCHEMA can't take a macro :(
  .NumInput(0, 99)
  .NumOutput(1)
  .AddArg("addend", "a value added to the sum of inputs", DALI_INT32, true);

// DALI_REGISTER_OPERATOR can't take a macro for the name
DALI_REGISTER_OPERATOR(Exec2TestOp, exec2::test::DummyOpCPU, CPU);

namespace exec2 {
namespace test {

TEST(ExecGraphTest, SimpleGraph) {
  int batch_size = 32;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "cpu")
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("name", "op1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);
  ExecGraph g;
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n2, 1);
  g.Link(n2, 0, no, 0);

  WorkspaceParams params = {};
  auto tp = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 0, false, "test");
  params.thread_pool = tp.get();
  params.batch_size = batch_size;

  auto iter = std::make_shared<IterationData>();
  g.PrepareIteration(iter, params);
  tasking::Executor ex(1);
  ex.Start();
  auto fut = g.Launch(ex);
  Workspace ws = fut.Value<Workspace>();

  auto &out = ws.Output<CPUBackend>(0);
  ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
  for (int i = 0; i < batch_size; i++)
    EXPECT_EQ(*out[i].data<int>(), 1110 + 3 * i);
}

TEST(ExecGraphTest, SimpleGraphRepeat) {
  int batch_size = 256;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "cpu")
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("name", "op1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);
  ExecGraph def;
  ExecNode *n2 = def.AddNode(std::move(op2));
  ExecNode *n1 = def.AddNode(std::move(op1));
  ExecNode *n0 = def.AddNode(std::move(op0));
  ExecNode *no = def.AddOutputNode();
  def.Link(n0, 0, n2, 0);
  def.Link(n1, 0, n2, 1);
  def.Link(n2, 0, no, 0);
  ThreadPool tp(4, 0, false, "test");
  WorkspaceParams params = {};
  params.thread_pool = &tp;
  params.batch_size = batch_size;

  {
    int N = 100;
    tasking::Executor ex(4);
    ex.Start();
    auto start = dali::test::perf_timer::now();
    for (int i = 0; i < N; i++) {
      def.PrepareIteration(std::make_shared<IterationData>(), params);
      auto ws = def.Launch(ex).Value<Workspace>();
      auto &out = ws.Output<CPUBackend>(0);
      ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
      for (int i = 0; i < batch_size; i++)
        EXPECT_EQ(*out[i].data<int>(), 1110 + 3 * i);
    }
    auto end = dali::test::perf_timer::now();
    print(std::cerr, "Average iteration time over ", N, " iterations is ",
          dali::test::format_time((end - start) / N), "\n");
  }
}


TEST(ExecGraphTest, Exception) {
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op0o0", "cpu")
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 200)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op1o0", "cpu")
       .AddArg("name", "op1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000.0f)  // this will cause a type error at run-time
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);
  ExecGraph def;
  ExecNode *n2 = def.AddNode(std::move(op2));
  ExecNode *n1 = def.AddNode(std::move(op1));
  ExecNode *n0 = def.AddNode(std::move(op0));
  ExecNode *no = def.AddOutputNode();
  def.Link(n0, 0, n2, 0);
  def.Link(n1, 0, n2, 1);
  def.Link(n2, 0, no, 0);
  ThreadPool tp(std::thread::hardware_concurrency(), 0, false, "test");
  WorkspaceParams params = {};
  params.thread_pool = &tp;
  params.batch_size = 32;
  {
    tasking::Executor ex(4);
    ex.Start();
    for (int i = 0; i < 10; i++) {
      def.PrepareIteration(std::make_shared<IterationData>(), params);
      auto fut = def.Launch(ex);
      EXPECT_THROW(fut.Value<Workspace>(), DALIException);
    }
  }
}

namespace {

auto GetTestGraph1() {
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

}  // namespace

TEST(ExecGraphTest, LoweredStructureMatch) {
  graph::OpGraph def = GetTestGraph1();
  ExecGraph g;
  g.Lower(def);
  ASSERT_EQ(g.nodes.size(), def.OpNodes().size() + 1);
  EXPECT_TRUE(g.nodes.back().is_pipeline_output);
  EXPECT_EQ(g.nodes.back().inputs.size(), 2_uz);
  auto def_it = def.OpNodes().begin();
  auto ex_it = g.nodes.begin();
  for (; def_it != def.OpNodes().end(); def_it++, ex_it++) {
    EXPECT_EQ(ex_it->def, &*def_it);
    EXPECT_EQ(ex_it->inputs.size(), def_it->inputs.size());
    EXPECT_EQ(ex_it->outputs.size(), CountOutgoingEdges(*def_it));
  }
  if (HasFailure())
    FAIL() << "Structure mismatch detected - test cannot proceed further.";
  def_it = def.OpNodes().begin();
  ex_it = g.nodes.begin();

  auto &def0 = *def_it++;
  auto &def1 = *def_it++;
  auto &def2 = *def_it++;
  auto &def3 = *def_it++;

  auto &ex0 = *ex_it++;
  auto &ex1 = *ex_it++;
  auto &ex2 = *ex_it++;
  auto &ex3 = *ex_it++;
  auto &ex_out = g.nodes.back();

  ASSERT_EQ(ex0.outputs.size(), 2_uz);
  EXPECT_EQ(ex0.outputs[0]->consumer, &ex2);
  EXPECT_EQ(ex0.outputs[1]->consumer, &ex3);

  ASSERT_EQ(ex1.outputs.size(), 2_uz);
  EXPECT_EQ(ex1.outputs[0]->consumer, &ex2);
  EXPECT_EQ(ex1.outputs[1]->consumer, &ex3);

  ASSERT_EQ(ex2.outputs.size(), 1_uz);
  EXPECT_EQ(ex2.outputs[0]->consumer, &ex_out);
  ASSERT_EQ(ex2.inputs.size(), 2_uz);
  EXPECT_EQ(ex2.inputs[0]->producer, &ex0);
  EXPECT_EQ(ex2.inputs[1]->producer, &ex1);

  ASSERT_EQ(ex3.outputs.size(), 1_uz);
  EXPECT_EQ(ex3.outputs[0]->consumer, &ex_out);
  EXPECT_EQ(ex3.inputs[0]->producer, &ex0);
  EXPECT_EQ(ex3.inputs[1]->producer, &ex1);

  ASSERT_EQ(ex_out.inputs.size(), 2_uz);
  EXPECT_EQ(ex_out.inputs[0]->producer, &ex3);
  EXPECT_EQ(ex_out.inputs[1]->producer, &ex2);
}

TEST(ExecGraphTest, LoweredExec) {
  graph::OpGraph def = GetTestGraph1();
  ExecGraph g;
  g.Lower(def);

  ThreadPool tp(std::thread::hardware_concurrency(), 0, false, "test");
  WorkspaceParams params = {};
  params.thread_pool = &tp;
  params.batch_size = 32;
  auto iter = std::make_shared<IterationData>();
  {
    tasking::Executor ex(4);
    ex.Start();
    g.PrepareIteration(iter, params);
    auto fut = g.Launch(ex);
    Workspace ws = fut.Value<Workspace>();
    auto &o0 = ws.Output<CPUBackend>(0);
    auto &o1 = ws.Output<CPUBackend>(1);
    for (int i = 0; i < params. batch_size; i++) {
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
}

}  // namespace test
}  // namespace exec2
}  // namespace dali
