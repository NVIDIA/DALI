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
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/test/timing.h"

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
  explicit DummyOp(const OpSpec &spec) : Operator<CPUBackend>(spec) {
    instance_name_ = spec_.GetArgument<string>("instance_name");
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

  static OpSchema &CreateSchema() {
    return CreateTestSchema("DummyOp")
      .NumInput(0, 99)
      .NumOutput(1)
      .AddArg("addend", "a value added to the sum of inputs", DALI_INT32);
  }

  std::string instance_name_;
};

TEST(Exec2Test, SimpleGraph) {
  int batch_size = 32;
  DummyOp::CreateSchema();
  OpSpec spec0("DummyOp");
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "cpu")
       .AddArg("instance_name", "op0");
  auto op0 = std::make_unique<DummyOp>(spec0);

  OpSpec spec1("DummyOp");
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("instance_name", "op1");
  auto op1 = std::make_unique<DummyOp>(spec1);

  OpSpec spec2("DummyOp");
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("instance_name", "op2");
  auto op2 = std::make_unique<DummyOp>(spec2);
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

TEST(Exec2Test, SimpleGraphRepeat) {
  int batch_size = 256;
  DummyOp::CreateSchema();
  OpSpec spec0("DummyOp");
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "cpu")
       .AddArg("instance_name", "op0");
  auto op0 = std::make_unique<DummyOp>(spec0);

  OpSpec spec1("DummyOp");
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("instance_name", "op1");
  auto op1 = std::make_unique<DummyOp>(spec1);

  OpSpec spec2("DummyOp");
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("instance_name", "op2");
  auto op2 = std::make_unique<DummyOp>(spec2);
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
      /*ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
      for (int i = 0; i < batch_size; i++)
        EXPECT_EQ(*out[i].data<int>(), 1110 + 3 * i);*/
    }
    auto end = dali::test::perf_timer::now();
    print(std::cerr, "Average iteration time over ", N, " iterations is ",
          dali::test::format_time((end - start) / N), "\n");
  }
}


TEST(Exec2Test, Exception) {
  DummyOp::CreateSchema();
  OpSpec spec0("DummyOp");
  spec0.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op0o0", "cpu")
       .AddArg("instance_name", "op0");
  auto op0 = std::make_unique<DummyOp>(spec0);

  OpSpec spec1("DummyOp");
  spec1.AddArg("addend", 200)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op1o0", "cpu")
       .AddArg("instance_name", "op1");
  auto op1 = std::make_unique<DummyOp>(spec1);

  OpSpec spec2("DummyOp");
  spec2.AddArg("addend", 1000.0f)  // this will cause a type error at run-time
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("instance_name", "op2");
  auto op2 = std::make_unique<DummyOp>(spec2);
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


}  // namespace test
}  // namespace exec2
}  // namespace dali
