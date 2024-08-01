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

#include <string>
#include "dali/pipeline/executor/executor2/exec2_ops_for_test.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/test/timing.h"

namespace dali {

namespace exec2 {
namespace test {

namespace {
// TODO(michalz): Avoid this code duplication without messing up encapsulation
void LimitBackendConcurrency(ExecGraph &graph, OpType backend, int max_concurrency = 1) {
  auto sem = std::make_shared<tasking::Semaphore>(max_concurrency);
  for (auto &n : graph.Nodes()) {
    if (n.backend == backend)
        n.concurrency = sem;
  }
  graph.Invalidate();
}
}  // namespace

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
  LimitBackendConcurrency(g, OpType::CPU);

  WorkspaceParams params = {};
  auto tp = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 0, false, "test");
  ExecEnv env;
  env.thread_pool = tp.get();
  params.env = &env;
  params.batch_size = batch_size;

  auto iter = std::make_shared<IterationData>();
  g.PrepareIteration(iter, params);
  tasking::Executor ex(1);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

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
  ExecGraph g;
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);
  ThreadPool tp(4, 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.batch_size = batch_size;

  {
    int N = 100;
    tasking::Executor ex(4);
    ex.Start();
    auto start = dali::test::perf_timer::now();
    for (int i = 0; i < N; i++) {
      g.PrepareIteration(std::make_shared<IterationData>(), params);
      auto fut = g.Launch(ex);
      auto &pipe_out = fut.Value<const PipelineOutput &>();
      auto &ws = pipe_out.workspace;
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

TEST(ExecGraphTest, SimpleGraphScheduleAhead) {
  int batch_size = 1;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "cpu")
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kCounterOpName);
  spec1.AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("name", "op1");
  auto op1 = std::make_unique<CounterOp>(spec1);

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
  LimitBackendConcurrency(g, OpType::CPU);

  ThreadPool tp(4, 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.batch_size = batch_size;

  int N = 100;
  tasking::Executor ex(4);
  ex.Start();
  std::vector<tasking::TaskFuture> fut;
  fut.reserve(N);
  for (int i = 0; i < N; i++) {
    g.PrepareIteration(std::make_shared<IterationData>(), params);
    fut.push_back(g.Launch(ex));
  }

  int ctr = 0;
  for (int i = 0; i < N; i++) {
    auto &pipe_out = fut[i].Value<const PipelineOutput &>();
    auto &out = pipe_out.workspace.Output<CPUBackend>(0);
    ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
    for (int s = 0; s < batch_size; s++)
      EXPECT_EQ(*out[s].data<int>(), 1010 + 2 * s + ctr++);
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
  ExecGraph g;
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);
  ThreadPool tp(std::thread::hardware_concurrency(), 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.batch_size = 32;
  {
    tasking::Executor ex(4);
    ex.Start();
    for (int i = 0; i < 10; i++) {
      g.PrepareIteration(std::make_shared<IterationData>(), params);
      auto fut = g.Launch(ex);
      EXPECT_THROW(fut.Value<const PipelineOutput &>(), std::runtime_error);
    }
  }
}

}  // namespace test
}  // namespace exec2
}  // namespace dali
