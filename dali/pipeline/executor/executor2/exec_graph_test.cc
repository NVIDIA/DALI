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
#include "dali/pipeline/executor/executor2/exec2_test.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/test/timing.h"
#include "dali/core/cuda_stream_pool.h"

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

void SetOutputDevice(ExecNode *node, size_t out_idx, StorageDevice device) {
  if (node->outputs.size() <= out_idx)
    node->outputs.resize(out_idx + 1);
  node->outputs[out_idx].device = device;
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
       .AddInput("op0o0", "cpu")
       .AddInput("op1o0", "cpu")
       .AddOutput("op2o0", "cpu")
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
  params.max_batch_size = batch_size;

  auto iter = std::make_shared<IterationData>();
  params.iter_data = iter;
  g.PrepareIteration(params);
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
       .AddInput("op0o0", "cpu")
       .AddInput("op1o0", "cpu")
       .AddOutput("op2o0", "cpu")
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
  params.max_batch_size = batch_size;

  {
    int N = 100;
    tasking::Executor ex(4);
    ex.Start();
    auto start = dali::test::perf_timer::now();
    for (int i = 0; i < N; i++) {
      params.iter_data = std::make_shared<IterationData>();
      g.PrepareIteration(params);
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

TEST(ExecGraphTest, SimpleGraphScheduleAheadCPU) {
  int batch_size = 1;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 4)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "cpu")
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kCounterOpName);
  spec1.AddArg("num_threads", 4)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("name", "op1");
  auto op1 = std::make_unique<CounterOp>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 4)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", "cpu")
       .AddInput("op1o0", "cpu")
       .AddOutput("op2o0", "cpu")
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
  params.max_batch_size = batch_size;

  int N = 100;
  tasking::Executor ex(4);
  ex.Start();
  std::vector<tasking::TaskFuture> fut;
  fut.reserve(N);
  for (int i = 0; i < N; i++) {
    params.iter_data = std::make_shared<IterationData>();
    g.PrepareIteration(params);
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

TEST(ExecGraphTest, GraphScheduleAheadGPU) {
  int batch_size = 1;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 4)
       .AddArg("device", "gpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "gpu")
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpGPU>(spec0);

  OpSpec spec1(kCounterOpName);
  spec1.AddArg("num_threads", 4)
       .AddArg("delay", 0)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("name", "op1");
  auto op1 = std::make_unique<CounterOp>(spec1);

  OpSpec spec1c("MakeContiguous");
  spec1c.AddArg("num_threads", 4)
        .AddArg("device", "mixed")
        .AddArg("max_batch_size", batch_size)
        .AddInput("op1o0", "cpu")
        .AddOutput("op1o0", "gpu")
        .AddArg("name", "op1c");
  auto op1c = InstantiateOperator(spec1c);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 4)
       .AddArg("device", "gpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", "gpu")
       .AddInput("op1o0", "gpu")
       .AddOutput("op2o0", "gpu")
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpGPU>(spec2);
  ExecGraph g;
  ExecNode *n2  = g.AddNode(std::move(op2));
  ExecNode *n1  = g.AddNode(std::move(op1));
  ExecNode *n1c = g.AddNode(std::move(op1c));
  ExecNode *n0  = g.AddNode(std::move(op0));
  ExecNode *no  = g.AddOutputNode();
  n0->output_queue_limit = std::make_shared<tasking::Semaphore>(3);
  SetOutputDevice(n0, 0, StorageDevice::GPU);
  SetOutputDevice(n1c, 0, StorageDevice::GPU);
  SetOutputDevice(n2, 0, StorageDevice::GPU);
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n1c, 0);
  g.Link(n1c, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);

  auto s0 = CUDAStreamPool::instance().Get();
  auto s1 = CUDAStreamPool::instance().Get();
  auto s2 = CUDAStreamPool::instance().Get();
  auto s3 = CUDAStreamPool::instance().Get();

  // Each GPU-capable operator uses a different stream
  n0->env.order = s0;
  n1c->env.order = s1;
  n2->env.order = s2;
  no->env.order = s3;

  ThreadPool tp(4, 0, false, "test");

  n1->env.thread_pool = &tp;

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;

  int N = 100;
  tasking::Executor ex(4);
  ex.Start();
  std::vector<tasking::TaskFuture> fut;
  fut.reserve(N);
  for (int i = 0; i < N; i++) {
    params.iter_data = std::make_shared<IterationData>();
    g.PrepareIteration(params);
    fut.push_back(g.Launch(ex));
  }

  int ctr = 0;
  for (int i = 0; i < N; i++) {
    auto &pipe_out = fut[i].Value<const PipelineOutput &>();
    ASSERT_TRUE(pipe_out.workspace.has_event());
    EXPECT_EQ(pipe_out.workspace.stream(), s3.get());
    EXPECT_EQ(pipe_out.workspace.event(), pipe_out.event.get());
    AccessOrder::host().wait(pipe_out.workspace.event());
    auto &out_gpu = pipe_out.workspace.Output<GPUBackend>(0);
    ASSERT_EQ(out_gpu.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
    TensorList<CPUBackend> out;
    out.Copy(out_gpu);
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
       .AddInput("op0o0", "cpu")
       .AddInput("op1o0", "cpu")
       .AddOutput("op2o0", "cpu")
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
  params.max_batch_size = 32;
  {
    tasking::Executor ex(4);
    ex.Start();
    for (int i = 0; i < 10; i++) {
      params.iter_data = std::make_shared<IterationData>();
      g.PrepareIteration(params);
      auto fut = g.Launch(ex);
      EXPECT_THROW(fut.Value<const PipelineOutput &>(), std::runtime_error);
    }
  }
}

TEST(ExecGraphTest, LoweredStructureMatch) {
  graph::OpGraph def = GetTestGraph1();
  ExecGraph g;
  g.Lower(def);
  ASSERT_EQ(g.Nodes().size(), def.OpNodes().size() + 1);
  EXPECT_TRUE(g.Nodes().back().is_pipeline_output);
  EXPECT_EQ(g.Nodes().back().inputs.size(), 2_uz);
  auto def_it = def.OpNodes().begin();
  auto ex_it = g.Nodes().begin();
  for (; def_it != def.OpNodes().end(); def_it++, ex_it++) {
    EXPECT_EQ(ex_it->inputs.size(), def_it->inputs.size());
    EXPECT_EQ(ex_it->outputs.size(), def_it->outputs.size());
  }
  if (HasFailure())
    FAIL() << "Structure mismatch detected - test cannot proceed further.";
  def_it = def.OpNodes().begin();
  ex_it = g.Nodes().begin();

  auto &def0 = *def_it++;
  auto &def1 = *def_it++;
  auto &def2 = *def_it++;
  auto &def3 = *def_it++;

  auto &ex0 = *ex_it++;
  auto &ex1 = *ex_it++;
  auto &ex2 = *ex_it++;
  auto &ex3 = *ex_it++;
  auto &ex_out = g.Nodes().back();

  ASSERT_EQ(ex0.outputs.size(), 1_uz);
  ASSERT_EQ(ex0.outputs[0].consumers.size(), 2_uz);
  EXPECT_EQ(ex0.outputs[0].consumers[0]->consumer, &ex2);
  EXPECT_EQ(ex0.outputs[0].consumers[1]->consumer, &ex3);

  ASSERT_EQ(ex1.outputs.size(), 1_uz);
  EXPECT_EQ(ex1.outputs[0].consumers[0]->consumer, &ex2);
  ASSERT_EQ(ex1.outputs[0].consumers.size(), 2_uz);
  EXPECT_EQ(ex1.outputs[0].consumers[1]->consumer, &ex3);

  ASSERT_EQ(ex2.outputs.size(), 1_uz);
  ASSERT_EQ(ex2.outputs[0].consumers.size(), 1_uz);
  EXPECT_EQ(ex2.outputs[0].consumers[0]->consumer, &ex_out);
  ASSERT_EQ(ex2.inputs.size(), 2_uz);
  EXPECT_EQ(ex2.inputs[0]->producer, &ex0);
  EXPECT_EQ(ex2.inputs[1]->producer, &ex1);

  ASSERT_EQ(ex3.outputs.size(), 1_uz);
  ASSERT_EQ(ex3.outputs[0].consumers.size(), 1_uz);
  EXPECT_EQ(ex3.outputs[0].consumers[0]->consumer, &ex_out);
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
  LimitBackendConcurrency(g, OpType::CPU);

  ThreadPool tp(std::thread::hardware_concurrency(), 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.batch_size = 32;
  auto iter = std::make_shared<IterationData>();
  {
    tasking::Executor ex(4);
    ex.Start();
    g.PrepareIteration(iter, params);
    auto fut = g.Launch(ex);
    auto &out = fut.Value<const PipelineOutput &>();
    CheckTestGraph1Results(out.workspace, params.batch_size);
  }
}

}  // namespace test
}  // namespace exec2
}  // namespace dali
