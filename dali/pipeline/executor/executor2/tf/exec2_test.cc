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
  DummyOp(const OpSpec &spec) : Operator<CPUBackend>(spec) {
    instance_name_ = spec_.GetArgument<string>("instance_name");
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    std::cerr << instance_name_ << " SetupImpl" << std::endl;
    int N = ws.GetRequestedBatchSize(0);
    std::cerr << "bs == " << N << std::endl;
    outs[0].shape = uniform_list_shape(N, TensorShape<>{});
    outs[0].type = DALI_INT32;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    std::cerr << instance_name_ << " RunImpl" << std::endl;
    int N = ws.GetRequestedBatchSize(0);
    std::cerr << "bs == " << N << std::endl;
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
  auto nvml_handle = nvml::NvmlInstance::CreateNvmlInstance();
  auto start = dali::test::perf_timer::now();
  int batch_size = 32;
  DummyOp::CreateSchema();
  OpSpec spec0("DummyOp");
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", "cpu")
       .AddArg("instance_name", "op0");
  DummyOp op0(spec0);

  OpSpec spec1("DummyOp");
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", "cpu")
       .AddArg("instance_name", "op1");
  DummyOp op1(spec1);

  OpSpec spec2("DummyOp");
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("instance_name", "op2");
  DummyOp op2(spec2);
  ExecGraph def;
  ExecNode *n2 = def.add_node(&op2);
  ExecNode *n1 = def.add_node(&op1);
  ExecNode *n0 = def.add_node(&op0);
  def.link(n0, 0, n2, 0);
  def.link(n1, 0, n2, 1);
  def.link(n2, 0, nullptr, 0);
  def.outputs.push_back(&def.edges.back());
  auto end = dali::test::perf_timer::now();
  std::cerr << "Test setup took " << dali::test::format_time(end-start) << std::endl;
  start = dali::test::perf_timer::now();
  auto tp = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 0, true, "test");
  end = dali::test::perf_timer::now();
  std::cerr << "Thread pool construction took " << dali::test::format_time(end-start) << std::endl;
  WorkspaceParams params = {};
  params.thread_pool = tp.get();
  params.batch_size = batch_size;
  start = dali::test::perf_timer::now();
  {
    auto sched = SchedGraph::from_def(def, params);
    tf::Taskflow tf;
    sched->schedule(tf);
    tf::Executor ex(4);
    ex.run(tf).get();
    auto &out = sched->outputs[0]->producer->ws->Output<CPUBackend>(0);
    ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
    for (int i = 0; i < batch_size; i++)
      EXPECT_EQ(*out[i].data<int>(), 1110 + 3 * i);
  }
  end = dali::test::perf_timer::now();
  std::cerr << "Graph lowering, execution and cleanup took " << dali::test::format_time(end-start) << std::endl;
  start = dali::test::perf_timer::now();
  tp.reset();
  end = dali::test::perf_timer::now();
  std::cerr << "Thread pool disposal took " << dali::test::format_time(end-start) << std::endl;
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
  DummyOp op0(spec0);

  OpSpec spec1("DummyOp");
  spec1.AddArg("addend", 200)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op1o0", "cpu")
       .AddArg("instance_name", "op1");
  DummyOp op1(spec1);

  OpSpec spec2("DummyOp");
  spec2.AddArg("addend", 1000.0f)  // this will cause a type error at run-time
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddInput("op1o0", "cpu")
       .AddInput("op2o0", "cpu")
       .AddOutput("op2e0", "cpu")
       .AddArg("instance_name", "op2");
  DummyOp op2(spec2);
  ExecGraph def;
  ExecNode *n2 = def.add_node(&op2);
  ExecNode *n1 = def.add_node(&op1);
  ExecNode *n0 = def.add_node(&op0);
  def.link(n0, 0, n2, 0);
  def.link(n1, 0, n2, 1);
  def.link(n2, 0, nullptr, 0);
  def.outputs.push_back(&def.edges.back());
  ThreadPool tp(std::thread::hardware_concurrency(), 0, true, "test");
  WorkspaceParams params = {};
  params.thread_pool = &tp;
  params.batch_size = 32;
  {
    auto sched_template = SchedGraph::from_def(def, params);
    tf::Executor ex(4);
    for (int i = 0; i < 10; i++) {
      tf::Taskflow tf;
      auto sched = sched_template->clone();
      sched->schedule(tf);
      EXPECT_THROW(ex.run(tf).get(), DALIException);
    }
  }
}



}  // namespace test
}  // namespace exec2
}  // namespace dali