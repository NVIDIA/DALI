// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/test/dali_test_decoder.h"
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/async_separated_pipelined_executor.h"
#include "dali/test/dali_test_utils.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {

template <typename ExecutorToTest>
class ExecutorTest : public GenericDecoderTest<RGB> {
 protected:
  template <typename... T>
  std::unique_ptr<ExecutorToTest> GetExecutor(T&&... args) {
    return std::unique_ptr<ExecutorToTest>(new ExecutorToTest(std::forward<T>(args)...));
  }

  uint32_t GetImageLoadingFlags() const override {
    return t_loadJPEGs + t_decodeJPEGs;
  }

  void SetUp() override {
    DALISingleOpTest::SetUp();
    set_batch_size(jpegs_.nImages());
  }

  inline void set_batch_size(int size) { batch_size_ = size; }

  inline OpSpec& PrepareSpec(OpSpec &spec) const {
    spec.AddArg("max_batch_size", batch_size_)
      .AddArg("num_threads", num_threads_);
    return spec;
  }

  inline void PruneGraph(ExecutorBase *exe) const {
    exe->PruneUnusedGraphNodes();
  }

  bool HasConditionals(ExecutorBase &exe) const {
    return exe.HasConditionals();
  }

  // TODO(klecki): adjust to refactored code
  vector<Workspace> CPUData(ExecutorBase *exe, int idx) const {
    // return std::get<static_cast<int>(OpType::CPU)>(exe->wss_[idx].op_data);
    return {};
  }

  vector<Workspace> MixedData(ExecutorBase *exe, int idx) const {
    // return std::get<static_cast<int>(OpType::MIXED)>(exe->wss_[idx].op_data);
    return {};
  }

  vector<Workspace> GPUData(ExecutorBase *exe, int idx) const {
    // return std::get<static_cast<int>(OpType::GPU)>(exe->wss_[idx].op_data);
    return {};
  }

  bool IsSeparated() {
    return std::is_same_v<ExecutorToTest, SeparatedPipelinedExecutor>
        || std::is_same_v<ExecutorToTest, AsyncSeparatedPipelinedExecutor>;
  }

  template<typename Factory>
  void RunCheckpointingTest(Factory executor_and_graph_factory,
                            int epoch_size, int epochs_cnt = 3) {
    auto collect_result = [&](const TensorList<CPUBackend> &data) {
      std::vector<uint8_t> result;
      for (int i = 0; i < data.num_samples(); i++)
        result.push_back(data.tensor<uint8_t>(i)[0]);
      return result;
    };

    Workspace ws;
    auto run_epoch = [&](std::unique_ptr<ExecutorToTest> &exec) {
      std::vector<std::vector<uint8_t>> results;
      for (int i = 0; i < epoch_size; i++) {
        exec->RunCPU();
        exec->RunMixed();
        exec->RunGPU();
        exec->Outputs(&ws);

        if (ws.OutputIsType<CPUBackend>(0)) {
          results.push_back(collect_result(ws.Output<CPUBackend>(0)));
        } else {
          TensorList<CPUBackend> cpu;
          cpu.Copy(ws.Output<GPUBackend>(0));
          results.push_back(collect_result(cpu));
        }
      }

      return results;
    };

    auto [exec1, graph1] = executor_and_graph_factory();
    auto [exec2, graph2] = executor_and_graph_factory();

    for (int i = 0; i < epochs_cnt; i++)
      run_epoch(exec1);

    auto cpt = exec1->GetCurrentCheckpoint();
    exec2->RestoreStateFromCheckpoint(cpt);

    for (int i = 0; i < epochs_cnt; i++)
      EXPECT_EQ(run_epoch(exec1), run_epoch(exec2));
  }

  int batch_size_, num_threads_ = 1;
};

using ExecutorTypes =
    ::testing::Types<SimpleExecutor, PipelinedExecutor, SeparatedPipelinedExecutor,
                     AsyncPipelinedExecutor, AsyncSeparatedPipelinedExecutor>;

TYPED_TEST_SUITE(ExecutorTest, ExecutorTypes);

template <typename ExecutorToTest>
using ExecutorSyncTest = ExecutorTest<ExecutorToTest>;

using ExecutorSyncTypes =
    ::testing::Types<SimpleExecutor, PipelinedExecutor, SeparatedPipelinedExecutor>;

TYPED_TEST_SUITE(ExecutorSyncTest, ExecutorSyncTypes);

TYPED_TEST(ExecutorTest, TestPruneBasicGraph) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data1", "cpu")
          .AddOutput("data2", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data1", "cpu")
          .AddOutput("data3", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("data3", "cpu")
          .AddOutput("data3_cont", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data1", "cpu")
          .AddOutput("data4", "cpu")), "");

  vector<string> outputs = {"data3_cont_cpu"};
  exe->Build(&graph, outputs);

  // Validate the graph - op 3 should
  // have been pruned as its outputs
  // are unused.
  ASSERT_EQ(graph.NumOp(OpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 1);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 0);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));

  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 1);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(node2.spec.Output(0), "data3_cpu");

  auto& node3 = graph.Node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 0);
  ASSERT_EQ(node3.parents.size(), 1);
  ASSERT_EQ(node3.parents.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node3.spec.Output(0)), 2);
  ASSERT_EQ(graph.TensorIdxInSource(node3.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node3.spec.Output(0)));
  ASSERT_EQ(node3.spec.Output(0), "data3_cont_cpu");
}

TYPED_TEST(ExecutorTest, TestPruneMultiple) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data1", "cpu")
          .AddOutput("data2", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("data1", "cpu")
          .AddOutput("data1_cont", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data1", "cpu")
          .AddOutput("data3", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data1", "cpu")
          .AddOutput("data4", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 0)
          .AddArg("preserve", true)
          .AddInput("data1", "cpu")), "");

  vector<string> outputs = {"data1_cont_cpu"};
  exe->Build(&graph, outputs);

  // Validate the graph - op 2&3 should
  // have been pruned.
  // Op 4 should not be pruned
  ASSERT_EQ(graph.NumOp(OpType::CPU), 2);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 1);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 0);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 2);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(node.spec.NumOutput(), 2);
  ASSERT_EQ(node.spec.Output(0), "data1_cpu");
  ASSERT_EQ(node.spec.Output(1), "data2_cpu");

  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(node2.spec.NumOutput(), 1);
  ASSERT_EQ(node2.spec.Output(0), "data1_cont_cpu");
}

TYPED_TEST(ExecutorTest, TestPruneRecursive) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddOutput("data1", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("data1", "cpu")
          .AddOutput("data1_cont", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data1", "cpu")
          .AddOutput("data2", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data2", "cpu")
          .AddOutput("data3", "cpu")), "");

  vector<string> outputs = {"data1_cont_cpu"};
  exe->Build(&graph, outputs);

  // Validate the graph - op 2&3 should
  // have been pruned
  ASSERT_EQ(graph.NumOp(OpType::CPU), 1);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 1);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 0);

  // Validate the source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(node.spec.NumOutput(), 1);
  ASSERT_EQ(node.spec.Output(0), "data1_cpu");

  auto& node2 = graph.Node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 0);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(graph.TensorSourceID(node2.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node2.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node2.spec.Output(0)));
  ASSERT_EQ(node2.spec.NumOutput(), 1);
  ASSERT_EQ(node2.spec.Output(0), "data1_cont_cpu");
}

TYPED_TEST(ExecutorTest, TestPruneWholeGraph) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddOutput("data1", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data1", "cpu")
          .AddOutput("data2", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddArg("num_outputs", 1)
          .AddInput("data2", "cpu")
          .AddOutput("data3", "cpu")), "");

  vector<string> outputs = {"data_that_does_not_exist"};
  ASSERT_THROW(this->PruneGraph(exe.get()),
      std::runtime_error);
}

// TODO(klecki): adjust to after refactor
TYPED_TEST(ExecutorTest, DISABLED_TestDataSetup) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("device_id", 0)
          .AddOutput("data1", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("data1", "cpu")
          .AddOutput("data2", "gpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "gpu")
          .AddArg("num_outputs", 1)
          .AddInput("data2", "gpu")
          .AddOutput("data3", "gpu")), "");

  vector<string> outputs = {"data3_gpu"};
  exe->Build(&graph, outputs);

  // Verify the data has been setup correctly
  for (int i = 0; i < 2; ++i) {
    auto host_workspaces = this->CPUData(exe.get(), i);
    ASSERT_EQ(host_workspaces.size(), 1);
    Workspace &hws = host_workspaces[0];
    ASSERT_EQ(hws.NumInput(), 0);
    ASSERT_EQ(hws.NumOutput(), 1);
    ASSERT_EQ(hws.GetRequestedBatchSize(0), this->batch_size_);
    ASSERT_TRUE(hws.OutputIsType<CPUBackend>(0));

    auto mixed_workspaces = this->MixedData(exe.get(), i);
    ASSERT_EQ(mixed_workspaces.size(), 1);
    Workspace &mws = mixed_workspaces[0];
    ASSERT_EQ(mws.NumInput(), 1);
    ASSERT_EQ(mws.GetInputBatchSize(0), this->batch_size_);
    ASSERT_TRUE(mws.InputIsType<CPUBackend>(0));
    ASSERT_EQ(mws.NumOutput(), 1);
    ASSERT_TRUE(mws.OutputIsType<GPUBackend>(0));

    auto device_workspaces = this->GPUData(exe.get(), i);
    ASSERT_EQ(device_workspaces.size(), 1);
    Workspace &dws = device_workspaces[0];
    ASSERT_EQ(dws.NumInput(), 1);
    ASSERT_TRUE(dws.InputIsType<GPUBackend>(0));
    ASSERT_EQ(dws.NumOutput(), 1);
    ASSERT_TRUE(dws.OutputIsType<GPUBackend>(0));
  }
}

TYPED_TEST(ExecutorTest, TestRunBasicGraph) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("device_id", 0)
          .AddOutput("data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("images", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("images", "cpu")
          .AddOutput("final_images", "cpu")), "");

  vector<string> outputs = {"final_images_cpu"};
  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  test::MakeRandomBatch(tl, this->batch_size_);
  src_op->SetDataSource(tl);

  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();

  Workspace ws;
  exe->Outputs(&ws);
  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_TRUE(ws.OutputIsType<CPUBackend>(0));
}

TYPED_TEST(ExecutorTest, TestRunBasicGraphWithCB) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("device_id", 0)
          .AddOutput("data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("images", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("images", "cpu")
          .AddOutput("final_images", "cpu")), "");

  vector<string> outputs = {"final_images_cpu"};

  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  test::MakeRandomBatch(tl, this->batch_size_);
  src_op->SetDataSource(tl);

  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();

  Workspace ws;
  exe->Outputs(&ws);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_TRUE(ws.OutputIsType<CPUBackend>(0));
}

// This test does not work with Async Executors
TYPED_TEST(ExecutorSyncTest, TestPrefetchedExecution) {
  int batch_size = this->batch_size_ / 2;
  this->set_batch_size(batch_size);
  this->SetEps(1.6);

  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("device_id", 0)
          .AddOutput("data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("images", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("images", "cpu")
          .AddOutput("images", "gpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("images", "gpu")
          .AddOutput("final_images", "gpu")), "");

  vector<string> outputs = {"final_images_gpu"};
  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);

  TensorList<CPUBackend> tl;
  test::MakeRandomBatch(tl, this->batch_size_ * 2);

  // Split the batch into two
  TensorList<CPUBackend> tl2;
  TensorList<CPUBackend> tl1;
  TensorListShape<> shape1(batch_size, tl.shape().sample_dim()),
      shape2(batch_size, tl.shape().sample_dim());
  for (int i = 0; i < batch_size; ++i) {
    shape1.set_tensor_shape(i, tl.tensor_shape(i));
    shape2.set_tensor_shape(i, tl.tensor_shape(i+batch_size));
  }
  tl1.Resize(shape1, DALI_UINT8);
  tl2.Resize(shape2, DALI_UINT8);
  for (int i = 0; i < batch_size; ++i) {
    std::memcpy(
        tl1.template mutable_tensor<uint8>(i),
        tl.template tensor<uint8>(i),
        volume(tl.tensor_shape(i)));
    std::memcpy(
        tl2.template mutable_tensor<uint8>(i),
        tl.template tensor<uint8>(i+batch_size),
        volume(tl.tensor_shape(i+batch_size)));
  }


  Workspace ws;


  auto run = [&src_op, &exe] (TensorList<CPUBackend> &input) {
    src_op->SetDataSource(input);
    exe->RunCPU();
    exe->RunMixed();
    exe->RunGPU();
  };

  auto check = [&exe, &ws, &tl, batch_size] (int batch_idx) {
    exe->Outputs(&ws);
    ASSERT_EQ(ws.NumOutput(), 1);
    ASSERT_EQ(ws.NumInput(), 0);
    ASSERT_TRUE(ws.OutputIsType<GPUBackend>(0));
    test::CheckResults(ws, batch_size, batch_idx, tl);
  };

  // Run twice without getting the results if we are not SimpleExecutor which will overwrite data
  // due to prefetch queue = 1.
  if (std::is_same_v<SimpleExecutor, TypeParam>) {
    run(tl1);
    check(0);
    run(tl2);
    check(1);
  } else {
    run(tl1);
    run(tl2);
    check(0);
    check(1);
  }
}


TYPED_TEST(ExecutorTest, TestPinning) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(OpSpec("ExternalSource")
                                    .AddArg("device", "cpu")
                                    .AddArg("device_id", 0)
                                    .AddOutput("data_0", "cpu")),
              "ExternalSource_0");

  // First set of Copy + Copy and Pass Through
  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddArg("device", "cpu")
                                    .AddInput("data_0", "cpu")
                                    .AddOutput("copy_0", "cpu")),
              "Copy_0");

  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddArg("device", "cpu")
                                    .AddInput("data_0", "cpu")
                                    .AddOutput("copy_1", "cpu")),
              "Copy_1");

  graph.AddOp(this->PrepareSpec(OpSpec("PassthroughOp")
                                    .AddArg("device", "cpu")
                                    .AddInput("copy_0", "cpu")
                                    .AddOutput("pass_through_0", "cpu")),
              "PassThrough_0");

  // Trigger pinning of first set when it moves CPU -> GPU
  graph.AddOp(this->PrepareSpec(OpSpec("MakeContiguous")
                                    .AddArg("device", "mixed")
                                    .AddInput("pass_through_0", "cpu")
                                    .AddOutput("out_0", "gpu")),
              "MakeContiguous_0");

  // but not the Copy_1 to compare against
  graph.AddOp(this->PrepareSpec(OpSpec("MakeContiguous")
                                    .AddArg("device", "mixed")
                                    .AddInput("copy_1", "cpu")
                                    .AddOutput("out_1", "cpu")),
              "MakeContiguous_1");


  // Second set of Copy and Pass Through
  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddArg("device", "cpu")
                                    .AddInput("data_0", "cpu")
                                    .AddOutput("copy_2", "cpu")),
              "Copy_2");

  graph.AddOp(this->PrepareSpec(OpSpec("PassthroughOp")
                                    .AddArg("device", "cpu")
                                    .AddInput("copy_2", "cpu")
                                    .AddOutput("pass_through_1", "cpu")),
              "PassThrough_1");

  // Check pinning argument inputs to operators in GPU stage
  graph.AddOp(this->PrepareSpec(OpSpec("CopyArgumentOp")
                                    .AddArg("device", "gpu")
                                    .AddArgumentInput("to_copy", "pass_through_1")
                                    .AddOutput("out_2", "gpu")),
              "DummyOpGpu");

  vector<string> outputs = {"copy_0_cpu",         "copy_1_cpu", "pass_through_0_cpu", "copy_2_cpu",
                            "pass_through_1_cpu", "out_0_gpu",  "out_1_cpu",          "out_2_gpu"};

  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op = dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  tl.Resize(uniform_list_shape(this->batch_size_, TensorShape<>{}), DALI_FLOAT);
  src_op->SetDataSource(tl);

  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();

  Workspace ws;
  exe->Outputs(&ws);

  // Utilize the fact that the outputs are shared from the executor, so we can check if they are
  // pinned in a way we expect
  // Currently we expect to pin anything that is CPU argument input into GPU operator, and
  // is a CPU -> GPU copy (not via a decoder), so CPU input to Mixed operator that returns GPU data.
  // The whole pass-through group should be pinned as well.

  EXPECT_TRUE(ws.Output<CPUBackend>(0).is_pinned());   // copy_0_cpu
  EXPECT_FALSE(ws.Output<CPUBackend>(1).is_pinned());  // copy_1_cpu
  EXPECT_TRUE(ws.Output<CPUBackend>(2).is_pinned());   // pass_through_0_cpu
  EXPECT_TRUE(ws.Output<CPUBackend>(3).is_pinned());   // copy_2_cpu
  EXPECT_TRUE(ws.Output<CPUBackend>(4).is_pinned());   // pass_through_1_cpu
}


TYPED_TEST(ExecutorTest, TestCondtionalDetection) {
  auto exe_no_cond = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  auto exe_with_cond = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe_no_cond->Init();
  exe_with_cond->Init();

  // Build a basic graph without conditionals.
  OpGraph graph_no_cond;
  graph_no_cond.AddOp(this->PrepareSpec(OpSpec("ExternalSource")
                                            .AddArg("device", "cpu")
                                            .AddArg("device_id", 0)
                                            .AddOutput("data", "cpu")),
                      "ExternalSource");

  // Build a basic graph without conditionals.
  OpGraph graph_with_cond;
  graph_with_cond.AddOp(this->PrepareSpec(OpSpec("ExternalSource")
                                            .AddArg("device", "cpu")
                                            .AddArg("device_id", 0)
                                            .AddOutput("input", "cpu")),
                      "ExternalSource");

  graph_with_cond.AddOp(this->PrepareSpec(OpSpec("_conditional__Split")
                                              .AddArg("device", "cpu")
                                              .AddInput("input", "cpu")
                                              .AddArgumentInput("predicate", "input")
                                              .AddOutput("true_output", "cpu")
                                              .AddOutput("false_output", "cpu")
                                              .AddArg("_if_stmt", true)),
                        "split");

  graph_with_cond.AddOp(this->PrepareSpec(OpSpec("_conditional__Merge")
                                              .AddArg("device", "cpu")
                                              .AddInput("true_output", "cpu")
                                              .AddInput("false_output", "cpu")
                                              .AddArgumentInput("predicate", "input")
                                              .AddOutput("output", "cpu")),
                        "merge");

  exe_no_cond->Build(&graph_no_cond, {"data_cpu"});
  exe_with_cond->Build(&graph_with_cond, {"output_cpu"});

  EXPECT_FALSE(this->HasConditionals(*exe_no_cond));
  EXPECT_TRUE(this->HasConditionals(*exe_with_cond));
}


TYPED_TEST(ExecutorTest, SimpleCheckpointingCPU) {
  constexpr int epoch_size = 4;
  auto prepare_executor_and_graph = [&] {
    auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe->EnableCheckpointing(true);
    exe->Init();

    auto graph = std::make_unique<OpGraph>();
    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulSource")
          .AddArg("checkpointing", true)
          .AddArg("epoch_size", epoch_size)
          .AddOutput("state", "cpu")),
      "dummy");

    exe->Build(graph.get(), {"state_cpu"});
    return std::pair{std::move(exe), std::move(graph)};
  };

  if (this->IsSeparated())
    EXPECT_THROW(
      this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size),
      DALIException);
  else
    this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size);
}

TYPED_TEST(ExecutorTest, PipelineCheckpointingCPU) {
  constexpr int epoch_size = 4;
  auto prepare_executor_and_graph = [&] {
    auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe->EnableCheckpointing(true);
    exe->Init();

    auto graph = std::make_unique<OpGraph>();
    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulSource")
          .AddArg("checkpointing", true)
          .AddArg("epoch_size", epoch_size)
          .AddOutput("data", "cpu")),
      "dummy_src");

    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulOp")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("processed", "cpu")),
      "dummy_op");

    exe->Build(graph.get(), {"processed_cpu"});
    return std::pair{std::move(exe), std::move(graph)};
  };

  if (this->IsSeparated())
    EXPECT_THROW(
      this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size),
      DALIException);
  else
    this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size);
}

TYPED_TEST(ExecutorTest, PipelineCheckpointingMixed) {
  constexpr int epoch_size = 4;
  auto prepare_executor_and_graph = [&] {
    auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe->EnableCheckpointing(true);
    exe->Init();

    auto graph = std::make_unique<OpGraph>();
    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulSource")
          .AddArg("checkpointing", true)
          .AddArg("epoch_size", epoch_size)
          .AddOutput("data1", "cpu")),
      "dummy_src");

    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulOp")
          .AddArg("device", "mixed")
          .AddInput("data1", "cpu")
          .AddOutput("data2", "gpu")),
      "dummy_op1");

    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulOp")
          .AddArg("device", "gpu")
          .AddInput("data2", "gpu")
          .AddOutput("processed", "gpu")),
      "dummy_op2");

    exe->Build(graph.get(), {"processed_gpu"});
    return std::pair{std::move(exe), std::move(graph)};
  };

  if (this->IsSeparated())
    EXPECT_THROW(
      this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size),
      DALIException);
  else
    this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size);
}

}  // namespace dali
