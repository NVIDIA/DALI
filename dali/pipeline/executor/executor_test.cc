// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <chrono>
#include <future>

#include "dali/test/dali_test_decoder.h"
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/async_separated_pipelined_executor.h"

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

  inline OpSpec PrepareSpec(OpSpec spec) const {
    spec.AddArg("batch_size", batch_size_)
      .AddArg("num_threads", num_threads_);
    return spec;
  }

  inline void PruneGraph(ExecutorBase *exe) const {
    exe->PruneUnusedGraphNodes();
  }

  // TODO(klecki): adjust to refactored code
  vector<HostWorkspace> CPUData(ExecutorBase *exe, int idx) const {
    // return std::get<static_cast<int>(OpType::CPU)>(exe->wss_[idx].op_data);
    return {};
  }

  vector<MixedWorkspace> MixedData(ExecutorBase *exe, int idx) const {
    // return std::get<static_cast<int>(OpType::MIXED)>(exe->wss_[idx].op_data);
    return {};
  }

  vector<DeviceWorkspace> GPUData(ExecutorBase *exe, int idx) const {
    // return std::get<static_cast<int>(OpType::GPU)>(exe->wss_[idx].op_data);
    return {};
  }

  void VerifyDecode(const uint8 *img, int h, int w, int img_id) const {
    // Load the image to host
    uint8 *host_img = new uint8[h*w*c_];
    CUDA_CALL(cudaMemcpy(host_img, img, h*w*c_, cudaMemcpyDefault));

#if DALI_DEBUG
    WriteHWCImage(host_img, h, w, c_, std::to_string(img_id) + "-img");
#endif
    GenericDecoderTest::VerifyDecode(host_img, h, w, jpegs_, img_id);
    delete [] host_img;
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
    HostWorkspace &hws = host_workspaces[0];
    ASSERT_EQ(hws.NumInput(), 0);
    ASSERT_EQ(hws.NumOutput(), 1);
    ASSERT_EQ(hws.NumOutputAtIdx(0), this->batch_size_);
    ASSERT_TRUE(hws.OutputIsType<CPUBackend>(0));

    auto mixed_workspaces = this->MixedData(exe.get(), i);
    ASSERT_EQ(mixed_workspaces.size(), 1);
    MixedWorkspace &mws = mixed_workspaces[0];
    ASSERT_EQ(mws.NumInput(), 1);
    ASSERT_EQ(mws.NumInputAtIdx(0), this->batch_size_);
    ASSERT_TRUE(mws.InputIsType<CPUBackend>(0));
    ASSERT_EQ(mws.NumOutput(), 1);
    ASSERT_TRUE(mws.OutputIsType<GPUBackend>(0));

    auto device_workspaces = this->GPUData(exe.get(), i);
    ASSERT_EQ(device_workspaces.size(), 1);
    DeviceWorkspace &dws = device_workspaces[0];
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
          .AddOutput("data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("HostDecoder")
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
  this->MakeJPEGBatch(&tl, this->batch_size_);
  src_op->SetDataSource(tl);

  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();

  DeviceWorkspace ws;
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
          .AddOutput("data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("HostDecoder")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("images", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("images", "cpu")
          .AddOutput("final_images", "cpu")), "");

  vector<string> outputs = {"final_images_cpu"};
  int cb_counter = 0;
  std::promise<void> barrier;
  auto barrier_future = barrier.get_future();
  exe->SetCompletionCallback([&cb_counter, &barrier]() mutable {
    ++cb_counter;
    barrier.set_value();
  });

  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  this->MakeJPEGBatch(&tl, this->batch_size_);
  src_op->SetDataSource(tl);

  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();

  DeviceWorkspace ws;
  exe->Outputs(&ws);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_EQ(ws.NumOutput(), 1);
  auto status = barrier_future.wait_for(std::chrono::seconds(5));
  ASSERT_EQ(status, std::future_status::ready);
  ASSERT_EQ(cb_counter, 1);
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
          .AddOutput("data", "cpu")), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("HostDecoder")
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
  int cb_counter = 0;
  std::promise<void> barrier_0, barrier_1;
  auto barrier_future_0 = barrier_0.get_future();
  auto barrier_future_1 = barrier_1.get_future();
  exe->SetCompletionCallback([&cb_counter, &barrier_0, &barrier_1]() mutable {
    if (cb_counter == 0) {
      barrier_0.set_value();
    }
    if (cb_counter == 1) {
      barrier_1.set_value();
    }
    ++cb_counter;
  });
  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  this->MakeJPEGBatch(&tl, this->batch_size_*2);

  // Split the batch into two
  TensorList<CPUBackend> tl2;
  TensorList<CPUBackend> tl1;
  vector<Dims> shape1(batch_size), shape2(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    shape1[i] = tl.tensor_shape(i);
    shape2[i] = tl.tensor_shape(i+batch_size);
  }
  tl1.Resize(shape1);
  tl2.Resize(shape2);
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

  // Run twice without getting the results
  src_op->SetDataSource(tl1);
  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();

  auto status_0 = barrier_future_0.wait_for(std::chrono::seconds(5));
  ASSERT_EQ(status_0, std::future_status::ready);
  ASSERT_EQ(cb_counter, 1);
  src_op->SetDataSource(tl2);
  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();

  // Verify that both sets of results are correct
  DeviceWorkspace ws;
  exe->Outputs(&ws);
  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_TRUE(ws.OutputIsType<GPUBackend>(0));
  TensorList<GPUBackend> &res1 = ws.Output<GPUBackend>(0);
  for (int i = 0; i < batch_size; ++i) {
    this->VerifyDecode(
        res1.template tensor<uint8>(i),
        res1.tensor_shape(i)[0],
        res1.tensor_shape(i)[1], i);
  }

  exe->Outputs(&ws);
  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_TRUE(ws.OutputIsType<GPUBackend>(0));

  auto status_1 = barrier_future_1.wait_for(std::chrono::seconds(5));
  ASSERT_EQ(status_1, std::future_status::ready);
  ASSERT_EQ(cb_counter, 2);
  TensorList<GPUBackend> &res2 = ws.Output<GPUBackend>(0);
  for (int i = 0; i < batch_size; ++i) {
    this->VerifyDecode(
        res2.template tensor<uint8>(i),
        res2.tensor_shape(i)[0],
        res2.tensor_shape(i)[1],
        i+batch_size);
  }
}

}  // namespace dali
