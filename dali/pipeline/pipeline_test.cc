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

#include "dali/pipeline/pipeline.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "dali/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/util/copy.h"
#include "dali/test/dali_test.h"
#include "dali/util/image.h"

namespace dali {

template <typename ThreadCount>
class PipelineTest : public DALITest {
 public:
  inline void SetUp() {
    DALITest::SetUp();
    DALITest::DecodeJPEGS(DALI_RGB);
  }

  template <typename T>
  inline void CompareData(const T* data, const T* ground_truth, int n) {
    CUDA_CALL(cudaDeviceSynchronize());
    vector<T> tmp_cpu(n);
    CUDA_CALL(cudaMemcpy(tmp_cpu.data(), data, sizeof(T)*n, cudaMemcpyDefault));

    vector<double> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(static_cast<double>(tmp_cpu[i]) - static_cast<double>(ground_truth[i]));
    }
    double mean, std;
    DALITest::MeanStdDev(abs_diff, &mean, &std);

#ifndef NDEBUG
    cout << "num: " << abs_diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif

    ASSERT_LT(mean, 0.000001);
    ASSERT_LT(std, 0.000001);
  }

  inline OpGraph& GetGraph(Pipeline *pipe) {
    return pipe->graph_;
  }
};

template <int number_of_threads>
struct ThreadCount {
  static const int nt = number_of_threads;
};

class PipelineTestOnce : public PipelineTest<ThreadCount<1>> {
};

typedef ::testing::Types<ThreadCount<1>,
                         ThreadCount<2>,
                         ThreadCount<3>,
                         ThreadCount<4>> NumThreads;
TYPED_TEST_CASE(PipelineTest, NumThreads);

TEST_F(PipelineTestOnce, TestInputNotKnown) {
  Pipeline pipe(1, 1, 0);

  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("copy_out", "cpu")),
      std::runtime_error);
}

TEST_F(PipelineTestOnce, TestEnforceCPUOpConstraints) {
  Pipeline pipe(1, 1, 0);

  // Inputs must be know to the pipeline, i.e. ops
  // must be added in a topological ordering.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("copy_out", "cpu")),
      std::runtime_error);

  pipe.AddOperator(
      OpSpec("ExternalSource")
      .AddArg("device", "gpu")
      .AddOutput("data", "gpu"));

  // Inputs to CPU ops must be on CPU
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "gpu")
          .AddOutput("copy_out", "cpu")),
      std::runtime_error);

  // Inputs to CPU ops must already exist on CPU,
  // we do not auto-copy them from gpu to cpu.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("copy_out", "cpu")),
      std::runtime_error);

  pipe.AddOperator(
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddOutput("data_2", "cpu"));

  pipe.AddOperator(
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddOutput("data_3", "cpu"));

  // Outputs must have unique names.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data_2", "cpu")
          .AddOutput("data_3", "cpu")),
      std::runtime_error);

  // All data must have unique names regardless
  // of the device they exist on.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data_2", "cpu")
          .AddOutput("data", "cpu")),
      std::runtime_error);

  // CPU ops can only produce CPU outputs
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data_2", "cpu")
          .AddOutput("data_4", "gpu")),
      std::runtime_error);
}

TEST_F(PipelineTestOnce, TestEnforceGPUOpConstraints) {
  Pipeline pipe(1, 1, 0);

  // Inputs must be know to the pipeline, i.e. ops
  // must be added in a topological ordering.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data", "gpu")
          .AddOutput("copy_out", "gpu")),
      std::runtime_error);

  pipe.AddOperator(
      OpSpec("ExternalSource")
      .AddArg("device", "gpu")
      .AddOutput("data", "gpu"));

  // CPU inputs to GPU ops must be on CPU, we will
  // not copy them back to the host.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data", "cpu")
          .AddOutput("copy_out", "gpu")),
      std::runtime_error);

  pipe.AddOperator(
      OpSpec("ExternalSource")
      .AddArg("device", "gpu")
      .AddOutput("data_2", "gpu"));

  pipe.AddOperator(
      OpSpec("ExternalSource")
      .AddArg("device", "gpu")
      .AddOutput("data_3", "gpu"));

  // Outputs must have unique names.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data_2", "gpu")
          .AddOutput("data_3", "gpu")),
      std::runtime_error);

  pipe.AddOperator(
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddOutput("data_4", "cpu"));

  // All data must have unique names regardless
  // of the device they exist on.
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data_2", "gpu")
          .AddOutput("data_4", "gpu")),
      std::runtime_error);

  // GPU ops can only produce GPU outputs
  ASSERT_THROW(
      pipe.AddOperator(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data_2", "gpu")
          .AddOutput("data_4", "cpu")),
      std::runtime_error);
}

TEST_F(PipelineTestOnce, TestTriggerToContiguous) {
  Pipeline pipe(1, 1, 0);

  pipe.AddExternalInput("data");

  pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "cpu")
      .AddOutput("data_copy", "gpu"));

  vector<std::pair<string, string>> outputs = {{"data_copy", "gpu"}};
  pipe.Build(outputs);

  OpGraph &graph = this->GetGraph(&pipe);

  // Validate the graph
  ASSERT_EQ(graph.NumCPUOp(), 1);
  ASSERT_EQ(graph.NumMixedOp(), 1);
  ASSERT_EQ(graph.NumGPUOp(), 1);

  ASSERT_EQ(graph.mixed_op(0).name(), "MakeContiguous");

  // Validate the source op
  auto& node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);

  // Validate the MakeContiguous op
  auto& node2 = graph.node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 1);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(node2.children.count(2), 1);

  // Validate the copy op
  auto& node3 = graph.node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 0);
  ASSERT_EQ(node3.parents.size(), 1);
  ASSERT_EQ(node3.parents.count(1), 1);
}

TEST_F(PipelineTestOnce, TestTriggerCopyToDevice) {
  Pipeline pipe(1, 1, 0);

  pipe.AddExternalInput("data");

  pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "gpu")
      .AddOutput("data_copy", "gpu"));

  vector<std::pair<string, string>> outputs = {{"data_copy", "gpu"}};
  pipe.Build(outputs);

  OpGraph &graph = this->GetGraph(&pipe);

  // Validate the graph
  ASSERT_EQ(graph.NumCPUOp(), 1);
  ASSERT_EQ(graph.NumMixedOp(), 1);
  ASSERT_EQ(graph.NumGPUOp(), 1);

  ASSERT_EQ(graph.mixed_op(0).name(), "MakeContiguous");

  // Validate the source op
  auto& node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);

  // Validate the MakeContiguous op
  auto& node2 = graph.node(1);
  ASSERT_EQ(node2.id, 1);
  ASSERT_EQ(node2.children.size(), 1);
  ASSERT_EQ(node2.parents.size(), 1);
  ASSERT_EQ(node2.parents.count(0), 1);
  ASSERT_EQ(node2.children.count(2), 1);

  // Validate the copy op
  auto& node3 = graph.node(2);
  ASSERT_EQ(node3.id, 2);
  ASSERT_EQ(node3.children.size(), 0);
  ASSERT_EQ(node3.parents.size(), 1);
  ASSERT_EQ(node3.parents.count(1), 1);
}

TYPED_TEST(PipelineTest, TestExternalSource) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.size();

  Pipeline pipe(batch_size, num_thread, 0);

  pipe.AddExternalInput("data");

  OpGraph &graph = this->GetGraph(&pipe);

  // Validate the graph
  ASSERT_EQ(graph.NumCPUOp(), 1);
  ASSERT_EQ(graph.NumMixedOp(), 0);
  ASSERT_EQ(graph.NumGPUOp(), 0);

  // Validate the gpu source op
  auto& node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 0);
}

TYPED_TEST(PipelineTest, TestSerialization) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.size();

  Pipeline pipe(batch_size, num_thread, 0);


  TensorList<CPUBackend> batch;
  this->MakeJPEGBatch(&batch, batch_size);

  pipe.AddExternalInput("data");
  pipe.SetExternalInput("data", batch);

  pipe.AddOperator(
      OpSpec("HostDecoder")
      .AddArg("device", "cpu")
      .AddInput("data", "cpu")
      .AddOutput("decoded", "cpu"));

  pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("decoded", "gpu")
      .AddOutput("copied", "gpu"));

  auto serialized = pipe.SerializeToProtobuf();

  Pipeline loaded_pipe(serialized, batch_size, num_thread, 0);
  loaded_pipe.SetExternalInput("data", batch);

  vector<std::pair<string, string>> outputs = {{"copied", "gpu"}};

  pipe.Build(outputs);
  loaded_pipe.Build(outputs);

  OpGraph &original_graph = this->GetGraph(&pipe);
  OpGraph &loaded_graph = this->GetGraph(&loaded_pipe);

  // Validate the graph contains the same ops
  ASSERT_EQ(loaded_graph.NumCPUOp(), original_graph.NumCPUOp());
  ASSERT_EQ(loaded_graph.NumMixedOp(), original_graph.NumMixedOp());
  ASSERT_EQ(loaded_graph.NumGPUOp(), original_graph.NumGPUOp());
}

/*
TYPED_TEST(PipelineTest, TestSinglePrefetchOp) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.size();
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  Pipeline pipe(batch_size, num_thread, stream, 0, true);

  // Add an external source

  // Add a single prefetch op
  pipe.AddTransform(OpSpec("Copy")
      .AddArg("stage", "Prefetch"));

  // Build the pipeline
  pipe.Build();

  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    CUDA_CALL(cudaStreamSynchronize(stream));

    // Verify the results
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}

TYPED_TEST(PipelineTest, TestNoOps) {
  int num_thread = TypeParam::nt;

  int batch_size = this->jpegs_.size();
  Pipeline pipe(batch_size, num_thread, 0, 0, true);

  Batch<CPUBackend> *batch =
    CreateJPEGBatch<CPUBackend>(this->jpegs_, this->jpeg_sizes_, batch_size);


  // Add a data reader
  pipe.AddDataReader(
      OpSpec("BatchDataReader")
      .AddArg("jpeg_folder", image_folder)
      );

  // Build the pipeline
  pipe.Build();

  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    CUDA_CALL(cudaStreamSynchronize(0));

    // Verify the results
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}

TYPED_TEST(PipelineTest, TestSingleForwardOp) {
  int num_thread = TypeParam::nt;

  int batch_size = this->jpegs_.size();
  Pipeline pipe(batch_size, num_thread, 0, 0, true);

  Batch<CPUBackend> *batch =
    CreateJPEGBatch<CPUBackend>(this->jpegs_, this->jpeg_sizes_, batch_size);

  // Add a data reader
  pipe.AddDataReader(
      OpSpec("BatchDataReader")
      .AddArg("jpeg_folder", image_folder)
      );

  // Add a single op to the forward stage
  pipe.AddTransform(OpSpec("Copy")
      .AddArg("stage", "Forward")
      );

  // Build the pipeline
  pipe.Build();

  // Run the pipeline
  for (int i = 0; i < 5; ++i) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();

    // Verify the results
    CUDA_CALL(cudaStreamSynchronize(0));
    this->CompareData(
        pipe.output_batch().template data<uint8>(),
        batch->template data<uint8>(),
        batch->size());
  }
}
*/
}  // namespace dali
