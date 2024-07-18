// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/builtin/copy.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_decoder.h"
#include "dali/util/image.h"
#include "dali/test/dali_test_utils.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {

namespace {

template <typename Pred>
auto CountNodes(const graph::OpGraph &graph, Pred &&pred) {
  return std::count_if(graph.OpNodes().begin(), graph.OpNodes().end(), std::forward<Pred>(pred));
}

auto CountNodes(const graph::OpGraph &graph, OpType type) {
  return CountNodes(graph, [type](auto &node) { return node.op_type == type; });
}

}  // namespace

template <typename ThreadCount>
class PipelineTest : public DALITest {
 public:
  inline void SetUp() override {
    DALITest::SetUp();
    DALITest::DecodeJPEGS(DALI_RGB);
  }

  void RunTestEnforce(const string &dev1, const string &dev2) {
    Pipeline pipe(1, 1, 0);

    // TODO(michalz): This is a totally artificial limitation. Remove the constraint and the tests.

    // Inputs must be know to the pipeline, i.e. ops
    // must be added in a topological ordering.
    ASSERT_THROW(
      pipe.AddOperator(
        OpSpec("Copy")
          .AddArg("device", dev1)
          .AddInput("data", dev1)
          .AddOutput("copy_out", dev1)),
      std::runtime_error);

    pipe.AddOperator(
      OpSpec("ExternalSource")
        .AddArg("device", "gpu")
        .AddOutput("data", "gpu"));

    // TODO(michalz): Remove this constraint and the tests. This should be a build-time error,
    //                with old executor, not a construction-time error.

    // For dev1 = "cpu": Inputs to CPU ops must be on CPU,
    //                   we do not auto-copy them from gpu to cpu.
    // For dev1 = "gpu": CPU inputs to GPU ops must be on CPU,
    //                   we will not copy them back to the host.
    ASSERT_THROW(
      pipe.AddOperator(
        OpSpec("Copy")
          .AddArg("device", dev1)
          .AddInput("data", dev2)
          .AddOutput("copy_out", dev1)),
      std::runtime_error);

    if (dev1 == "cpu") {
      // Inputs to CPU ops must already exist on CPU,
      // we do not auto-copy them from gpu to cpu.
      ASSERT_THROW(
        pipe.AddOperator(
          OpSpec("Copy")
            .AddArg("device", dev1)
            .AddInput("data", dev1)
            .AddOutput("copy_out", dev1)),
        std::runtime_error);
    }

    pipe.AddOperator(
      OpSpec("ExternalSource")
        .AddArg("device", dev1)
        .AddOutput("data_2", dev1));

    pipe.AddOperator(
      OpSpec("ExternalSource")
        .AddArg("device", dev1)
        .AddOutput("data_3", dev1));

    // Outputs must have unique names.
    ASSERT_THROW(
      pipe.AddOperator(
        OpSpec("Copy")
          .AddArg("device", dev1)
          .AddInput("data_2", dev1)
          .AddOutput("data_3", dev1)),
      std::runtime_error);

    if (dev1 == "gpu") {
      pipe.AddOperator(
        OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddOutput("data_4", "cpu"));
    }
    // All data must have unique names regardless
    // of the device they exist on.
    ASSERT_THROW(
      pipe.AddOperator(
        OpSpec("Copy")
          .AddArg("device", dev1)
          .AddInput("data_2", dev1)
          .AddOutput("data", dev1)),
      std::runtime_error);


    // CPU ops can only produce CPU outputs
    ASSERT_THROW(
      pipe.AddOperator(
        OpSpec("Copy")
          .AddArg("device", dev1)
          .AddInput("data_2", dev1)
          .AddOutput("data_4", dev2)),
      std::runtime_error);
  }

  void RunTestTrigger(const string &dev) {
    Pipeline pipe(1, 1, 0);

    pipe.AddExternalInput("data");

    pipe.AddOperator(
      OpSpec("Copy")
        .AddArg("device", "gpu")
        .AddInput("data", dev)
        .AddOutput("data_copy", "gpu"));

    vector<std::pair<string, string>> outputs = {{"data_copy", "gpu"}};
    pipe.Build(outputs);

    auto &graph = this->GetGraph(&pipe);

      // Validate the graph
    EXPECT_EQ(CountNodes(graph, OpType::CPU), 1);
    EXPECT_EQ(CountNodes(graph, OpType::MIXED), 1);
    EXPECT_EQ(CountNodes(graph, OpType::GPU), 2);

    ASSERT_EQ(graph.OpNodes().size(), 4);
    auto it = graph.OpNodes().begin();
    graph::OpNode &node1 = *it++;
    graph::OpNode &node2 = *it++;
    graph::OpNode &node3 = *it++;
    graph::OpNode &node4 = *it++;

    // The graph is linear, so topological sort is unambiguous
    EXPECT_EQ(node1.instance_name, "data");
    EXPECT_EQ(node1.spec.SchemaName(), "ExternalSource");
    EXPECT_EQ(node1.op_type, OpType::CPU);

    EXPECT_EQ(node2.spec.SchemaName(), "MakeContiguous");
    EXPECT_EQ(node2.op_type, OpType::MIXED);

    EXPECT_EQ(node3.spec.SchemaName(), "Copy");
    EXPECT_EQ(node3.op_type, OpType::GPU);

    EXPECT_EQ(node4.spec.SchemaName(), "MakeContiguous");
    EXPECT_EQ(node4.op_type, OpType::GPU);

    EXPECT_EQ(node1.inputs.size(), 0);
    ASSERT_EQ(node1.outputs.size(), 1_uz);
    ASSERT_EQ(node1.outputs[0]->consumers.size(), 1_uz);
    EXPECT_EQ(node1.outputs[0]->consumers[0].op, &node2);

    ASSERT_EQ(node2.inputs.size(), 1);
    EXPECT_EQ(node2.inputs[0]->producer.op, &node1);
    ASSERT_EQ(node2.outputs.size(), 1);
    ASSERT_EQ(node2.outputs[0]->consumers.size(), 1);
    EXPECT_EQ(node2.outputs[0]->consumers[0].op, &node3);

    ASSERT_EQ(node3.inputs.size(), 1);
    EXPECT_EQ(node3.inputs[0]->producer.op, &node2);
    ASSERT_EQ(node3.outputs.size(), 1);
    ASSERT_EQ(node3.outputs[0]->consumers.size(), 1);
    EXPECT_EQ(node3.outputs[0]->consumers[0].op, &node4);

    ASSERT_EQ(node4.inputs.size(), 1);
    EXPECT_EQ(node4.inputs[0]->producer.op, &node3);
    ASSERT_EQ(node4.outputs.size(), 1);
    EXPECT_TRUE(node4.outputs[0]->pipeline_output);
    EXPECT_TRUE(node4.outputs[0]->consumers.empty());
  }

  inline auto &GetGraph(Pipeline *pipe) {
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
TYPED_TEST_SUITE(PipelineTest, NumThreads);

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
  RunTestEnforce("cpu", "gpu");
}

TEST_F(PipelineTestOnce, TestEnforceGPUOpConstraints) {
  RunTestEnforce("gpu", "cpu");
}

TEST_F(PipelineTestOnce, TestTriggerCopyToDevice) {
  RunTestTrigger("gpu");
}

TYPED_TEST(PipelineTest, TestExternalSource) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.nImages();

  Pipeline pipe(batch_size, num_thread, 0);

  pipe.AddExternalInput("data");
  pipe.Build({{"data", "cpu"}});

  auto &graph = this->GetGraph(&pipe);

  // Validate the graph
  EXPECT_EQ(CountNodes(graph, OpType::CPU), 2);
  EXPECT_EQ(CountNodes(graph, OpType::MIXED), 0);
  EXPECT_EQ(CountNodes(graph, OpType::GPU), 0);

  // Validate the gpu source op
  auto it = graph.OpNodes().begin();
  graph::OpNode &node_external_source = *it++;
  EXPECT_EQ(node_external_source.inputs.size(), 0);
  EXPECT_EQ(node_external_source.outputs.size(), 1);
  EXPECT_EQ(node_external_source.instance_name, "data");


  graph::OpNode &node_make_contiguous = *it++;
  ASSERT_EQ(node_make_contiguous.inputs.size(), 1);
  ASSERT_EQ(node_make_contiguous.outputs.size(), 1);
  EXPECT_TRUE(node_make_contiguous.outputs[0]->consumers.empty());
  EXPECT_NE(node_make_contiguous.instance_name.find("MakeContiguous"), std::string::npos);
}

TYPED_TEST(PipelineTest, TestSerialization) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.nImages();

  Pipeline pipe(batch_size, num_thread, 0);

  pipe.AddExternalInput("data");

  pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "gpu")
      .AddOutput("copied", "gpu"));

  auto serialized = pipe.SerializeToProtobuf();

  Pipeline loaded_pipe(serialized, batch_size, num_thread, 0);

  vector<std::pair<string, string>> outputs = {{"copied", "gpu"}};

  pipe.Build(outputs);
  loaded_pipe.Build(outputs);

  auto &original_graph = this->GetGraph(&pipe);
  auto &loaded_graph = this->GetGraph(&loaded_pipe);

  // Validate the graph contains the same ops
  EXPECT_EQ(CountNodes(loaded_graph, OpType::CPU), CountNodes(original_graph, OpType::CPU));
  EXPECT_EQ(CountNodes(loaded_graph, OpType::MIXED), CountNodes(original_graph, OpType::MIXED));
  EXPECT_EQ(CountNodes(loaded_graph, OpType::GPU), CountNodes(original_graph, OpType::GPU));
}

class DummyPresizeOpCPU : public Operator<CPUBackend> {
 public:
  explicit DummyPresizeOpCPU(const OpSpec &spec)
      : Operator<CPUBackend>(spec) {
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    int num_samples = input.shape().num_samples();
    auto &output = ws.Output<CPUBackend>(0);
    auto tmp_size = output.capacity();
    output.set_type<size_t>();
    output.Resize(uniform_list_shape(num_samples, std::vector<int64_t>{2}));
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto *out = output.mutable_tensor<size_t>(sample_idx);
      out[0] = tmp_size;
      out[1] = input.capacity();
    }
  }
};

class DummyPresizeOpGPU : public Operator<GPUBackend> {
 public:
  explicit DummyPresizeOpGPU(const OpSpec &spec)
      : Operator<GPUBackend>(spec) {
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    int num_samples = input.shape().num_samples();
    auto &output = ws.Output<GPUBackend>(0);
    output.set_type<size_t>();
    size_t tmp_size[2] = {output.capacity(), input.capacity()};
    output.Resize(uniform_list_shape(num_samples, std::vector<int64_t>{2}));
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto *out = output.mutable_tensor<size_t>(sample_idx);
      CUDA_CALL(cudaStreamSynchronize(ws.stream()));
      CUDA_CALL(cudaMemcpy(out, &tmp_size, sizeof(size_t) * 2, cudaMemcpyDefault));
    }
  }
};

class DummyPresizeOpMixed : public Operator<MixedBackend> {
 public:
  explicit DummyPresizeOpMixed(const OpSpec &spec)
      : Operator<MixedBackend>(spec) {
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    auto &input = ws.Input<CPUBackend>(0);
    int num_samples = input.shape().num_samples();
    auto &output = ws.Output<GPUBackend>(0);
    output.set_type<size_t>();
    size_t tmp_size[2] = {output.capacity(), input.capacity()};
    output.Resize(uniform_list_shape(num_samples, std::vector<int64_t>{2}));
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto *out = output.mutable_tensor<size_t>(sample_idx);
      CUDA_CALL(cudaStreamSynchronize(ws.stream()));
      CUDA_CALL(cudaMemcpy(out, &tmp_size, sizeof(size_t) * 2, cudaMemcpyDefault));
    }
  }
};

DALI_REGISTER_OPERATOR(DummyPresizeOp, DummyPresizeOpCPU, CPU);
DALI_REGISTER_OPERATOR(DummyPresizeOp, DummyPresizeOpGPU, GPU);
DALI_REGISTER_OPERATOR(DummyPresizeOp, DummyPresizeOpMixed, Mixed);

DALI_SCHEMA(DummyPresizeOp)
  .DocStr("Dummy")
  .NumInput(1)
  .NumOutput(1);

TEST_F(PipelineTestOnce, TestPresize) {
  const int batch_size = 1;
  const int num_thread = 1;
  const bool pipelined = false;
  const bool async =  false;
  DALIImageType img_type = DALI_RGB;

  const int presize_val_CPU = 11;
  const int presize_val_Mixed = 157;
  const int presize_val_GPU = 971;
  const int presize_val_default = 55;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1, pipelined, 3,
      async,
      presize_val_default);

  TensorList<CPUBackend> data;
  test::MakeRandomBatch(data, batch_size);
  pipe.AddExternalInput("raw_jpegs");

  pipe.AddOperator(
      OpSpec("DummyPresizeOp")
      .AddArg("device", "cpu")
      .AddArg("bytes_per_sample_hint", presize_val_CPU)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("out_1", "cpu"));

  pipe.AddOperator(
      OpSpec("DummyPresizeOp")
      .AddArg("device", "cpu")
      .AddArg("bytes_per_sample_hint", presize_val_CPU)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("out_2", "cpu"));

  pipe.AddOperator(
      OpSpec("DummyPresizeOp")
      .AddArg("device", "mixed")
      .AddArg("bytes_per_sample_hint", presize_val_Mixed)
      .AddInput("out_2", "cpu")
      .AddOutput("out_3", "gpu"));

  pipe.AddOperator(
      OpSpec("MakeContiguous")
      .AddArg("device", "mixed")
      .AddInput("out_2", "cpu")
      .AddOutput("out_4", "gpu"));

  pipe.AddOperator(
      OpSpec("DummyPresizeOp")
      .AddArg("device", "gpu")
      .AddArg("bytes_per_sample_hint", presize_val_GPU)
      .AddInput("out_4", "gpu")
      .AddOutput("out_5", "gpu"));

  pipe.AddOperator(
      OpSpec("DummyPresizeOp")
      .AddArg("device", "gpu")
      .AddArg("bytes_per_sample_hint", presize_val_GPU)
      .AddInput("out_4", "gpu")
      .AddOutput("out_6", "gpu"));

  pipe.AddOperator(
      OpSpec("DummyPresizeOp")
      .AddArg("device", "gpu")
      .AddInput("out_4", "gpu")
      .AddOutput("out_7", "gpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"out_1", "cpu"}, {"out_2", "cpu"},
                                               {"out_3", "gpu"}, {"out_5", "gpu"},
                                               {"out_6", "gpu"}, {"out_7", "gpu"}};

  pipe.Build(outputs);
  pipe.SetExternalInput("raw_jpegs", data);
  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  // we should not presize CPU buffers if they are not pinned
  ASSERT_EQ(*(ws.Output<CPUBackend>(0).tensor<size_t>(0)), 0);

  // this one is also going through mixed CPU -> GPU operator, so it is pinned and presized
  ASSERT_EQ(*(ws.Output<CPUBackend>(1).tensor<size_t>(0)), presize_val_CPU);

  size_t tmp[2];
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(2).tensor<size_t>(0),
            sizeof(size_t) * 2, cudaMemcpyDefault));
  ASSERT_EQ(tmp[0], presize_val_Mixed);
  ASSERT_EQ(tmp[1], 2 * sizeof(size_t));

  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(3).tensor<size_t>(0),
            sizeof(size_t) * 2, cudaMemcpyDefault));
  ASSERT_EQ(tmp[0], presize_val_GPU);
  ASSERT_EQ(tmp[1], 2 * sizeof(size_t));

  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(4).tensor<size_t>(0),
            sizeof(size_t) * 2, cudaMemcpyDefault));
  ASSERT_EQ(tmp[0], presize_val_GPU);
  ASSERT_EQ(tmp[1], 2 * sizeof(size_t));

  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(5).tensor<size_t>(0),
            sizeof(size_t) * 2, cudaMemcpyDefault));
  ASSERT_EQ(tmp[0], presize_val_default);
  ASSERT_EQ(tmp[1], 2 * sizeof(size_t));
}

TYPED_TEST(PipelineTest, TestSeedSet) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.nImages();
  constexpr int seed_set = 567;

  Pipeline pipe(batch_size, num_thread, 0);


  TensorList<CPUBackend> batch;
  test::MakeRandomBatch(batch, batch_size);

  pipe.AddExternalInput("data");

  pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "cpu")
      .AddArg("seed", seed_set)
      .AddInput("data", "cpu")
      .AddOutput("copied0", "cpu"), "copy1");

  pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddArg("seed", seed_set)
      .AddInput("copied0", "gpu")
      .AddOutput("copied", "gpu"), "copy2");

  vector<std::pair<string, string>> outputs = {{"copied", "gpu"}};

  pipe.Build(outputs);

  pipe.SetExternalInput("data", batch);

  graph::OpGraph &original_graph = this->GetGraph(&pipe);

  // Check if seed can be manually set
  EXPECT_EQ(original_graph.GetOp("copy1")->spec.GetArgument<int64_t>("seed"), seed_set);
  EXPECT_EQ(original_graph.GetOp("copy2")->spec.GetArgument<int64_t>("seed"), seed_set);
  EXPECT_NE(original_graph.GetOp("data")->spec.GetArgument<int64_t>("seed"), seed_set);
}


class PrefetchedPipelineTest : public DALITest {
 public:
  int batch_size_ = 5, num_threads_ = 1;
};

TEST_F(PrefetchedPipelineTest, SetQueueSizesSeparatedFail) {
  Pipeline pipe(this->batch_size_, 4, 0);
  // By default we are non-separated execution
  ASSERT_THROW(pipe.SetQueueSizes(5, 3), std::runtime_error);
}

TEST_F(PrefetchedPipelineTest, SetExecutionTypesFailAfterBuild) {
  Pipeline pipe(this->batch_size_, 4, 0);
  pipe.AddExternalInput("data");
  pipe.AddOperator(OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data", "gpu")
          .AddOutput("final_images", "gpu"));

  vector<std::pair<string, string>> outputs = {{"final_images", "gpu"}};
  pipe.Build(outputs);
  ASSERT_THROW(pipe.SetExecutionTypes(), std::runtime_error);
}

TEST_F(PrefetchedPipelineTest, SetQueueSizesFailAfterBuild) {
  Pipeline pipe(this->batch_size_, 4, 0);
  pipe.AddExternalInput("data");
  pipe.AddOperator(OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data", "gpu")
          .AddOutput("final_images", "gpu"));

  vector<std::pair<string, string>> outputs = {{"final_images", "gpu"}};
  pipe.Build(outputs);
  ASSERT_THROW(pipe.SetQueueSizes(2, 2), std::runtime_error);
}

TEST_F(PrefetchedPipelineTest, TestFillQueues) {
  // Test coprime queue sizes
  constexpr int CPU = 5, GPU = 3;
  constexpr int N = CPU + GPU + 5;
  int batch_size = this->batch_size_;

  Pipeline pipe(batch_size, 4, 0);
  // Cannot test async while setting external input - need to make sure that
  pipe.SetExecutionTypes(true, true, true);
  // Test coprime queue sizes
  pipe.SetQueueSizes(CPU, GPU);
  pipe.AddExternalInput("data");
  pipe.AddOperator(OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("data1", "cpu"));
  pipe.AddOperator(OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("data1", "gpu")
          .AddOutput("final_images", "gpu"));

  vector<std::pair<string, string>> outputs = {{"final_images", "gpu"}};
  pipe.Build(outputs);

  TensorList<CPUBackend> tl;
  test::MakeRandomBatch(tl, batch_size * N);

  // Split the batch into 5
  std::array<TensorList<CPUBackend>, N> split_tl;
  std::array<std::vector<TensorShape<>>, N> shapes;
  for (int i = 0; i < N; i++) {
    shapes[i].resize(batch_size);
    for (int j = 0; j < batch_size; j++) {
      shapes[i][j] = tl.tensor_shape(i * batch_size + j);
    }
    split_tl[i].Resize({shapes[i]}, DALI_UINT8);
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < batch_size; j++) {
      std::memcpy(
        split_tl[i].template mutable_tensor<uint8>(j),
        tl.template tensor<uint8>(i * batch_size + j),
        volume(tl.tensor_shape(i * batch_size + j)));
    }
  }

  // Fill queues
  int i = 0;
  int feed_count = pipe.InputFeedCount("data");
  for (; i < feed_count; i++)
    pipe.SetExternalInput("data", split_tl[i]);
  pipe.Prefetch();

  // Now we interleave the calls to Outputs() and Run() for the rest of the batch
  int obtained_outputs = 0;
  for (; i < N; i++) {
    Workspace ws;
    pipe.Outputs(&ws);
    test::CheckResults(ws, batch_size, obtained_outputs++, tl);
    pipe.SetExternalInput("data", split_tl[i]);
    pipe.Run();
  }
}

class DummyOpToAdd : public Operator<CPUBackend> {
 public:
  explicit DummyOpToAdd(const OpSpec &spec) : Operator<CPUBackend>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {}
};

DALI_REGISTER_OPERATOR(DummyOpToAdd, DummyOpToAdd, CPU);

DALI_SCHEMA(DummyOpToAdd)
  .DocStr("DummyOpToAdd")
  .NumInput(1)
  .NumOutput(1);


class DummyOpNoSync : public Operator<CPUBackend> {
 public:
  explicit DummyOpNoSync(const OpSpec &spec) : Operator<CPUBackend>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {}
};

DALI_REGISTER_OPERATOR(DummyOpNoSync, DummyOpNoSync, CPU);

DALI_SCHEMA(DummyOpNoSync)
  .DocStr("DummyOpNoSync")
  .DisallowInstanceGrouping()
  .NumInput(1)
  .NumOutput(1);

TEST(PipelineTest, AddOperator) {
  Pipeline pipe(10, 4, 0);
  int input_0 = pipe.AddExternalInput("data_in0");
  int input_1 = pipe.AddExternalInput("data_in1");

  int first_op = pipe.AddOperator(OpSpec("DummyOpToAdd")
          .AddArg("device", "cpu")
          .AddInput("data_in0", "cpu")
          .AddOutput("data_out0", "cpu"), "first_op");

  int second_op = pipe.AddOperator(OpSpec("DummyOpToAdd")
          .AddArg("device", "cpu")
          .AddInput("data_in1", "cpu")
          .AddOutput("data_out1", "cpu"), "second_op", first_op);
  EXPECT_EQ(first_op, second_op);

  ASSERT_THROW(pipe.AddOperator(OpSpec("Copy"), "another_op", first_op), std::runtime_error);

  int third_op = pipe.AddOperator(OpSpec("DummyOpToAdd")
          .AddArg("device", "cpu")
          .AddArg("seed", 0xDEADBEEF)
          .AddInput("data_in1", "cpu")
          .AddOutput("data_out2", "cpu"), "third_op");

  EXPECT_EQ(third_op, second_op + 1);

  int disallow_sync_op = pipe.AddOperator(OpSpec("DummyOpNoSync")
          .AddArg("device", "cpu")
          .AddInput("data_in0", "cpu")
          .AddOutput("data_out3", "cpu"), "DummyOpNoSync");

  ASSERT_THROW(pipe.AddOperator(OpSpec("DummyOpNoSync")
          .AddArg("device", "cpu")
          .AddInput("data_in0", "cpu")
          .AddOutput("data_out4", "cpu"), "DummyOpNoSync2", disallow_sync_op), std::runtime_error);

  vector<std::pair<string, string>> outputs = {
      {"data_out0", "cpu"}, {"data_out1", "cpu"}, {"data_out2", "cpu"}};
  pipe.Build(outputs);
  ASSERT_TRUE(pipe.IsLogicalIdUsed(0));
  ASSERT_TRUE(pipe.IsLogicalIdUsed(input_0));
  ASSERT_TRUE(pipe.IsLogicalIdUsed(input_1));
  ASSERT_TRUE(pipe.IsLogicalIdUsed(first_op));
  ASSERT_TRUE(pipe.IsLogicalIdUsed(second_op));
  ASSERT_TRUE(pipe.IsLogicalIdUsed(third_op));
  ASSERT_EQ(pipe.GetOperatorNode("first_op")->spec.GetArgument<int64_t>("seed"),
            pipe.GetOperatorNode("second_op")->spec.GetArgument<int64_t>("seed"));
  ASSERT_EQ(pipe.GetOperatorNode("third_op")->spec.GetArgument<int64_t>("seed"), 0xDEADBEEF);
}

TEST(PipelineTest, InputsListing) {
  Pipeline pipe(10, 4, 0);
  pipe.AddExternalInput("ZINPUT");
  pipe.AddExternalInput("AINPUT1");
  pipe.AddExternalInput("AINPUT0");

  pipe.AddOperator(OpSpec("DummyOpToAdd")
          .AddArg("device", "cpu")
          .AddInput("ZINPUT", "cpu")
          .AddOutput("OUTPUT", "cpu"), "first_op");

  pipe.Build({{"AINPUT0", "cpu"}, {"AINPUT1", "cpu"}, {"OUTPUT", "cpu"}});

  ASSERT_EQ(pipe.num_inputs(), 3);
  ASSERT_EQ(pipe.input_name(0), "AINPUT0");
  ASSERT_EQ(pipe.input_name(1), "AINPUT1");
  ASSERT_EQ(pipe.input_name(2), "ZINPUT");
}

TEST(PipelineTest, InputDetails) {
  Pipeline pipe(1, 1, 0);
  pipe.AddExternalInput("INPUT", "cpu", DALI_UINT32, 3, "HWC");
  pipe.AddExternalInput("INPUT2", "gpu", DALI_FLOAT16, -1, "NHWC");
  pipe.AddExternalInput("INPUT3");

  pipe.Build({{"INPUT", "cpu"}, {"INPUT2", "gpu"}, {"INPUT3", "cpu"}});

  EXPECT_EQ(pipe.GetInputLayout("INPUT"), "HWC");
  EXPECT_EQ(pipe.GetInputNdim("INPUT"), 3);
  EXPECT_EQ(pipe.GetInputDtype("INPUT"), DALI_UINT32);

  EXPECT_EQ(pipe.GetInputLayout("INPUT2"), "NHWC");
  EXPECT_EQ(pipe.GetInputNdim("INPUT2"), 4);
  EXPECT_EQ(pipe.GetInputDtype("INPUT2"), DALI_FLOAT16);

  EXPECT_EQ(pipe.GetInputLayout("INPUT3"), "");
  EXPECT_EQ(pipe.GetInputNdim("INPUT3"), -1);
  EXPECT_EQ(pipe.GetInputDtype("INPUT3"), DALI_NO_TYPE);
}

class DummyInputOperator: public InputOperator<CPUBackend> {
 public:
  explicit DummyInputOperator(const OpSpec &spec) : InputOperator<CPUBackend>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    TensorList<CPUBackend> input;
    std::optional<std::string> data_id;
    ForwardCurrentData(input, data_id, ws.GetThreadPool());

    int data = input.tensor<int>(0)[0];
    auto &out0 = ws.Output<CPUBackend>(0);
    auto &out1 = ws.Output<CPUBackend>(1);
    auto out_shape = TensorListShape<-1>(1, 1);
    out_shape.set_tensor_shape(0, {1});

    out0.Resize(out_shape, DALIDataType::DALI_FLOAT);
    out0.mutable_tensor<float>(0)[0] = static_cast<float>(data) * 0.5;

    out1.Resize(out_shape, DALIDataType::DALI_INT32);
    out1.mutable_tensor<int>(0)[0] = data;
  }

  const TensorLayout &in_layout() const override {
    return in_layout_;
  }

  int in_ndim() const override {
    return 1;
  }

  DALIDataType in_dtype() const override {
    return DALIDataType::DALI_INT32;
  }

  TensorLayout in_layout_{};
};

DALI_REGISTER_OPERATOR(DummyInputOperator, DummyInputOperator, CPU);

DALI_SCHEMA(DummyInputOperator)
  .DocStr("DummyInputOperator")
  .DisallowInstanceGrouping()
  .NumInput(0)
  .NumOutput(2);

TEST(PipelineTest, MultiOutputInputOp) {
  Pipeline pipe(1, 1, 0);
  pipe.AddOperator(OpSpec("DummyInputOperator")
    .AddArg("blocking", true)
    .AddArg("no_copy", false)
    .AddOutput("out0", "cpu")
    .AddOutput("out1", "cpu"), "DummyInput");

  pipe.Build({{"out0", "cpu"}, {"out1", "cpu"}});
  int input = 3;
  TensorList<CPUBackend> inp;
  TensorListShape<1> inp_shape(1);
  inp_shape.set_tensor_shape(0, {1});
  inp.Resize(inp_shape, DALIDataType::DALI_INT32);
  inp.mutable_tensor<int>(0)[0] = input;
  pipe.SetExternalInput("DummyInput", inp);

  pipe.Run();
  Workspace ws;
  pipe.Outputs(&ws);

  auto &out0  = ws.Output<CPUBackend>(0);
  ASSERT_EQ(out0.type(), DALIDataType::DALI_FLOAT);
  ASSERT_EQ(out0.tensor<float>(0)[0], static_cast<float>(input) * 0.5f);

  auto &out1  = ws.Output<CPUBackend>(1);
  ASSERT_EQ(out1.type(), DALIDataType::DALI_INT32);
  ASSERT_EQ(out1.tensor<int>(0)[0], input);
}

TEST(PipelineTest, DuplicateInstanceName) {
  Pipeline pipe(1, 1, 0);
  pipe.AddExternalInput("data");

  EXPECT_THROW(pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "gpu")
      .AddOutput("copied", "gpu"), "data"), std::runtime_error);

  EXPECT_NO_THROW(pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "gpu")
      .AddOutput("copied", "gpu"), "data1"));
}

TEST(PipelineTest, AutoName) {
  Pipeline pipe(1, 1, 0);
  pipe.AddExternalInput("data");

  int id = pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "gpu")
      .AddOutput("copied1", "gpu"), 1);

  EXPECT_NO_THROW(pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "gpu")
      .AddOutput("copied2", "gpu"), id));

  EXPECT_NO_THROW(pipe.AddOperator(
      OpSpec("Copy")
      .AddArg("device", "gpu")
      .AddInput("data", "gpu")
      .AddOutput("copied3", "gpu"), id));

  pipe.SetOutputDescs({{ "copied1", "gpu"}, {"copied2", "gpu"}, {"copied3", "gpu"}});

  auto name = make_string("__Copy_", id);

  pipe.Build();
  EXPECT_NE(pipe.GetOperatorNode(name), nullptr);
  EXPECT_NE(pipe.GetOperatorNode(name + "_1"), nullptr);
  EXPECT_NE(pipe.GetOperatorNode(name + "_2"), nullptr);
  EXPECT_EQ(pipe.GetOperatorNode(name + "_3"), nullptr);
}


}  // namespace dali
