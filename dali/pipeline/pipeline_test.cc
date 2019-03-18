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
#include "dali/test/dali_test_decoder.h"
#include "dali/util/image.h"

namespace dali {

template <typename ThreadCount>
class PipelineTest : public DALITest {
 public:
  inline void SetUp() override {
    DALITest::SetUp();
    DALITest::DecodeJPEGS(DALI_RGB);
  }

  void RunTestEnforce(const string &dev1, const string &dev2) {
    Pipeline pipe(1, 1, 0);

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

    OpGraph &graph = this->GetGraph(&pipe);

      // Validate the graph
    ASSERT_EQ(graph.NumOp(OpType::CPU), 1);
    ASSERT_EQ(graph.NumOp(OpType::MIXED), 1);
    ASSERT_EQ(graph.NumOp(OpType::GPU), 1);

    ASSERT_EQ(graph.Node(OpType::MIXED, 0).op->name(), "MakeContiguous");

    // Validate the source op
    auto &node = graph.Node(0);
    ASSERT_EQ(node.id, 0);
    ASSERT_EQ(node.children.size(), 1);
    ASSERT_EQ(node.parents.size(), 0);
    ASSERT_EQ(node.children.count(1), 1);

    // Validate the MakeContiguous op
    auto &node2 = graph.Node(1);
    ASSERT_EQ(node2.id, 1);
    ASSERT_EQ(node2.children.size(), 1);
    ASSERT_EQ(node2.parents.size(), 1);
    ASSERT_EQ(node2.parents.count(0), 1);
    ASSERT_EQ(node2.children.count(2), 1);

    // Validate the copy op
    auto &node3 = graph.Node(2);
    ASSERT_EQ(node3.id, 2);
    ASSERT_EQ(node3.children.size(), 0);
    ASSERT_EQ(node3.parents.size(), 1);
    ASSERT_EQ(node3.parents.count(1), 1);
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

TEST_F(PipelineTestOnce, TestTriggerToContiguous) {
  RunTestTrigger("cpu");
}

TEST_F(PipelineTestOnce, TestTriggerCopyToDevice) {
  RunTestTrigger("gpu");
}

TYPED_TEST(PipelineTest, TestExternalSource) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.nImages();

  Pipeline pipe(batch_size, num_thread, 0);

  pipe.AddExternalInput("data");

  OpGraph &graph = this->GetGraph(&pipe);

  // Validate the graph
  ASSERT_EQ(graph.NumOp(OpType::CPU), 1);
  ASSERT_EQ(graph.NumOp(OpType::MIXED), 0);
  ASSERT_EQ(graph.NumOp(OpType::GPU), 0);

  // Validate the gpu source op
  auto& node = graph.Node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 0);
}

TYPED_TEST(PipelineTest, TestSerialization) {
  int num_thread = TypeParam::nt;
  int batch_size = this->jpegs_.nImages();

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
  ASSERT_EQ(loaded_graph.NumOp(OpType::CPU),
            original_graph.NumOp(OpType::CPU));
  ASSERT_EQ(loaded_graph.NumOp(OpType::MIXED),
            original_graph.NumOp(OpType::MIXED));
  ASSERT_EQ(loaded_graph.NumOp(OpType::GPU),
            original_graph.NumOp(OpType::GPU));
}

class DummyPresizeOpCPU : public Operator<CPUBackend> {
 public:
  explicit DummyPresizeOpCPU(const OpSpec &spec)
      : Operator<CPUBackend>(spec) {
  }

  void RunImpl(Workspace<CPUBackend>* ws, int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto &output = ws->Output<CPUBackend>(idx);
    auto tmp_size = output.capacity();
    output.mutable_data<size_t>();
    output.Resize({2});
    auto out = output.mutable_data<size_t>();
    out[0] = tmp_size;
    out[1] = input.capacity();
  }
};

class DummyPresizeOpGPU : public Operator<GPUBackend> {
 public:
  explicit DummyPresizeOpGPU(const OpSpec &spec)
      : Operator<GPUBackend>(spec) {
  }

  void RunImpl(Workspace<GPUBackend>* ws, int idx) override {
    auto &input = ws->Input<GPUBackend>(0);
    auto &output = ws->Output<GPUBackend>(0);
    output.mutable_data<size_t>();
    size_t tmp_size[2] = {output.capacity(), input.capacity()};
    std::vector< std::vector<Index> > shape {{1}};
    output.Resize(shape);
    auto out = output.mutable_data<size_t>();
    CUDA_CALL(cudaStreamSynchronize(ws->stream()));
    CUDA_CALL(cudaMemcpy(out, &tmp_size, sizeof(size_t) * 2, cudaMemcpyDefault));
  }
};

class DummyPresizeOpMixed : public Operator<MixedBackend> {
 public:
  explicit DummyPresizeOpMixed(const OpSpec &spec)
      : Operator<MixedBackend>(spec) {
  }

  using Operator<MixedBackend>::Run;
  void Run(MixedWorkspace* ws) override {
    auto &input = ws->Input<CPUBackend>(0, 0);
    auto &output = ws->Output<GPUBackend>(0);
    output.mutable_data<size_t>();
    size_t tmp_size[2] = {output.capacity(), input.capacity()};
    std::vector< std::vector<Index> > shape {{1}};
    output.Resize(shape);
    auto out = output.mutable_data<size_t>();
    CUDA_CALL(cudaStreamSynchronize(ws->stream()));
    CUDA_CALL(cudaMemcpy(out, &tmp_size, sizeof(size_t) * 2, cudaMemcpyDefault));
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
  const bool pipelined = true;
  const bool async =  true;
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
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

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
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  // we should not presize CPU buffers if they are not pined
  ASSERT_EQ(*(ws.Output<CPUBackend>(0).data<size_t>()), 0);

  ASSERT_EQ(*(ws.Output<CPUBackend>(1).data<size_t>()), presize_val_CPU);

  size_t tmp[2];
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(2).mutable_data<size_t>(),
            sizeof(size_t) * 2, cudaMemcpyDefault));
  ASSERT_EQ(tmp[0], presize_val_Mixed);
  ASSERT_EQ(tmp[1], std::max(Buffer<CPUBackend>::padding(), 2 * sizeof(size_t)));

  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(3).mutable_data<size_t>(),
            sizeof(size_t) * 2, cudaMemcpyDefault));
  ASSERT_EQ(tmp[0], presize_val_GPU);
  ASSERT_EQ(tmp[1], 2 * sizeof(size_t));

  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(4).mutable_data<size_t>(),
            sizeof(size_t) * 2, cudaMemcpyDefault));
  ASSERT_EQ(tmp[0], presize_val_GPU);
  ASSERT_EQ(tmp[1], 2 * sizeof(size_t));

  CUDA_CALL(cudaMemcpy(&tmp, ws.Output<GPUBackend>(5).mutable_data<size_t>(),
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
      .AddArg("seed", seed_set)
      .AddInput("decoded", "gpu")
      .AddOutput("copied", "gpu"));

  vector<std::pair<string, string>> outputs = {{"copied", "gpu"}};

  pipe.Build(outputs);

  OpGraph &original_graph = this->GetGraph(&pipe);

  // Check if seed can be manually set to the reader
  ASSERT_EQ(original_graph.Node(3).spec.Arguments().at("seed")->Get<int64_t>(), seed_set);
  ASSERT_NE(original_graph.Node(0).spec.Arguments().at("seed")->Get<int64_t>(), seed_set);
}


class PrefetchedPipelineTest : public GenericDecoderTest<RGB> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_loadJPEGs + t_decodeJPEGs;
  }

  void SetUp() override {
    DALISingleOpTest::SetUp();
    batch_size_ = 5;
    // set_batch_size(jpegs_.nImages());
  }

  void CheckResults(Pipeline &pipe, int batch_size, int Iter) {
    DeviceWorkspace ws;
    pipe.Outputs(&ws);
    ASSERT_EQ(ws.NumOutput(), 1);
    ASSERT_EQ(ws.NumInput(), 0);
    ASSERT_TRUE(ws.OutputIsType<GPUBackend>(0));
    TensorList<GPUBackend> &res1 = ws.Output<GPUBackend>(0);
    for (int j = 0; j < batch_size; ++j) {
      this->VerifyDecode(
          res1.template tensor<uint8>(j),
          res1.tensor_shape(j)[0],
          res1.tensor_shape(j)[1], (Iter * batch_size + j));
    }
  }

  void VerifyDecode(const uint8 *img, int h, int w, int img_id) const {
    // Although MakeJPEGBatch() allows us to create arbitrary big data set,
    // by cyclically repeating the images, VerifyDecode does not, and we
    // must handle the wrap-around ourselves.
    const int total_images = jpegs_.nImages();

    // Load the image to host
    uint8 *host_img = new uint8[h*w*c_];
    CUDA_CALL(cudaMemcpy(host_img, img, h*w*c_, cudaMemcpyDefault));

#if DALI_DEBUG
    WriteHWCImage(host_img, h, w, c_, std::to_string(img_id) + "-img");
#endif
    GenericDecoderTest::VerifyDecode(host_img, h, w, jpegs_, img_id % total_images);
    delete [] host_img;
  }

  int batch_size_, num_threads_ = 1;
};

TEST_F(PrefetchedPipelineTest, SetQueueSizesSeparatedFail) {
  Pipeline pipe(this->batch_size_, 4, 0);
  // By default we are non-separated execution
  ASSERT_THROW(pipe.SetQueueSizes(5, 3), std::runtime_error);
}

TEST_F(PrefetchedPipelineTest, SetExecutionTypesFailAfterBuild) {
  Pipeline pipe(this->batch_size_, 4, 0);
  pipe.AddExternalInput("data");
  pipe.AddOperator(OpSpec("HostDecoder")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("images", "cpu"));

  pipe.AddOperator(OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("images", "gpu")
          .AddOutput("final_images", "gpu"));

  vector<std::pair<string, string>> outputs = {{"final_images", "gpu"}};
  pipe.Build(outputs);
  ASSERT_THROW(pipe.SetExecutionTypes(), std::runtime_error);
}

TEST_F(PrefetchedPipelineTest, SetQueueSizesFailAfterBuild) {
  Pipeline pipe(this->batch_size_, 4, 0);
  pipe.AddExternalInput("data");
  pipe.AddOperator(OpSpec("HostDecoder")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("images", "cpu"));

  pipe.AddOperator(OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("images", "gpu")
          .AddOutput("final_images", "gpu"));

  vector<std::pair<string, string>> outputs = {{"final_images", "gpu"}};
  pipe.Build(outputs);
  ASSERT_THROW(pipe.SetQueueSizes(2, 2), std::runtime_error);
}

TEST_F(PrefetchedPipelineTest, TestFillQueues) {
  // Test coprime queue sizes
  constexpr int CPU = 5, GPU = 3;
  constexpr int N = CPU + GPU + 5;
  // this->set_batch_size(this->batch_size_);
  int batch_size = this->batch_size_;
  this->SetEps(1.6);

  Pipeline pipe(batch_size, 4, 0);
  // Cannot test async while setting external input - need to make sure that
  pipe.SetExecutionTypes(true, true, true);
  // Test coprime queue sizes
  pipe.SetQueueSizes(CPU, GPU);
  pipe.AddExternalInput("data");

  pipe.AddOperator(OpSpec("HostDecoder")
          .AddArg("device", "cpu")
          .AddInput("data", "cpu")
          .AddOutput("images", "cpu"));

  pipe.AddOperator(OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("images", "gpu")
          .AddOutput("final_images", "gpu"));

  vector<std::pair<string, string>> outputs = {{"final_images", "gpu"}};
  pipe.Build(outputs);

  TensorList<CPUBackend> tl;
  this->MakeJPEGBatch(&tl, batch_size * N);

  // Split the batch into 5
  std::array<TensorList<CPUBackend>, N> splited_tl;
  std::array<std::vector<Dims>, N> shapes;
  for (int i = 0; i < N; i++) {
    shapes[i].resize(batch_size);
    for (int j = 0; j < batch_size; j++) {
      shapes[i][j] = tl.tensor_shape(i * batch_size + j);
    }
    splited_tl[i].Resize(shapes[i]);
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < batch_size; j++) {
      std::memcpy(
        splited_tl[i].template mutable_tensor<uint8>(j),
        tl.template tensor<uint8>(i * batch_size + j),
        volume(tl.tensor_shape(i * batch_size + j)));
    }
  }

  // Fill queues in the same way as Python - this would be the first pipe.run()
  for (int i = 0; i < GPU; i++) {
    pipe.SetExternalInput("data", splited_tl[i]);
    pipe.RunCPU();
    pipe.RunGPU();
  }
  // We run CPU stage additional `CPU`-times, to fill the output queue
  for (int i = GPU; i < GPU + CPU; i++) {
    pipe.SetExternalInput("data", splited_tl[i]);
    pipe.RunCPU();
  }

  // Now we interleave the calls to Outputs() and Run() for the rest of the batch
  int obtained_outputs = 0;
  for (int i = GPU + CPU; i < N; i++) {
    CheckResults(pipe, batch_size, obtained_outputs++);
    pipe.SetExternalInput("data", splited_tl[i]);
    pipe.RunCPU();
    pipe.RunGPU();
  }

  // We consumed all the data and have it in the Pipeline, now we need to run
  // Mixed and GPU stage to consume what was produced by the CPU
  for (int i = 0; i < CPU; i++) {
    CheckResults(pipe, batch_size, obtained_outputs++);
    pipe.RunGPU();
  }
  // Now we consule what we buffered in the beggining
  for (int i = 0; i < GPU; i++) {
    CheckResults(pipe, batch_size, obtained_outputs++);
  }
}


}  // namespace dali
