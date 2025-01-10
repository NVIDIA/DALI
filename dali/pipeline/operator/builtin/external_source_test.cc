// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <memory>
#include <tuple>
#include <utility>

#include "dali/test/dali_test_decoder.h"
#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/operator/builtin/external_source.h"
#include "dali/test/dali_test_config.h"
#include "dali/c_api.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
namespace dali {


template<typename Backend>
class ExternalSourceBasicTest : public ::testing::Test {
 protected:
  static constexpr bool is_cpu = std::is_same_v<Backend, CPUBackend>;


  void CreateAndSerializePipeline(bool repeat_last) {
    std::string dev_str = is_cpu ? "cpu" : "gpu";
    auto pipeline = std::make_unique<Pipeline>(batch_size_, num_threads_, device_id_, false, 1,
                                               false);
    pipeline->AddOperator(
            OpSpec("ExternalSource")
                    .AddArg("repeat_last", repeat_last)
                    .AddArg("device", dev_str)
                    .AddArg("name", input_name_)
                    .AddOutput(input_name_, is_cpu ? StorageDevice::CPU : StorageDevice::GPU),
            input_name_);

    std::vector<std::pair<std::string, std::string>> outputs = {
            {input_name_, dev_str},
    };

    pipeline->SetOutputDescs(outputs);

    serialized_pipeline_ = pipeline->SerializeToProtobuf();
  }


  void FeedExternalInput(daliPipelineHandle *h) {
    daliSetExternalInputBatchSize(h, input_name_.c_str(), batch_size_);
    int bytes_per_sample = 10;
    std::vector<int64_t> shapes(batch_size_, bytes_per_sample);
    kernels::TestTensorList<uint8_t> data;
    data.reshape(uniform_list_shape(batch_size_, {bytes_per_sample}));
    assert(bytes_per_sample * batch_size_ < 255);
    SequentialFill(data.cpu(), 1);
    if constexpr (is_cpu) {
      daliSetExternalInput(h, input_name_.c_str(), device_type_t::CPU, data.cpu().tensor_data(0),
                           dali_data_type_t::DALI_UINT8, shapes.data(), 1, nullptr,
                           DALI_ext_force_copy);
    } else {
      daliSetExternalInput(h, input_name_.c_str(), device_type_t::GPU,
                           data.gpu().tensor_data(0),
                           dali_data_type_t::DALI_UINT8, shapes.data(), 1, nullptr,
                           DALI_ext_force_copy);
    }
  }


  /**
   * Check, if the "depleted" trace exists.
   */
  void AssertDepletedTraceExist(daliPipelineHandle *h) {
    // The "depleted" trace should always exist.
    ASSERT_TRUE(daliHasOperatorTrace(h, input_name_.c_str(), depleted_trace_name_.c_str()));
  }


  /**
   * Verify the value of "depleted" trace.
   * @param shall_be_depleted Expected value.
   */
  void CheckDepletedTraceValue(daliPipelineHandle *h, bool shall_be_depleted) {
    EXPECT_STREQ(daliGetOperatorTrace(h, input_name_.c_str(), depleted_trace_name_.c_str()),
                 shall_be_depleted ? "true" : "false");
  }


  const int batch_size_ = 3;
  const int num_threads_ = 2;
  const int device_id_ = 0;
  daliPipelineHandle handle_;
  std::string serialized_pipeline_;
  const std::string depleted_trace_name_ = "depleted";
  const std::string input_name_ = "INPUT";
};

using ExternalSourceTestTypes = ::testing::Types<CPUBackend, GPUBackend>;
TYPED_TEST_SUITE(ExternalSourceBasicTest, ExternalSourceTestTypes);


TYPED_TEST(ExternalSourceBasicTest, DepletedTraceTest) {
  this->CreateAndSerializePipeline(false);
  daliDeserializeDefault(&this->handle_, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));

  this->FeedExternalInput(&this->handle_);

  // At this point, "depleted" trace does not yet exist. It needs a Workspace.

  daliRun(&this->handle_);
  daliShareOutput(&this->handle_);
  this->AssertDepletedTraceExist(&this->handle_);
  this->CheckDepletedTraceValue(&this->handle_, true);
  daliOutputRelease(&this->handle_);

  this->FeedExternalInput(&this->handle_);
  this->FeedExternalInput(&this->handle_);

  daliRun(&this->handle_);
  daliShareOutput(&this->handle_);
  this->AssertDepletedTraceExist(&this->handle_);
  this->CheckDepletedTraceValue(&this->handle_, false);
  daliOutputRelease(&this->handle_);

  daliRun(&this->handle_);
  daliShareOutput(&this->handle_);
  this->AssertDepletedTraceExist(&this->handle_);
  this->CheckDepletedTraceValue(&this->handle_, true);
  daliOutputRelease(&this->handle_);

  daliDeletePipeline(&this->handle_);
}


TYPED_TEST(ExternalSourceBasicTest, DepletedTraceRepeatLastTest) {
  this->CreateAndSerializePipeline(true);
  daliDeserializeDefault(&this->handle_, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));

  this->FeedExternalInput(&this->handle_);

  // At this point, "depleted" trace does not yet exist. It needs a Workspace.

  daliRun(&this->handle_);
  daliShareOutput(&this->handle_);
  this->AssertDepletedTraceExist(&this->handle_);
  this->CheckDepletedTraceValue(&this->handle_, false);
  daliOutputRelease(&this->handle_);

  daliRun(&this->handle_);
  daliShareOutput(&this->handle_);
  this->AssertDepletedTraceExist(&this->handle_);
  this->CheckDepletedTraceValue(&this->handle_, false);
  daliOutputRelease(&this->handle_);

  this->FeedExternalInput(&this->handle_);
  this->FeedExternalInput(&this->handle_);

  daliRun(&this->handle_);
  daliShareOutput(&this->handle_);
  this->AssertDepletedTraceExist(&this->handle_);
  this->CheckDepletedTraceValue(&this->handle_, false);
  daliOutputRelease(&this->handle_);

  daliRun(&this->handle_);
  daliShareOutput(&this->handle_);
  this->AssertDepletedTraceExist(&this->handle_);
  this->CheckDepletedTraceValue(&this->handle_, false);
  daliOutputRelease(&this->handle_);

  daliDeletePipeline(&this->handle_);
}


template <typename LoopsCount>
class ExternalSourceTest : public::testing::WithParamInterface<int>,
                           public GenericDecoderTest<RGB> {
 protected:
  template <typename... T>
  std::unique_ptr<AsyncPipelinedExecutor> GetExecutor(T&&... args) {
    return std::make_unique<AsyncPipelinedExecutor>(std::forward<T>(args)...);
  }

  uint32_t GetImageLoadingFlags() const override {
    return t_loadJPEGs + t_decodeJPEGs;
  }

  void SetUp() override {
    DALISingleOpTest::SetUp();
    set_batch_size(10);
    vt_cpu_.resize(this->batch_size_);
    for (auto &vt : vt_cpu_) {
      vt.Reset();
      vt.set_pinned(false);
    }
    vt_gpu_.resize(this->batch_size_);
    for (auto &vt : vt_gpu_) {
      vt.Reset();
      vt.set_pinned(false);
    }
    tl_cpu_.Reset();
    tl_cpu_.set_pinned(false);
    tl_gpu_.Reset();
    fill_counter_ = 0;
    check_counter_ = 0;
  }

  inline void set_batch_size(int size) { batch_size_ = size; }

  inline OpSpec& PrepareSpec(OpSpec &spec) const {
    spec.AddArg("max_batch_size", batch_size_)
      .AddArg("num_threads", num_threads_);
    return spec;
  }

  void BuildCPUGraph() {
    graph_.AddOp(this->PrepareSpec(
        OpSpec("ExternalSource")
        .AddArg("device", "cpu")
        .AddArg("device_id", 0)
        .AddOutput("data", StorageDevice::CPU)), "");

    graph_.AddOp(this->PrepareSpec(
            OpSpec("MakeContiguous")
            .AddArg("device", "mixed")
            .AddInput("data", StorageDevice::CPU)
            .AddOutput("final_images", StorageDevice::GPU)), "");
  }

  void BuildGPUGraph() {
    graph_.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddArg("device_id", 0)
          .AddOutput("data", StorageDevice::GPU)), "");
    graph_.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "gpu")
          .AddInput("data", StorageDevice::GPU)
          .AddOutput("final_images", StorageDevice::GPU)), "");
  }

  ExternalSource<CPUBackend>* CreateCPUExe() {
    exe_ = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe_->Init();
    BuildCPUGraph();

    vector<string> outputs = {"final_images_gpu"};
    exe_->Build(&graph_, outputs);
    return dynamic_cast<ExternalSource<CPUBackend> *>(graph_.Node(OpType::CPU, 0).op.get());
  }

  ExternalSource<GPUBackend>* CreateGPUExe() {
    exe_ = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe_->Init();
    this->BuildGPUGraph();

    vector<string> outputs = {"final_images_gpu"};
    exe_->Build(&graph_, outputs);
    return dynamic_cast<ExternalSource<GPUBackend> *>(this->graph_.Node(OpType::GPU, 0).op.get());
  }

  TensorShape<> GetRandShape(int dims) {
    TensorShape<> shape;
    shape.resize(dims);
    for (auto &val : shape) {
      val = this->RandInt(1, 15);
    }
    return shape;
  }

  template<typename Backend>
  void FeedWithCpuVector(ExternalSource<Backend> *src_op, int dims) {
    for (int j = 0; j < this->batch_size_; ++j) {
      auto &tensor = vt_cpu_[j];
      auto shape = GetRandShape(dims);
      tensor.Resize(shape, DALI_INT32);
      auto data = tensor.template mutable_data<int>();
      for (int i = 0; i < tensor.size(); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
    }
    src_op->SetDataSource(vt_cpu_);
  }

  template<typename Backend>
  void FeedWithGpuVector(ExternalSource<Backend> *src_op, int dims) {
    AccessOrder order(cudaStream_t(0));
    for (int j = 0; j < this->batch_size_; ++j) {
      Tensor<CPUBackend> tensor;
      tensor.set_pinned(false);
      auto shape = GetRandShape(dims);
      tensor.Resize(shape, DALI_INT32);
      auto data = tensor.template mutable_data<int>();
      for (int i = 0; i < tensor.size(); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
      vt_gpu_[j].set_order(order);
      vt_gpu_[j].Copy(tensor);
    }
    src_op->SetDataSource(vt_gpu_, order);
    AccessOrder::host().wait(order);
  }

  template<typename Backend>
  void FeedWithCpuList(ExternalSource<Backend> *src_op, int dims) {
    auto rand_shape = GetRandShape(dims);
    TensorListShape<> shape = uniform_list_shape(this->batch_size_, rand_shape);
    tl_cpu_.Resize(shape, DALI_INT32);
    for (int j = 0; j < this->batch_size_; ++j) {
      auto data = tl_cpu_.template mutable_tensor<int>(j);
      for (int i = 0; i < volume(tl_cpu_.tensor_shape(j)); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
    }
    src_op->SetDataSource(tl_cpu_, {});
  }

  template<typename Backend>
  void FeedWithGpuList(ExternalSource<Backend> *src_op, int dims) {
    TensorList<CPUBackend> tensor_list;
    auto rand_shape = GetRandShape(dims);
    TensorListShape<> shape = uniform_list_shape(this->batch_size_, rand_shape);
    tensor_list.Resize(shape, DALI_INT32);
    for (int j = 0; j < this->batch_size_; ++j) {
      auto data = tensor_list.template mutable_tensor<int>(j);
      for (int i = 0; i < volume(tensor_list.tensor_shape(j)); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
    }
    AccessOrder order(cudaStream_t(0));
    tl_gpu_.set_order(order);
    tl_gpu_.Copy(tensor_list);
    src_op->SetDataSource(tl_gpu_, order);
    AccessOrder::host().wait(order);
  }

  void RunExe() {
    exe_->Run();
  }

  bool RunOutputs() {
    Workspace ws;
    exe_->Outputs(&ws);
    auto &tensor_gpu_list = ws.Output<GPUBackend>(0);
    TensorList<CPUBackend> tensor_cpu_list;
    AccessOrder order = ws.has_stream() ? AccessOrder(ws.stream()) : AccessOrder::host();
    tensor_cpu_list.Copy(tensor_gpu_list, order);
    CUDA_CALL(cudaStreamSynchronize(ws.has_stream() ? ws.stream() : 0));

    for (int j = 0; j < this->batch_size_; ++j) {
      const auto *data = tensor_cpu_list.template tensor<int>(j);
      for (int i = 0; i < volume(tensor_cpu_list.tensor_shape(j)); ++i) {
        if (data[i] != check_counter_) {
          return false;
        }
        ++check_counter_;
      }
    }
    return true;
  }

  int batch_size_, num_threads_ = 1;
  std::unique_ptr<AsyncPipelinedExecutor> exe_;
  OpGraph graph_;
  TensorList<CPUBackend> tl_cpu_;
  std::vector<Tensor<CPUBackend>> vt_cpu_;
  TensorList<GPUBackend> tl_gpu_;
  std::vector<Tensor<GPUBackend>> vt_gpu_;
  int fill_counter_;
  int check_counter_;
};

template <int number_of_loops>
struct FeedCount {
  static const int loops = number_of_loops;
};

typedef ::testing::Types<FeedCount<1>,
                         FeedCount<2>,
                         FeedCount<4>,
                         FeedCount<10>> NumLoops;
TYPED_TEST_SUITE(ExternalSourceTest, NumLoops);

TYPED_TEST(ExternalSourceTest, FeedThenConsume) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op, dims);
  }
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, Interleaved) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op, dims);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, InterleavedVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuVector(src_op, dims);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, InterleavedGPU) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op, dims);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuVector(src_op, dims);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op, dims);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPUVector) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuVector(src_op, dims);
  }

  this->RunExe();
  EXPECT_TRUE(this->RunOutputs());
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2GPU) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuList(src_op, dims);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2GPUVector) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuVector(src_op, dims);
  }

  this->RunExe();
  EXPECT_TRUE(this->RunOutputs());
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2CPU) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuList(src_op, dims);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2CPUVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuVector(src_op, dims);
  }

  this->RunExe();
  EXPECT_TRUE(this->RunOutputs());
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeMixed) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  for (int i = 0; i < TypeParam::loops; ++i) {
    if (i % 2 == 0) {
      this->FeedWithCpuVector(src_op, dims);
    } else {
      this->FeedWithCpuList(src_op, dims);
    }
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, ConsumeOneThenFeeds) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  this->FeedWithCpuList(src_op, dims);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op, dims);
  }

  EXPECT_TRUE(this->RunOutputs());
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, ConsumeOneThenFeedsVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  this->FeedWithCpuVector(src_op, dims);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
      this->FeedWithCpuVector(src_op, dims);
  }

  EXPECT_TRUE(this->RunOutputs());
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, ConsumeOneThenFeedsMixed) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  this->FeedWithCpuVector(src_op, dims);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    if (i % 2 == 0) {
      this->FeedWithCpuVector(src_op, dims);
    } else {
      this->FeedWithCpuList(src_op, dims);
    }
  }

  EXPECT_TRUE(this->RunOutputs());
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, ConsumeOneThenFeedsGPU) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  int dims = this->RandInt(1, 4);
  this->FeedWithCpuList(src_op, dims);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op, dims);
  }

  EXPECT_TRUE(this->RunOutputs());
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TEST(ExternalSourceTestNoInput, ThrowCpu) {
  OpGraph graph;
  int batch_size = 1;
  int num_threads = 1;

  auto exe = std::make_unique<SimpleExecutor>(batch_size, num_threads, 0, 1);
  exe->Init();

  graph.AddOp(
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddArg("device_id", 0)
      .AddOutput("data_out", StorageDevice::CPU)
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", num_threads), "");

  vector<string> outputs = {"data_out_cpu"};

  exe->Build(&graph, outputs);
  exe->Run();
  Workspace ws;
  EXPECT_THROW(exe->ShareOutputs(&ws), std::exception);
}


void TestOnlyExternalSource(Pipeline &pipe, const std::string &name, const std::string &dev) {
  // Check if the external source is the only operator deserialized
  auto *op = pipe.GetOperatorNode(name);
  ASSERT_TRUE(op->inputs.empty());
  ASSERT_EQ(op->spec.SchemaName(), "ExternalSource");
  ASSERT_EQ(pipe.num_outputs(), 1);
  ASSERT_EQ(pipe.output_device(0), ParseStorageDevice(dev));
  ASSERT_EQ(pipe.output_name(0), name);
  // Make Contiguous is always added at the end
  ASSERT_EQ(op->outputs.size(), 1_uz);
  ASSERT_EQ(op->outputs[0]->consumers.size(), 1_uz);
}


void TestRunExternalSource(Pipeline &pipe, const std::string &name,
                                    const std::string &dev) {
  TensorListShape<> input_shape =  uniform_list_shape(10, {42, 42, 3});
  TensorList<CPUBackend> input_cpu;
  input_cpu.Resize(input_shape, DALI_UINT8);
  int64_t counter = 0;
  for (int sample_idx = 0; sample_idx < input_shape.num_samples(); sample_idx++) {
    for (int64_t i = 0; i < input_shape[sample_idx].num_elements(); i++, counter++) {
      input_cpu.mutable_tensor<uint8_t>(sample_idx)[i] = counter % 255;
    }
  }
  Workspace ws;
  if (dev == "cpu") {
    // take Make Contiguous into account
    pipe.SetExternalInput("es", input_cpu);
  } else {
    TensorList<GPUBackend> input_gpu;
    input_gpu.Copy(input_cpu);
    cudaStreamSynchronize(0);
    pipe.SetExternalInput("es", input_gpu);
  }
  pipe.Run();

  TensorList<CPUBackend> output_cpu;
  pipe.Outputs(&ws);
  if (dev == "cpu") {
    output_cpu.Copy(ws.Output<CPUBackend>(0), AccessOrder::host());
  } else {
    auto &output = ws.Output<GPUBackend>(0);
    output_cpu.Copy(output, output.order());
    CUDA_CALL(cudaStreamSynchronize(output.order().stream()));
  }
  ASSERT_EQ(input_cpu.shape(), output_cpu.shape());
  ASSERT_EQ(input_cpu.type(), output_cpu.type());
  for (int sample_idx = 0; sample_idx < input_shape.num_samples(); sample_idx++) {
    ASSERT_EQ(memcmp(input_cpu.tensor<uint8_t>(sample_idx), output_cpu.tensor<uint8_t>(sample_idx),
                     input_shape[sample_idx].num_elements()),
              0);
  }
}


TEST(ExternalSourceTest, SerializeDeserializeOpSpec) {
  std::string name = "es";
  for (std::string dev : {"cpu", "gpu"}) {
    Pipeline pipe_to_serialize(10, 4, 0);
    pipe_to_serialize.AddOperator(OpSpec("ExternalSource")
                      .AddArg("device", dev)
                      .AddArg("name", name)
                      .AddOutput("es", ParseStorageDevice(dev)),
                  name);
    pipe_to_serialize.Build({{name, dev}});
    auto serialized = pipe_to_serialize.SerializeToProtobuf();

    Pipeline pipe(serialized);
    pipe.Build();

    TestOnlyExternalSource(pipe, name, dev);
    TestRunExternalSource(pipe, name, dev);
  }
}


TEST(ExternalSourceTest, SerializeDeserializeAddExternalInput) {
  std::string name = "es";
  for (std::string dev : {"cpu", "gpu"}) {
    Pipeline pipe_to_serialize(10, 4, 0);
    pipe_to_serialize.AddExternalInput(name, dev);
    pipe_to_serialize.Build({{name, dev}});
    auto serialized = pipe_to_serialize.SerializeToProtobuf();

    Pipeline pipe(serialized);
    pipe.Build();

    TestOnlyExternalSource(pipe, name, dev);
    TestRunExternalSource(pipe, name, dev);
  }
}


// Data for `DeserializeLegacyExternalSource` was generated with DALI 1.0.0 using the code below:
// TEST(ExternalSourceGen, GeneratePipelines) {
//   {
//     Pipeline p(10, 4, 0);
//     p.AddExternalInput("es");
//     p.Build({{"es", "cpu"}});
//     std::ofstream file("add_external_input_v1.0.0.dali",
//                        std::ios::out | std::ios::binary);
//     file << p.SerializeToProtobuf();
//   }
//   for (auto dev : {"cpu"s, "gpu"s}) {
//     Pipeline p(10, 4, 0);
//     p.AddOperator(OpSpec("_ExternalSource")
//                       .AddArg("device", dev)
//                       .AddArg("name", "es")
//                       .AddOutput("es", dev),
//                   "es");
//     p.Build({{"es", dev}});
//     std::ofstream file("underscore_ext_src_" + dev + "_v1.0.0.dali",
//                        std::ios::out | std::ios::binary);
//     file << p.SerializeToProtobuf();
//   }
// }
//
// and:
//
// for dev in ["cpu", "gpu"]:
//     @pipeline_def
//     def my_pipeline():
//         es = fn.external_source(name="es", device=dev)
//         return es
//
//     pipe = my_pipeline(batch_size=10, num_threads=2, device_id=0)
//     pipe.build()
//     pipe.serialize(filename="python_"+dev+"_v1.0.0.dali")
TEST(ExternalSourceTest, DeserializeLegacyExternalSource) {
  std::string name = "es";
  std::string path = testing::dali_extra_path() + "/db/serialized_pipes/";
  std::tuple<std::string, std::string> es_pipes[] = {
      {"add_external_input_v1.0.0.dali", "cpu"},
      {"underscore_ext_src_cpu_v1.0.0.dali", "cpu"},
      {"underscore_ext_src_gpu_v1.0.0.dali", "gpu"},
      {"python_cpu_v1.0.0.dali", "cpu"},
      {"python_gpu_v1.0.0.dali", "gpu"}};
  for (const auto &file_dev : es_pipes) {
    std::string path_to_deserialize = path + std::get<0>(file_dev);
    std::string dev = std::get<1>(file_dev);
    std::fstream file(path_to_deserialize, std::ios::in | std::ios::binary);

    // Shortest (and not the slowest) way to read whole file with C++ API
    std::stringstream tmp_ss;
    tmp_ss << file.rdbuf();
    auto pipe_str = tmp_ss.str();

    Pipeline pipe(pipe_str);
    pipe.Build();

    TestOnlyExternalSource(pipe, name, dev);
    TestRunExternalSource(pipe, name, dev);
  }
}


}  // namespace dali
