// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <utility>

#include "dali/test/dali_test_decoder.h"
#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/operator/builtin/external_source.h"

namespace dali {

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
        .AddOutput("data", "cpu")), "");

    graph_.AddOp(this->PrepareSpec(
            OpSpec("MakeContiguous")
            .AddArg("device", "mixed")
            .AddInput("data", "cpu")
            .AddOutput("final_images", "gpu")), "");
  }

  void BuildGPUGraph() {
    graph_.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddArg("device_id", 0)
          .AddOutput("final_images", "gpu")), "");
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
  void FeedWithCpuVector(ExternalSource<Backend> *src_op) {
    int dims = this->RandInt(1, 4);
    for (int j = 0; j < this->batch_size_; ++j) {
      auto &tensor = vt_cpu_[j];
      tensor.set_type(TypeInfo::Create<int>());
      auto shape = GetRandShape(dims);
      tensor.Resize(shape);
      auto data = tensor.template mutable_data<int>();
      for (int i = 0; i < tensor.size(); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
    }
    src_op->SetDataSource(vt_cpu_);
  }

  template<typename Backend>
  void FeedWithGpuVector(ExternalSource<Backend> *src_op) {
    int dims = this->RandInt(1, 4);
    for (int j = 0; j < this->batch_size_; ++j) {
      Tensor<CPUBackend> tensor;
      tensor.set_type(TypeInfo::Create<int>());
      auto shape = GetRandShape(dims);
      tensor.Resize(shape);
      auto data = tensor.template mutable_data<int>();
      for (int i = 0; i < tensor.size(); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
      vt_gpu_[j].Copy(tensor, 0);
    }
    cudaStreamSynchronize(0);
    src_op->SetDataSource(vt_gpu_);
  }

  template<typename Backend>
  void FeedWithCpuList(ExternalSource<Backend> *src_op) {
    tl_cpu_.set_type(TypeInfo::Create<int>());
    auto rand_shape = GetRandShape(this->RandInt(1, 4));
    TensorListShape<> shape = uniform_list_shape(this->batch_size_, rand_shape);
    tl_cpu_.Resize(shape);
    for (int j = 0; j < this->batch_size_; ++j) {
      auto data = tl_cpu_.template mutable_tensor<int>(j);
      for (int i = 0; i < volume(tl_cpu_.tensor_shape(j)); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
    }
    src_op->SetDataSource(tl_cpu_);
  }

  template<typename Backend>
  void FeedWithGpuList(ExternalSource<Backend> *src_op) {
    TensorList<CPUBackend> tensor_list;
    tensor_list.set_type(TypeInfo::Create<int>());
    auto rand_shape = GetRandShape(this->RandInt(1, 4));
    TensorListShape<> shape = uniform_list_shape(this->batch_size_, rand_shape);
    tensor_list.Resize(shape);
    for (int j = 0; j < this->batch_size_; ++j) {
      auto data = tensor_list.template mutable_tensor<int>(j);
      for (int i = 0; i < volume(tensor_list.tensor_shape(j)); ++i) {
        data[i] = fill_counter_;
        ++fill_counter_;
      }
      tl_gpu_.Copy(tensor_list, 0);
    }
    cudaStreamSynchronize(0);
    src_op->SetDataSource(tl_gpu_);
  }

  void RunExe() {
    exe_->RunCPU();
    exe_->RunMixed();
    exe_->RunGPU();
  }

  bool RunOutputs() {
    DeviceWorkspace ws;
    exe_->Outputs(&ws);
    auto &tensor_gpu_list = ws.Output<GPUBackend>(0);
    TensorList<CPUBackend> tensor_cpu_list;
    tensor_cpu_list.Copy(tensor_gpu_list, (ws.has_stream() ? ws.stream() : 0));
    cudaStreamSynchronize(ws.has_stream() ? ws.stream() : 0);

    for (int j = 0; j < this->batch_size_; ++j) {
      auto data = tensor_cpu_list.template mutable_tensor<int>(j);
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
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op);
  }
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, Interleaved) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, InterleavedVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuVector(src_op);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, InterleavedGPU) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuVector(src_op);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPUVector) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithCpuVector(src_op);
  }

  this->RunExe();
  EXPECT_TRUE(this->RunOutputs());
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2GPU) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuList(src_op);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2GPUVector) {
  auto *src_op = this->CreateGPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuVector(src_op);
  }

  this->RunExe();
  EXPECT_TRUE(this->RunOutputs());
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2CPU) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuList(src_op);
  }

  for (int i = 0; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeGPU2CPUVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithGpuVector(src_op);
  }

  this->RunExe();
  EXPECT_TRUE(this->RunOutputs());
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeMixed) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    if (i % 2 == 0) {
      this->FeedWithCpuVector(src_op);
    } else {
      this->FeedWithCpuList(src_op);
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
  this->FeedWithCpuList(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op);
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
  this->FeedWithCpuVector(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
      this->FeedWithCpuVector(src_op);
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
  this->FeedWithCpuVector(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    if (i % 2 == 0) {
      this->FeedWithCpuVector(src_op);
    } else {
      this->FeedWithCpuList(src_op);
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
  this->FeedWithCpuList(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->FeedWithCpuList(src_op);
  }

  EXPECT_TRUE(this->RunOutputs());
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TEST(ExternalSourceTestNoInput, Throw) {
  OpGraph graph;
  int batch_size = 1;
  int num_threads = 1;

  auto exe = std::make_unique<SimpleExecutor>(batch_size, num_threads, 0, 1);
  exe->Init();

  graph.AddOp(
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddArg("device_id", 0)
      .AddOutput("data_out", "cpu")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", num_threads), "");

  vector<string> outputs = {"data_out_cpu"};

  exe->Build(&graph, outputs);
  exe->RunCPU();
  exe->RunMixed();
  exe->RunGPU();
  DeviceWorkspace ws;
  ASSERT_THROW(exe->Outputs(&ws), std::runtime_error);
}

}  // namespace dali
