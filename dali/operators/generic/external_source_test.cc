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
    vt_.resize(this->batch_size_);
    fill_counter_ = 0;
    check_counter_ = 0;
  }

  inline void set_batch_size(int size) { batch_size_ = size; }

  inline OpSpec& PrepareSpec(OpSpec &spec) const {
    spec.AddArg("batch_size", batch_size_)
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

  template<typename T>
  void FeedWithVector(T *src_op) {
    for (int j = 0; j < this->batch_size_; ++j) {
      auto &tensor = vt_[j];
      tensor.set_type(TypeInfo::Create<int>());
      tensor.Resize({10, 10});
      auto data = tensor.template mutable_data<int>();
      for (int i = 0; i < tensor.size(); ++i) {
        data[i] = fill_counter_;
      }
       ++fill_counter_;
    }
    src_op->SetDataSource(vt_);
  }

  template<typename T>
  void FeedWithList(T *src_op) {
    tl_.set_type(TypeInfo::Create<int>());
    TensorListShape<> shape = uniform_list_shape(this->batch_size_, {10, 10});
    tl_.Resize(shape);
    for (int j = 0; j < this->batch_size_; ++j) {
      auto data = tl_.template mutable_tensor<int>(j);
      for (int i = 0; i < volume(tl_.tensor_shape(j)); ++i) {
        data[i] = fill_counter_;
      }
      ++fill_counter_;
    }
    src_op->SetDataSource(tl_);
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
      }
      ++check_counter_;
    }
    return true;
  }

  int batch_size_, num_threads_ = 1;
  std::unique_ptr<AsyncPipelinedExecutor> exe_;
  OpGraph graph_;
  TensorList<CPUBackend> tl_;
  std::vector<Tensor<CPUBackend>> vt_;
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
    this->FeedWithList(src_op);
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
    this->FeedWithList(src_op);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, InterleavedVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithVector(src_op);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, InterleavedGPU) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithList(src_op);
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeVector) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    this->FeedWithVector(src_op);
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
    this->FeedWithList(src_op);
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
    this->FeedWithVector(src_op);
  }

  this->RunExe();
  EXPECT_ANY_THROW(this->RunOutputs());
}

TYPED_TEST(ExternalSourceTest, FeedThenConsumeMixed) {
  auto *src_op = this->CreateCPUExe();
  ASSERT_NE(src_op, nullptr);
  for (int i = 0; i < TypeParam::loops; ++i) {
    if (i % 2 == 0) {
      this->FeedWithVector(src_op);
    } else {
      this->FeedWithList(src_op);
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
  this->FeedWithList(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->FeedWithList(src_op);
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
  this->FeedWithVector(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
      this->FeedWithVector(src_op);
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
  this->FeedWithVector(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    if (i % 2 == 0) {
      this->FeedWithVector(src_op);
    } else {
      this->FeedWithList(src_op);
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
  this->FeedWithList(src_op);

  this->RunExe();
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->FeedWithList(src_op);
  }

  EXPECT_TRUE(this->RunOutputs());
  for (int i = 1; i < TypeParam::loops; ++i) {
    this->RunExe();
    EXPECT_TRUE(this->RunOutputs());
  }
}

}  // namespace dali
