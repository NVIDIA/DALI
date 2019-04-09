// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/test/dali_test.h"
#include "dali/pipeline/operators/reader/reader_op.h"

namespace dali {

class DummyLoader : public Loader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit DummyLoader(const OpSpec& spec) :
    Loader<CPUBackend, Tensor<CPUBackend>>(spec) {}

  void ReadSample(Tensor<CPUBackend> &t) override {
    t.Resize({1});
    t.set_type(TypeInfo::Create<uint8_t>());
  }

  Index SizeImpl() override {
    return 1;
  }

  void Reset(bool wrap_to_shard) override {}
};
class DummyDataReader : public DataReader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit DummyDataReader(const OpSpec &spec)
      : DataReader<CPUBackend, Tensor<CPUBackend>>(spec),
        count_(0) {
    loader_.reset(new DummyLoader(spec));
  }

  ~DummyDataReader() override {
    DataReader<CPUBackend, Tensor<CPUBackend>>::StopPrefetchThread();
  }
  /*
  using DataReader<CPUBackend>::prefetched_batch_;
  bool Prefetch() override {
    for (int i = 0; i < Operator::batch_size_; ++i) {
      printf("new tensor %d\n", i);
      auto *t = loader_->ReadOne();
      prefetched_batch_.push_back(t);
    }
    return true;
  }
  */

  void RunImpl(SampleWorkspace* ws, int idx) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    ws->Output<CPUBackend>(0).Copy(GetSample(ws->data_idx()), 0);
  }

 private:
  std::atomic<int> count_;
};

DALI_REGISTER_OPERATOR(DummyDataReader, DummyDataReader, CPU);

DALI_SCHEMA(DummyDataReader)
  .DocStr("Dummy")
  .OutputFn([](const OpSpec& spec) { return 1; })
  .NumInput(0)
  .NumOutput(1)
  .AddParent("LoaderBase");

template <typename Backend>
class ReaderTest : public DALITest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_SUITE(ReaderTest, TestTypes);

TYPED_TEST(ReaderTest, SimpleTest) {
  Pipeline pipe(128, 1, 0);

  pipe.AddOperator(
      OpSpec("DummyDataReader")
      .AddOutput("data_out", "cpu"));

  std::vector<std::pair<string, string>> outputs = {{"data_out", "cpu"}};
  pipe.Build(outputs);

  DeviceWorkspace ws;
  for (int i=0; i < 5; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  }

  return;
}

TYPED_TEST(ReaderTest, PrefetchQueueTest) {
  Pipeline pipe(128, 1, 0);

  pipe.AddOperator(
      OpSpec("DummyDataReader")
      .AddOutput("data_out", "cpu")
      .AddArg("prefetch_queue_depth", 3));

  std::vector<std::pair<string, string>> outputs = {{"data_out", "cpu"}};
  pipe.Build(outputs);

  DeviceWorkspace ws;
  for (int i=0; i < 5; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  }
  return;
}


TYPED_TEST(ReaderTest, SequenceTest) {
  Pipeline pipe(128, 4, 0);

  pipe.AddOperator(
      OpSpec("SequenceReader")
      .AddArg("file_root", image_folder + "/frames/")
      .AddArg("sequence_length", 3)
      .AddArg("image_type", DALI_RGB)
      .AddOutput("seq_out", "cpu"));

  std::vector<std::pair<string, string>> outputs = {{"seq_out", "cpu"}};
  pipe.Build(outputs);

  DeviceWorkspace ws;
  for (int i = 0; i < 4; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    auto shape = ws.Output<CPUBackend>(0).AsTensor()->shape();
    // We have NFHWC format
    const auto batch_size = shape[0];
    const auto frame_count = shape[1];
    const auto H = shape[2];
    const auto W = shape[3];
    const auto C = shape[4];
    const auto frame_size = H * W * C;
    const auto seq_size = frame_size * frame_count;
    for (int sample = 0; sample < batch_size; sample++) {
      // We read samples sequentially. We have 2 "videos" of 16 frames,
      // as sequence do not cross the boundary of one video, the highest starting frame is
      // 13 (counting from 0, the last 3-element sequence is [13, 14, 15]).
      auto start_frame = (i * batch_size + sample) % (16 - 3 + 1);
      for (int frame = 0; frame < frame_count; frame++) {
        auto off = sample * seq_size + frame * frame_size;
        auto val = ws.Output<CPUBackend>(0).AsTensor()->data<uint8_t>()[off];
        decltype(val) expected = start_frame + frame;
        ASSERT_EQ(val, expected);
      }
    }
  }

  return;
}

class TestLoader : public Loader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit TestLoader(const OpSpec& spec) :
    Loader<CPUBackend, Tensor<CPUBackend>>(spec), current_index_(0) {}

  void ReadSample(Tensor<CPUBackend> &t) override {}

  Index SizeImpl() override {
    return 10;
  }

  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, Size());
    } else {
      current_index_ = 0;
    }
  }

  bool IsNextShard(Index current_index) override {
    return Loader::IsNextShard(current_index);
  }

  void MoveToNextShard(Index current_index) override {
    Loader::MoveToNextShard(current_index);
  }

  Index current_index_;
};

TYPED_TEST(ReaderTest, ResetLoaderTestWrap) {
  TestLoader tl(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 0)
      .AddArg("num_shards", 2)
      .AddArg("stick_to_shard", false)
      .AddArg("batch_size", 2)
      .AddArg("device_id", 0));
  tl.PrepareMetadata();

  ASSERT_EQ(tl.IsNextShard(0)            , false);
  ASSERT_EQ(tl.IsNextShard(tl.Size() / 2), false);
  ASSERT_EQ(tl.IsNextShard(tl.Size() - 1), false);
  ASSERT_EQ(tl.IsNextShard(tl.Size())    , true);
  ASSERT_EQ(tl.IsNextShard(tl.Size() + 1), true);
  tl.current_index_ = 1;
  tl.Reset(true);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = 0;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = tl.Size() / 2;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, tl.Size() / 2);

  tl.current_index_ = tl.Size() - 1;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, tl.Size() - 1);

  tl.current_index_ = tl.Size();
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = tl.Size() + 1;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);
}

TYPED_TEST(ReaderTest, ResetLoaderTestStickToShard) {
  TestLoader tl(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 0)
      .AddArg("num_shards", 2)
      .AddArg("stick_to_shard", true)
      .AddArg("batch_size", 2)
      .AddArg("device_id", 0));
  tl.PrepareMetadata();

  ASSERT_EQ(tl.IsNextShard(0)            , false);
  ASSERT_EQ(tl.IsNextShard(tl.Size() / 2), true);
  ASSERT_EQ(tl.IsNextShard(tl.Size() - 1), true);
  ASSERT_EQ(tl.IsNextShard(tl.Size())    , true);
  ASSERT_EQ(tl.IsNextShard(tl.Size() + 1), true);
  tl.current_index_ = 1;
  tl.Reset(true);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = 0;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = tl.Size() / 2;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = tl.Size() - 1;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = tl.Size();
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);

  tl.current_index_ = tl.Size() + 1;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, 0);
}

TYPED_TEST(ReaderTest, ResetLoaderTestStickToShard2) {
  TestLoader tl(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 1)
      .AddArg("num_shards", 2)
      .AddArg("stick_to_shard", true)
      .AddArg("batch_size", 2)
      .AddArg("device_id", 0));
  tl.PrepareMetadata();

  ASSERT_EQ(tl.IsNextShard(tl.Size() / 2), false);
  ASSERT_EQ(tl.IsNextShard(tl.Size() - 1), false);
  ASSERT_EQ(tl.IsNextShard(tl.Size())    , true);
  ASSERT_EQ(tl.IsNextShard(tl.Size() + 1), true);
  tl.current_index_ = 1;
  tl.Reset(true);
  ASSERT_EQ(tl.current_index_, tl.Size() / 2);

  tl.current_index_ = tl.Size() / 2;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, tl.Size() / 2);

  tl.current_index_ = tl.Size() - 1;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, tl.Size() - 1);

  tl.current_index_ = tl.Size();
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, tl.Size() / 2);

  tl.current_index_ = tl.Size() + 1;
  tl.MoveToNextShard(tl.current_index_);
  ASSERT_EQ(tl.current_index_, tl.Size() / 2);
}

};  // namespace dali
