// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {

class DummyLoader : public Loader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit DummyLoader(const OpSpec& spec, const std::string &dummyfile = "") :
    Loader<CPUBackend, Tensor<CPUBackend>>(spec),
    dummyfile_(dummyfile) {}

  void ReadSample(Tensor<CPUBackend> &t) override {
    t.Resize({1}, DALI_UINT8);
  }

  void PrepareMetadataImpl() override {
    if (dummyfile_ != "") {
      std::ifstream f(dummyfile_);
      if (!f)
        throw std::runtime_error("Failed to open " + dummyfile_);
    }
  }

  Index SizeImpl() override {
    return 1;
  }

  void Reset(bool wrap_to_shard) override {}

 private:
  std::string dummyfile_;
};
class DummyDataReader : public DataReader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit DummyDataReader(const OpSpec &spec)
      : DataReader<CPUBackend, Tensor<CPUBackend>>(spec),
        count_(0) {
    std::string dummyfile("");
    if (spec.HasArgument("dummyfile")) {
      dummyfile = spec.GetArgument<std::string>("dummyfile");
    }
    loader_ = InitLoader<DummyLoader>(spec, std::move(dummyfile));
  }

  ~DummyDataReader() override {
    DataReader<CPUBackend, Tensor<CPUBackend>>::StopPrefetchThread();
  }
  /*
  using DataReader<CPUBackend>::prefetched_batch_;
  bool Prefetch() override {
    for (int i = 0; i < Operator::batch_size_; ++i) {
      printf("new tensor %d\n", i);
      auto *t = loader_->ReadOne(false, false);
      prefetched_batch_.push_back(t);
    }
    return true;
  }
  */

  void RunImpl(SampleWorkspace &ws) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    ws.Output<CPUBackend>(0).Copy(GetSample(ws.data_idx()));
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

  Workspace ws;
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

  Workspace ws;
  for (int i=0; i < 5; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  }
  return;
}

TYPED_TEST(ReaderTest, LazyInitTest) {
  Pipeline eager_pipe(32, 1, 0);
  Pipeline lazy_pipe(32, 1, 0);

  // This file does not exist yet
  std::string filename("/tmp/dalidummyfile.txt");
  std::remove(filename.c_str());

  eager_pipe.AddOperator(
      OpSpec("DummyDataReader")
      .AddOutput("data_out", "cpu")
      .AddArg("prefetch_queue_depth", 3)
      .AddArg("lazy_init", false)
      .AddArg("dummyfile", filename));

  lazy_pipe.AddOperator(
      OpSpec("DummyDataReader")
      .AddOutput("data_out", "cpu")
      .AddArg("prefetch_queue_depth", 3)
      .AddArg("lazy_init", true)
      .AddArg("dummyfile", filename));

  // File `filename` doesnt exist yet
  std::vector<std::pair<string, string>> outputs = {{"data_out", "cpu"}};

  ASSERT_ANY_THROW(eager_pipe.Build(outputs));
  // Eager pipeline threw, we don't care anymore

  ASSERT_NO_THROW(lazy_pipe.Build(outputs));

  // Creating the file
  std::ofstream dummyfs(filename);
  dummyfs.close();

  // This calls PrepareMetadataImpl
  ASSERT_NO_THROW(lazy_pipe.GetReaderMeta());

  Workspace ws;
  for (int i=0; i < 5; ++i) {
    lazy_pipe.RunCPU();
    lazy_pipe.RunGPU();
    lazy_pipe.Outputs(&ws);
  }
  std::remove(filename.c_str());
  return;
}

TYPED_TEST(ReaderTest, SequenceTest) {
  Pipeline pipe(128, 4, 0);

  pipe.AddOperator(
      OpSpec("SequenceReader")
      .AddArg("file_root", testing::dali_extra_path() + "/db/sequence/frames")
      .AddArg("sequence_length", 3)
      .AddArg("image_type", DALI_RGB)
      .AddOutput("seq_out", "cpu"));

  std::vector<std::pair<string, string>> outputs = {{"seq_out", "cpu"}};
  pipe.Build(outputs);

  Workspace ws;
  for (int i = 0; i < 4; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    auto shape = ws.Output<CPUBackend>(0).AsTensor().shape();
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
        auto off = frame * frame_size;
        auto val = ws.Output<CPUBackend>(0).tensor<uint8_t>(sample)[off];
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
    return internal_size_;
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
  Index internal_size_ = 10;
};

TYPED_TEST(ReaderTest, ResetLoaderTestWrap) {
  TestLoader tl(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 0)
      .AddArg("num_shards", 2)
      .AddArg("stick_to_shard", false)
      .AddArg("max_batch_size", 2)
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
      .AddArg("max_batch_size", 2)
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
      .AddArg("max_batch_size", 2)
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

TYPED_TEST(ReaderTest, ResetLoaderTestNoPad) {
  TestLoader tl_even(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 1)
      .AddArg("num_shards", 2)
      .AddArg("stick_to_shard", true)
      .AddArg("max_batch_size", 2)
      .AddArg("device_id", 0)
      .AddArg("pad_last_batch", false));
  tl_even.PrepareMetadata();

  TestLoader tl_odd(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 1)
      .AddArg("num_shards", 3)
      .AddArg("stick_to_shard", true)
      .AddArg("max_batch_size", 2)
      .AddArg("device_id", 0)
      .AddArg("pad_last_batch", false));
  tl_odd.PrepareMetadata();

  tl_even.internal_size_ = 10;
  tl_odd.internal_size_ = 10;

  ASSERT_EQ(tl_even.Size(), tl_even.Size(true));
  ASSERT_EQ(tl_odd.Size(), tl_odd.Size(true));
  ASSERT_EQ(tl_even.Size(), tl_even.Size(false));
  ASSERT_EQ(tl_odd.Size(), tl_odd.Size(false));
}

TYPED_TEST(ReaderTest, ResetLoaderTestPad) {
  TestLoader tl_even(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 1)
      .AddArg("num_shards", 2)
      .AddArg("stick_to_shard", true)
      .AddArg("max_batch_size", 2)
      .AddArg("device_id", 0)
      .AddArg("pad_last_batch", true));
  tl_even.PrepareMetadata();

  TestLoader tl_odd(
      OpSpec("FileReader")
      .AddOutput("data_out", "cpu")
      .AddArg("shard_id", 1)
      .AddArg("num_shards", 3)
      .AddArg("stick_to_shard", true)
      .AddArg("max_batch_size", 2)
      .AddArg("device_id", 0)
      .AddArg("pad_last_batch", true));
  tl_odd.PrepareMetadata();

  tl_even.internal_size_ = 10;
  tl_odd.internal_size_ = 10;

  ASSERT_EQ(tl_even.Size(), tl_even.internal_size_);
  ASSERT_EQ(tl_odd.Size(), tl_odd.internal_size_);

  ASSERT_EQ(tl_even.Size(), tl_even.Size(true));
  ASSERT_NE(tl_odd.Size(), tl_odd.Size(true));

  ASSERT_EQ(tl_even.Size(), tl_even.Size(false));
  ASSERT_EQ(tl_odd.Size(), tl_odd.Size(false));

  ASSERT_EQ(tl_even.Size(true), tl_even.internal_size_);
  ASSERT_EQ(tl_odd.Size(true),
            static_cast<Index>(std::ceil(tl_odd.internal_size_ * 1.0 / 3)) * 3);
}

TEST(ReaderTestSimple, CheckNumSamples) {
  ASSERT_EQ(num_samples(1, 10), 10 / 1);
  ASSERT_EQ(num_samples(2, 10), 10 / 2);
  ASSERT_EQ(num_samples(3, 10), static_cast<Index>(std::ceil(10 * 1.0 / 3)));
}

class FileReaderTest : public DALITest {
 public:
  void SetUp() override {
    constexpr int filecount = 16;

    for (int i = 0; i < filecount; i++) {
      std::string path = make_string("/tmp/dali_test_file_", to_string(i));
      filepaths_.push_back(path);

      // write a single byte to the file
      std::ofstream f(path);
      f << (uint8_t) i;
    }
  }

  void TearDown() override {
    for (const auto &path : filepaths_) {
      int ret = std::remove(path.data());
      DALI_ENFORCE(ret == 0,
                   make_string("Failed to remove the file: ", path, ". Error code: ", ret));
    }
  }

  std::vector<std::string> filepaths_;

  OpSpec MakeOpSpec() {
      // Currently, only pad_last_batch=true is supported.
      return OpSpec("FileReader")
            .AddOutput("data_out", "cpu")
            .AddOutput("labels", "cpu")
            .AddArg("files", filepaths_)
            .AddArg("checkpointing", true)
            .AddArg("pad_last_batch", true);
  }

  void BuildPipeline(Pipeline &pipe) {
    std::vector<std::pair<string, string>> outputs = {{"data_out", "cpu"}};
    pipe.Build(outputs);
  }

  std::vector<uint8_t> RunEpoch(Pipeline &pipe, int batch_size,
                                int num_shards = 1, bool stick_to_shard = false) {
    std::vector<uint8_t> result;
    int samples_per_shard = (filepaths_.size() + num_shards - 1) / num_shards;
    int batches_per_shard = (samples_per_shard + batch_size - 1) / batch_size;
    for (int it = 0; it < batches_per_shard; it++) {
      pipe.RunCPU();
      pipe.RunGPU();
      pipe.Outputs(&ws_);

      auto shape = ws_.Output<CPUBackend>(0).AsTensor().shape();
      for (int nr = 0; nr < shape[0]; nr++)
        result.push_back(ws_.Output<CPUBackend>(0).tensor<uint8_t>(0)[nr]);
    }
    return result;
  }

  std::pair<vector<uint8_t>, OpCheckpoint> CheckpointEpoch(Pipeline &pipe, int batch_size,
                                                           int epoch_nr, int num_shards = 1,
                                                           bool stick_to_shard = false) {
    auto node = pipe.GetOperatorNode("file_reader");
    OpCheckpoint cpt(node->spec);
    node->op->SaveState(cpt, std::nullopt);
    EXPECT_EQ(cpt.CheckpointState<LoaderStateSnapshot>().current_epoch, epoch_nr);
    return {RunEpoch(pipe, batch_size, num_shards, stick_to_shard), cpt};
  }

  void RestoreCheckpointedState(Pipeline &pipe, const OpCheckpoint &cpt) {
    auto node = pipe.GetOperatorNode("file_reader");
    node->op->RestoreState(cpt);
  }

 protected:
  Workspace ws_;
};

TEST_F(FileReaderTest, SimpleCheckpointing) {
  constexpr int batch_size = 3;
  constexpr int epochs = 4;

  auto prepare_pipeline = [this](Pipeline &pipe) {
    pipe.AddOperator(
        MakeOpSpec()
        .AddArg("prefetch_queue_depth", 20)
        .AddArg("initial_fill", 3), "file_reader");
    BuildPipeline(pipe);
  };

  Pipeline pipe(batch_size, 1, 0);
  prepare_pipeline(pipe);
  std::vector<std::vector<uint8_t>> results;
  std::vector<OpCheckpoint> checkpoints;
  for (int i = 0; i < epochs; i++) {
    auto [res, cpt] = CheckpointEpoch(pipe, batch_size, i);
    results.push_back(res);
    checkpoints.push_back(cpt);
  }

  for (int i = 0; i < epochs; i++) {
    Pipeline fresh_pipe(batch_size, 1, 0);
    prepare_pipeline(fresh_pipe);
    RestoreCheckpointedState(fresh_pipe, checkpoints[i]);
    EXPECT_EQ(RunEpoch(fresh_pipe, batch_size), results[i]);
  }
}

TEST_F(FileReaderTest, CheckpointingRandomShuffle) {
  constexpr int batch_size = 7;
  constexpr int epochs = 8;

  auto prepare_pipeline = [this](Pipeline &pipe) {
    pipe.AddOperator(
        MakeOpSpec()
        .AddArg("random_shuffle", true)
        .AddArg("initial_fill", 3), "file_reader");
    BuildPipeline(pipe);
  };

  Pipeline pipe(batch_size, 1, 0);
  prepare_pipeline(pipe);
  std::vector<std::vector<uint8_t>> results;
  std::vector<OpCheckpoint> checkpoints;
  for (int i = 0; i < epochs; i++) {
    auto [res, cpt] = CheckpointEpoch(pipe, batch_size, i);
    results.push_back(res);
    checkpoints.push_back(cpt);
  }

  for (int i = 0; i < epochs; i++) {
    Pipeline fresh_pipe(batch_size, 1, 0);
    prepare_pipeline(fresh_pipe);
    RestoreCheckpointedState(fresh_pipe, checkpoints[i]);
    EXPECT_EQ(RunEpoch(fresh_pipe, batch_size), results[i]);
  }
}

TEST_F(FileReaderTest, CheckpointingShuffleAfterEpoch) {
  constexpr int batch_size = 5;
  constexpr int epochs = 8;

  auto prepare_pipeline = [this](Pipeline &pipe) {
    pipe.AddOperator(
        MakeOpSpec()
        .AddArg("shuffle_after_epoch", true)
        .AddArg("prefetch_queue_depth", 9)
        .AddArg("initial_fill", 10), "file_reader");
    BuildPipeline(pipe);
  };

  Pipeline pipe(batch_size, 1, 0);
  prepare_pipeline(pipe);
  std::vector<std::vector<uint8_t>> results;
  std::vector<OpCheckpoint> checkpoints;
  for (int i = 0; i < epochs; i++) {
    auto [res, cpt] = CheckpointEpoch(pipe, batch_size, i);
    results.push_back(res);
    checkpoints.push_back(cpt);
  }

  for (int i = 0; i < epochs; i++) {
    Pipeline fresh_pipe(batch_size, 1, 0);
    prepare_pipeline(fresh_pipe);
    RestoreCheckpointedState(fresh_pipe, checkpoints[i]);
    EXPECT_EQ(RunEpoch(fresh_pipe, batch_size), results[i]);
  }
}

TEST_F(FileReaderTest, CheckpointingMultipleShards) {
  constexpr int batch_size = 5;
  constexpr int epochs = 8;
  constexpr int num_shards = 5;

  auto prepare_pipeline = [this, num_shards](Pipeline &pipe) {
    pipe.AddOperator(
        MakeOpSpec()
        .AddArg("shard_id", 2)
        .AddArg("num_shards", num_shards)
        .AddArg("prefetch_queue_depth", 11)
        .AddArg("initial_fill", 20), "file_reader");
    BuildPipeline(pipe);
  };

  Pipeline pipe(batch_size, 1, 0);
  prepare_pipeline(pipe);
  std::vector<std::vector<uint8_t>> results;
  std::vector<OpCheckpoint> checkpoints;
  for (int i = 0; i < epochs; i++) {
    auto [res, cpt] = CheckpointEpoch(pipe, batch_size, i, num_shards);
    results.push_back(res);
    checkpoints.push_back(cpt);
  }

  for (int i = 0; i < epochs; i++) {
    Pipeline fresh_pipe(batch_size, 1, 0);
    prepare_pipeline(fresh_pipe);
    RestoreCheckpointedState(fresh_pipe, checkpoints[i]);
    EXPECT_EQ(RunEpoch(fresh_pipe, batch_size, num_shards), results[i]);
  }
}

TEST_F(FileReaderTest, CheckpointingStickStoShard) {
  constexpr int batch_size = 3;
  constexpr int epochs = 8;
  constexpr int num_shards = 3;

  auto prepare_pipeline = [this, num_shards](Pipeline &pipe) {
    pipe.AddOperator(
        MakeOpSpec()
        .AddArg("stick_to_shard", true)
        .AddArg("shard_id", 1)
        .AddArg("num_shards", num_shards)
        .AddArg("initial_fill", 5), "file_reader");
    BuildPipeline(pipe);
  };

  Pipeline pipe(batch_size, 1, 0);
  prepare_pipeline(pipe);
  std::vector<std::vector<uint8_t>> results;
  std::vector<OpCheckpoint> checkpoints;
  for (int i = 0; i < epochs; i++) {
    auto [res, cpt] = CheckpointEpoch(pipe, batch_size, i, num_shards, true);
    results.push_back(res);
    checkpoints.push_back(cpt);
  }

  for (int i = 0; i < epochs; i++) {
    Pipeline fresh_pipe(batch_size, 1, 0);
    prepare_pipeline(fresh_pipe);
    RestoreCheckpointedState(fresh_pipe, checkpoints[i]);
    EXPECT_EQ(RunEpoch(fresh_pipe, batch_size, num_shards, true), results[i]);
  }
}


TEST_F(FileReaderTest, CheckpointingResumeThenSave) {
  constexpr int batch_size = 3;
  constexpr int epochs = 4;

  auto prepare_pipeline = [this](Pipeline &pipe) {
    pipe.AddOperator(
        MakeOpSpec()
        .AddArg("shuffle_after_epoch", true)
        .AddArg("prefetch_queue_depth", 2)
        .AddArg("initial_fill", 3), "file_reader");
    BuildPipeline(pipe);
  };

  Pipeline pipe(batch_size, 1, 0);
  prepare_pipeline(pipe);
  std::vector<std::vector<uint8_t>> results;
  std::vector<OpCheckpoint> checkpoints;
  for (int i = 0; i < epochs; i++) {
    auto [res, cpt] = CheckpointEpoch(pipe, batch_size, i);
    results.push_back(res);
    checkpoints.push_back(cpt);
  }

  for (int i = 0; i < epochs; i++) {
    Pipeline intermediate_pipe(batch_size, 1, 0);
    prepare_pipeline(intermediate_pipe);
    RestoreCheckpointedState(intermediate_pipe, checkpoints[i]);

    std::vector<OpCheckpoint> cpts_after_resume;
    for (int j = 0; i + j < epochs; j++) {
      auto [res, cpt] = CheckpointEpoch(intermediate_pipe, batch_size, i + j);
      EXPECT_EQ(res, results[i + j]);
      cpts_after_resume.push_back(cpt);
    }

    for (int j = 0; i + j < epochs; j++) {
      Pipeline fresh_pipe(batch_size, 1, 0);
      prepare_pipeline(fresh_pipe);
      RestoreCheckpointedState(fresh_pipe, cpts_after_resume[j]);
      EXPECT_EQ(RunEpoch(fresh_pipe, batch_size), results[i + j]);
    }
  }
}

};  // namespace dali
