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

#include <gtest/gtest.h>
#include <memory>

#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/test/dali_test.h"

#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/operators/reader/loader/recordio_loader.h"
#include "dali/operators/reader/loader/indexed_file_loader.h"
#include "dali/operators/reader/loader/coco_loader.h"

#if LMDB_ENABLED
#include "dali/operators/reader/loader/lmdb.h"
#endif

namespace dali {

template <typename Backend>
class DataLoadStoreTest : public DALITest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;
string loader_test_image_folder = testing::dali_extra_path() + "/db/single/jpeg";  // NOLINT

TYPED_TEST_SUITE(DataLoadStoreTest, TestTypes);

#if LMDB_ENABLED
TYPED_TEST(DataLoadStoreTest, LMDBTest) {
  shared_ptr<dali::LMDBLoader> reader(
      new LMDBLoader(
          OpSpec("CaffeReader")
          .AddArg("max_batch_size", 32)
          .AddArg("path", testing::dali_extra_path() + "/db/c2lmdb/")
          .AddArg("device_id", 0)));

  reader->PrepareMetadata();

  for (int i = 0; i < 10500; ++i) {
    // grab an entry from the reader
    // tensor should be returned automatically when
    // shared_ptr to sample is destroyed
    auto sample = reader->ReadOne(false, false);
  }
}
#endif

TYPED_TEST(DataLoadStoreTest, FileLabelLoaderMmmap) {
  bool shuffle_after_epoch = false;
  for (bool dont_use_mmap : {true, false}) {
    shared_ptr<dali::FileLabelLoader> reader(
        new FileLabelLoader(
            OpSpec("FileReader")
            .AddArg("file_root", loader_test_image_folder)
            .AddArg("max_batch_size", 32)
            .AddArg("device_id", 0)
            .AddArg("dont_use_mmap", dont_use_mmap), shuffle_after_epoch));

    reader->PrepareMetadata();
    auto sample = reader->ReadOne(false, false);
    EXPECT_EQ(sample->image.shares_data(), !dont_use_mmap);
  }
}

TYPED_TEST(DataLoadStoreTest, RecordIOLoaderMmmap) {
  for (bool dont_use_mmap : {true, false}) {
    std::vector<std::string> path =  {testing::dali_extra_path() + "/db/recordio/train.rec"};
    std::vector<std::string> index_path = {testing::dali_extra_path() + "/db/recordio/train.idx"};
    shared_ptr<dali::RecordIOLoader> reader(
        new RecordIOLoader(
            OpSpec("MXNetReader")
            .AddArg("path", path)
            .AddArg("index_path", index_path)
            .AddArg("max_batch_size", 32)
            .AddArg("device_id", 0)
            .AddArg("dont_use_mmap", dont_use_mmap)));

    reader->PrepareMetadata();
    auto sample = reader->ReadOne(false, false);
    EXPECT_EQ(sample->tensor.shares_data(), !dont_use_mmap);
  }
}

TYPED_TEST(DataLoadStoreTest, TFRecordLoaderMmmap) {
  for (bool dont_use_mmap : {true, false}) {
    std::vector<std::string> path = {testing::dali_extra_path() + "/db/tfrecord/train"};
    std::vector<std::string> index_path = {testing::dali_extra_path() + "/db/tfrecord/train.idx"};
    shared_ptr<dali::IndexedFileLoader> reader(
        new IndexedFileLoader(
            OpSpec("TFRecordReader")
            .AddArg("path", path)
            .AddArg("index_path", index_path)
            .AddArg("max_batch_size", 32)
            .AddArg("device_id", 0)
            .AddArg("dont_use_mmap", dont_use_mmap)));

    reader->PrepareMetadata();
    auto sample = reader->ReadOne(false, false);
    EXPECT_EQ(sample->tensor.shares_data(), !dont_use_mmap);
  }
}

TYPED_TEST(DataLoadStoreTest, CocoLoaderMmmap) {
  for (bool dont_use_mmap : {true, false}) {
    std::string file_root = testing::dali_extra_path() + "/db/coco/images";
    std::string annotations_file = testing::dali_extra_path() + "/db/coco/instances.json";
    auto coco_spec = OpSpec("COCOReader")
                      .AddArg("file_root", file_root)
                      .AddArg("annotations_file", annotations_file)
                      .AddArg("max_batch_size", 32)
                      .AddArg("device_id", 0)
                      .AddArg("dont_use_mmap", dont_use_mmap);
    shared_ptr<dali::CocoLoader> reader(new CocoLoader(coco_spec));
    reader->PrepareMetadata();
    auto sample = reader->ReadOne(false, false);
    EXPECT_EQ(sample->image.shares_data(), !dont_use_mmap);
  }
}

TYPED_TEST(DataLoadStoreTest, LoaderTest) {
  bool shuffle_after_epoch = false;
  shared_ptr<dali::FileLabelLoader> reader(
      new FileLabelLoader(
          OpSpec("FileReader")
          .AddArg("file_root", loader_test_image_folder)
          .AddArg("max_batch_size", 32)
          .AddArg("device_id", 0), shuffle_after_epoch));

  reader->PrepareMetadata();

  for (int i = 0; i < 11; ++i) {
    // grab an entry from the reader
    // tensor should be returned automatically when
    // shared_ptr to sample is destroyed
    auto sample = reader->ReadOne(false, false);
  }
}

TYPED_TEST(DataLoadStoreTest, LoaderTestFail) {
  bool shuffle_after_epoch = false;
  shared_ptr<dali::FileLabelLoader> reader(
      new FileLabelLoader(OpSpec("FileReader")
                         .AddArg("file_root", loader_test_image_folder + "/does_not_exist")
                         .AddArg("max_batch_size", 32)
                         .AddArg("device_id", 0), shuffle_after_epoch));
  ASSERT_THROW(reader->PrepareMetadata(), std::runtime_error);
}

#if 0
TYPED_TEST(DataLoadStoreTest, CachedLMDBTest) {
  shared_ptr<dali::LMDBLoader> reader(
      new LMDBLoader(
          OpSpec("CaffeReader")
          .AddArg("max_batch_size", 32)
          .AddArg("enable_cache", true)
          .AddArg("path", string(path))));

  for (int i = 0; i < 10500; ++i) {
    // grab an entry from the reader
    auto sample = reader->ReadOne(false, false);
  }

  return;
}
#endif


class DummyCountingLoader : public Loader<CPUBackend, Tensor<CPUBackend>, true> {
 public:
  explicit DummyCountingLoader(const OpSpec& spec, uint64_t size, uint64_t mark_epoch = 0) :
    Loader<CPUBackend, Tensor<CPUBackend>, true>(spec),
    size_(size), counter_(0), mark_epoch_(mark_epoch) {}

  void ReadSample(Tensor<CPUBackend> &t) override {
    t.Resize({1}, DALI_UINT64);
    *t.mutable_data<uint64_t>() = (counter_++) + epoch_ * mark_epoch_;
    if (counter_ % size_ == 0) Reset(stick_to_shard_);
  }

  void PrepareMetadataImpl() override {}

  Index SizeImpl() override {
    return size_;
  }

  void Reset(bool wrap_to_shard) override {
    counter_ = 0;
    epoch_++;
  }

  void RestoreStateImpl(const LoaderStateSnapshot &state) override {
    epoch_ = state.current_epoch;
  }

  uint64_t ReadInt(bool is_new_batch, bool is_end_of_batch) {
    return *ReadOne(is_new_batch, is_end_of_batch)->data<uint64_t>();
  }

  std::vector<uint64_t> ReadInts(size_t n) {
    std::vector<uint64_t> result(n);
    for (size_t i = 0; i < n; i++) {
      result[i] = ReadInt(counter_ % max_batch_size_ == 0, (counter_ + 1) % max_batch_size_ == 0);
    }
    return result;
  }

 private:
  uint64_t size_;
  uint64_t counter_;
  uint64_t mark_epoch_;
  uint64_t epoch_ = 1;
};

void TestLoaderCheckpointing(const std::unique_ptr<DummyCountingLoader> &loader, int n) {
  std::vector<uint64_t> reference;
  std::vector<LoaderStateSnapshot> snapshots;
  for (int i = 0; i < n; i++) {
    snapshots.push_back(loader->GetStateSnapshot());
    reference.push_back(loader->ReadInt(false, false));
  }

  for (int start = 0; start < n; start++) {
    loader->RestoreStateFromSnapshot(snapshots[start]);
    std::vector<uint64_t> rest;
    for (int i = start; i < n; i++) {
      rest.push_back(loader->ReadInt(false, false));
    }
    std::vector<uint64_t> expected{reference.begin() + start, reference.end()};
    EXPECT_EQ(rest, expected);
  }
}

TEST(LoaderCheckpointingTest, TestCheckpoint) {
  auto spec = OpSpec("FileReader")
                .AddArg("device_id", 0)
                .AddArg("max_batch_size", 32)
                .AddArg("seed", 123)
                .AddArg("checkpointing", true)
                .AddArg("pad_last_batch", true);

  TestLoaderCheckpointing(InitLoader<DummyCountingLoader>(spec, 30), 20);
}

TEST(LoaderCheckpointingTest, TestCheckpointShuffled) {
  auto spec = OpSpec("FileReader")
                .AddArg("device_id", 0)
                .AddArg("max_batch_size", 32)
                .AddArg("seed", 123)
                .AddArg("random_shuffle", true)
                .AddArg("initial_fill", 10)
                .AddArg("checkpointing", true)
                .AddArg("pad_last_batch", true);

  TestLoaderCheckpointing(InitLoader<DummyCountingLoader>(spec, 30), 20);
}

TEST(LoaderCheckpointingTest, TestCheckpointNoPadding) {
  auto spec = OpSpec("FileReader")
                .AddArg("device_id", 0)
                .AddArg("max_batch_size", 32)
                .AddArg("seed", 123)
                .AddArg("random_shuffle", true)
                .AddArg("initial_fill", 10)
                .AddArg("checkpointing", true);

  TestLoaderCheckpointing(InitLoader<DummyCountingLoader>(spec, 10), 20);
}

/* This test represents an unlikely situation where dataset is much smaller than sample buffer,
   resulting in multiple epochs of data being stored in the sample buffer at the same time. 
   If loader doesn't return the same sequence of samples in each epoch (for example due to
   shuffle_after_epoch) it is important to keep track of each sample's epoch during 
   fast-forward. */
TEST(LoaderCheckpointingTest, TestCheckpointShortEpoch) {
  auto spec = OpSpec("FileReader")
                .AddArg("device_id", 0)
                .AddArg("max_batch_size", 4)
                .AddArg("seed", 123)
                .AddArg("random_shuffle", true)
                .AddArg("initial_fill", 30)
                .AddArg("checkpointing", true);

  TestLoaderCheckpointing(InitLoader<DummyCountingLoader>(
    spec,
    8   /* epoch size, that's 3 epochs in a sample buffer! */,
    100 /* add 100*current_epoch to each output to differentiate samples */),
  32);
}

};  // namespace dali
