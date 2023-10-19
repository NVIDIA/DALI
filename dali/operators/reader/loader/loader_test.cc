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
  for (bool dont_use_mmap : {true, false}) {
    shared_ptr<dali::FileLabelLoader> reader(
        new FileLabelLoader(
            OpSpec("FileReader")
            .AddArg("file_root", loader_test_image_folder)
            .AddArg("max_batch_size", 32)
            .AddArg("device_id", 0)
            .AddArg("dont_use_mmap", dont_use_mmap)));

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
    EXPECT_EQ(sample->shares_data(), !dont_use_mmap);
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
    EXPECT_EQ(sample->shares_data(), !dont_use_mmap);
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
  shared_ptr<dali::FileLabelLoader> reader(
      new FileLabelLoader(
          OpSpec("FileReader")
          .AddArg("file_root", loader_test_image_folder)
          .AddArg("max_batch_size", 32)
          .AddArg("device_id", 0)));

  reader->PrepareMetadata();

  for (int i = 0; i < 11; ++i) {
    // grab an entry from the reader
    // tensor should be returned automatically when
    // shared_ptr to sample is destroyed
    auto sample = reader->ReadOne(false, false);
  }
}

TYPED_TEST(DataLoadStoreTest, LoaderTestFail) {
  shared_ptr<dali::FileLabelLoader> reader(
      new FileLabelLoader(OpSpec("FileReader")
                         .AddArg("file_root", loader_test_image_folder + "/does_not_exist")
                         .AddArg("max_batch_size", 32)
                         .AddArg("device_id", 0)));
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
  explicit DummyCountingLoader(const OpSpec& spec, uint64_t size) :
    Loader<CPUBackend, Tensor<CPUBackend>, true>(spec),
    size_(size), counter_(0) {}

  void ReadSample(Tensor<CPUBackend> &t) override {
    t.Resize({1}, DALI_UINT64);
    *t.mutable_data<uint64_t>() = counter_++;
  }

  void PrepareMetadataImpl() override {}

  Index SizeImpl() override {
    return size_;
  }

  void Skip(uint64_t n) override {
    counter_ += n;
  }

  void Rewind(bool wrap_to_shard) override {
    counter_ = 0;
  }

  void Reset(bool wrap_to_shard) override {
    Rewind(wrap_to_shard);
  }

  uint64_t ReadInt(bool is_new_batch, bool is_end_of_batch) {
    return *ReadOne(is_new_batch, is_end_of_batch)->data<uint64_t>();
  }

  std::vector<uint64_t> ReadInts(size_t n) {
    std::vector<uint64_t> result(n);
    for (size_t i = 0; i < n; i++) {
      result[i] = ReadInt(i % max_batch_size_ == 0, (i + 1) % max_batch_size_ == 0);
    }
    return result;
  }

 private:
  uint64_t size_;
  uint64_t counter_;
};

void testFastForward(const OpSpec &spec, uint64_t data_size, int steps) {
  auto reference = InitLoader<DummyCountingLoader>(spec, data_size)->ReadInts(steps);
  auto loader = InitLoader<DummyCountingLoader>(spec, data_size);

  int pos = 0;
  int fast_forward_distance = 0;
  while (pos + 3 < steps) {
    for (int i = 0; i < 3; i++) {
      EXPECT_EQ(loader->ReadInt(false, false), reference[pos]);
      pos++;
    }
    loader->FastForward(fast_forward_distance);
    pos += fast_forward_distance;
    fast_forward_distance++;
  }
}

TEST(LoaderCheckpointingTest, TestFastForwardNoShuffle) {
  auto spec = OpSpec("FileReader")
                .AddArg("device_id", 0)
                .AddArg("max_batch_size", 256);
  testFastForward(spec, 200, 50);
}

TEST(LoaderCheckpointingTest, TestFastForwardShuffled) {
  auto spec = OpSpec("FileReader")
                .AddArg("device_id", 0)
                .AddArg("max_batch_size", 256)
                .AddArg("initial_fill", 10)
                .AddArg("random_shuffle", true)
                .AddArg("seed", 123);
  testFastForward(spec, 200, 50);
}

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

};  // namespace dali
