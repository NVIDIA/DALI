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

#include "dali/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/test/dali_test.h"

#include "dali/pipeline/operators/reader/loader/loader.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/pipeline/operators/reader/loader/lmdb.h"

namespace dali {

template <typename Backend>
class DataLoadStoreTest : public DALITest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;
string loader_test_image_folder = "/data/dali/benchmark";  // NOLINT

TYPED_TEST_SUITE(DataLoadStoreTest, TestTypes);

const char* path = std::getenv("DALI_TEST_CAFFE_LMDB_PATH");

TYPED_TEST(DataLoadStoreTest, LMDBTest) {
  shared_ptr<dali::LMDBReader> reader(
      new LMDBReader(
          OpSpec("CaffeReader")
          .AddArg("batch_size", 32)
          .AddArg("path", string(path))
          .AddArg("device_id", 0)));

  reader->PrepareMetadata();

  for (int i = 0; i < 10500; ++i) {
    // grab an entry from the reader
    // return the tensor to the reader for refilling
    reader->RecycleTensor(reader->ReadOne());
  }

  return;
}

TYPED_TEST(DataLoadStoreTest, LoaderTest) {
  shared_ptr<dali::FileLoader> reader(
      new FileLoader(
          OpSpec("FileReader")
          .AddArg("file_root", loader_test_image_folder)
          .AddArg("batch_size", 32)
          .AddArg("device_id", 0)));

  reader->PrepareMetadata();

  for (int i = 0; i < 11; ++i) {
    // grab an entry from the reader
    // return the tensor to the reader for refilling
    reader->RecycleTensor(reader->ReadOne());
  }

  return;
}

TYPED_TEST(DataLoadStoreTest, LoaderTestFail) {
  try {
    shared_ptr<dali::FileLoader> reader(
        new FileLoader(
            OpSpec("FileReader")
            .AddArg("file_root", loader_test_image_folder + "/benchmark_images")
            .AddArg("batch_size", 32)
            .AddArg("device_id", 0)));
        reader->PrepareMetadata();
  }
  catch (std::runtime_error &e) {
    return;
  }
  throw std::runtime_error("LoaderTestFail failed");
}

#if 0
TYPED_TEST(DataLoadStoreTest, CachedLMDBTest) {
  shared_ptr<dali::LMDBReader> reader(
      new LMDBReader(
          OpSpec("CaffeReader")
          .AddArg("batch_size", 32)
          .AddArg("enable_cache", true)
          .AddArg("path", string(path))));

  for (int i = 0; i < 10500; ++i) {
    // grab an entry from the reader
    auto* sample = reader->ReadOne();
    // return the tensor to the reader for refilling
    reader->RecycleTensor(sample);
  }

  return;
}
#endif

};  // namespace dali
