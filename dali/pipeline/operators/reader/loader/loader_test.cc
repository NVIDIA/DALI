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
#include "dali/pipeline/operators/reader/loader/lmdb.h"

namespace dali {

template <typename Backend>
class DataStoreTest : public DALITest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(DataStoreTest, TestTypes);

const char* path = std::getenv("DALI_TEST_CAFFE_LMDB_PATH");

TYPED_TEST(DataStoreTest, LMDBTest) {
  shared_ptr<dali::LMDBReader> reader(
      new LMDBReader(
          OpSpec("CaffeReader")
          .AddArg("batch_size", 32)
          .AddArg("path", string(path))));

  for (int i = 0; i < 10500; ++i) {
    // grab an entry from the reader
    auto* sample = reader->ReadOne();
    // return the tensor to the reader for refilling
    reader->ReturnTensor(sample);
  }

  return;
}

#if 0
TYPED_TEST(DataStoreTest, CachedLMDBTest) {
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
    reader->ReturnTensor(sample);
  }

  return;
}
#endif

};  // namespace dali
