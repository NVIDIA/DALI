// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include <gtest/gtest.h>
#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/operators/op_spec.h"
#include "ndll/test/ndll_test.h"

#include "ndll/pipeline/operators/reader/loader/loader.h"
#include "ndll/pipeline/operators/reader/loader/lmdb.h"

namespace ndll {

template <typename Backend>
class DataStoreTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(DataStoreTest, TestTypes);

const char* path = std::getenv("NDLL_TEST_CAFFE_LMDB_PATH");

TYPED_TEST(DataStoreTest, LMDBTest) {
  shared_ptr<ndll::LMDBReader> reader(
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
  shared_ptr<ndll::LMDBReader> reader(
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

};  // namespace ndll
