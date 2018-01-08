// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include <gtest/gtest.h>
#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/test/ndll_test.h"

#include "ndll/pipeline/loader/loader.h"
#include "ndll/pipeline/loader/lmdb.h"

namespace ndll {

template <typename Backend>
class DataStoreTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(DataStoreTest, TestTypes);

const char* path =
    "/home/slayton/opt/caffe2-18.01/nvidia-examples/mnist/mnist_test_lmdb";

TYPED_TEST(DataStoreTest, LMDB_test) {
  shared_ptr<ndll::LMDBReader> reader(
      new LMDBReader(
          OpSpec("lmdb")
          .AddArg("path", string(path))));

  for (int i = 0; i < 10500; ++i) {
    // grab an entry from the reader
    auto* sample = reader->ReadOne();
    // return the tensor to the reader for refilling
    reader->ReturnTensor(sample);
  }

  return;
}

};  // namespace ndll
