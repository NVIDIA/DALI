// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <thread>

#include "ndll/pipeline/reader_op.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/sample_workspace.h"
#include "ndll/test/ndll_test.h"

namespace ndll {

template <typename Backend>
class DummyDataReader : public DataReader<Backend> {
 public:
  explicit DummyDataReader(const OpSpec &spec)
      : DataReader<Backend>(spec),
        count_(0) {}

  ~DummyDataReader() {
    DataReader<Backend>::StopPrefetchThread();
  }

  bool Prefetch() override {
    if (count_.load() < max_count) {
      printf("prefetched %d\n", count_.load());
      count_++;
    }
    return true;
  }

  void RunPerSampleCPU(SampleWorkspace* ws) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  inline int MaxNumInput() const override { return 0; }
  inline int MinNumInput() const override { return 0; }
  inline int MaxNumOutput() const override { return 1; }
  inline int MinNumOutput() const override { return 1; }

 private:
  std::atomic<int> count_;
  const int max_count = 100;
};

template <typename Backend>
class PrefetchedDataReaderTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(PrefetchedDataReaderTest, TestTypes);

TYPED_TEST(PrefetchedDataReaderTest, test) {
  shared_ptr<DummyDataReader<TypeParam>> op(
      new DummyDataReader<TypeParam>(
        OpSpec("")
        .AddArg("num_threads", 4)
        .AddArg("batch_size", 128)));

  op->Run((SampleWorkspace*)nullptr);

  return;
}

};  // namespace ndll
