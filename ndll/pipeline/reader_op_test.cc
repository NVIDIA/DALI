// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <thread>

#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/reader_op.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/sample_workspace.h"
#include "ndll/test/ndll_test.h"

#include <cstdio>

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
#if 0
    if (count_.load() < max_count) {
      printf("prefetched %d\n", count_.load());
      count_++;
    }
#else
    for (int i = 0; i < 1024; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (i % 1000 == 0) {
        printf("prefetched %d\n", i);
      }
    }
#endif
    return true;
  }

  void RunPerSampleCPU(SampleWorkspace* ws) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  inline int MaxNumInput() const override { return 0; }
  inline int MinNumInput() const override { return 0; }
  inline int MaxNumOutput() const override { return 1; }
  inline int MinNumOutput() const override { return 1; }

 private:
  std::atomic<int> count_;
  const int max_count = 100;
};

NDLL_REGISTER_CPU_OPERATOR(DummyDataReader, DummyDataReader<CPUBackend>);

template <typename Backend>
class ReaderTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(ReaderTest, TestTypes);

TYPED_TEST(ReaderTest, test) {

  Pipeline pipe(128, 4, 0);

  pipe.AddOperator(
      OpSpec("DummyDataReader")
      .AddOutput("data_out", "cpu"));

  std::vector<std::pair<string, string>> outputs = {{"data_out", "cpu"}};
  pipe.Build(outputs);

  for (int i=0; i < 5; ++i) {
    printf(" ======= ITER %d ======\n", i);
    pipe.RunCPU();
  }

  return;
}

};  // namespace ndll
