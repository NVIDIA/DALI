// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/sample_workspace.h"
#include "ndll/test/ndll_test.h"
#include "ndll/pipeline/operators/reader/reader_op.h"

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
    for (int i = 0; i < 1000; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (i % 1000 == 0) {
        printf("prefetched %d\n", i);
      }
    }
    return true;
  }

  void RunPerSampleCPU(SampleWorkspace* ws) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

 private:
  std::atomic<int> count_;
  const int max_count = 100;
};

NDLL_REGISTER_CPU_OPERATOR(DummyDataReader, DummyDataReader<CPUBackend>);

NDLL_OPERATOR_SCHEMA(DummyDataReader)
  .DocStr("Dummy")
  .OutputFn([](const OpSpec& spec) { return 1; })
  .NumInput(0)
  .NumOutput(1);

template <typename Backend>
class ReaderTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(ReaderTest, TestTypes);

TYPED_TEST(ReaderTest, test) {
  Pipeline pipe(128, 1, 0);

  pipe.AddOperator(
      OpSpec("DummyDataReader")
      .AddOutput("data_out", "cpu"));

  std::vector<std::pair<string, string>> outputs = {{"data_out", "cpu"}};
  pipe.Build(outputs);

  DeviceWorkspace ws;
  for (int i=0; i < 5; ++i) {
    printf(" ======= ITER %d ======\n", i);
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  }

  return;
}

};  // namespace ndll
