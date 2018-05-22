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
#include "ndll/pipeline/workspace/sample_workspace.h"
#include "ndll/test/ndll_test.h"
#include "ndll/pipeline/operators/reader/reader_op.h"

namespace ndll {

class DummyLoader : public Loader<CPUBackend> {
 public:
  explicit DummyLoader(const OpSpec& spec) :
    Loader<CPUBackend>(spec) {}

  void ReadSample(Tensor<CPUBackend> *t) override {
    t->Resize({1});
    t->mutable_data<uint8>();
    return;
  }

  Index Size() override {
    return 1;
  }
};
class DummyDataReader : public DataReader<CPUBackend> {
 public:
  explicit DummyDataReader(const OpSpec &spec)
      : DataReader<CPUBackend>(spec),
        count_(0) {
    loader_.reset(new DummyLoader(spec));
  }

  ~DummyDataReader() {
    DataReader<CPUBackend>::StopPrefetchThread();
  }
  /*
  using DataReader<CPUBackend>::prefetched_batch_;
  bool Prefetch() override {
    for (int i = 0; i < Operator::batch_size_; ++i) {
      printf("new tensor %d\n", i);
      auto *t = loader_->ReadOne();
      prefetched_batch_.push_back(t);
    }
    return true;
  }
  */

  void RunImpl(SampleWorkspace* ws, int idx) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    ws->Output<CPUBackend>(0)->Copy(*prefetched_batch_[ws->data_idx()], 0);
  }

 private:
  std::atomic<int> count_;
  const int max_count = 100;
};

NDLL_REGISTER_OPERATOR(DummyDataReader, DummyDataReader, CPU);

NDLL_SCHEMA(DummyDataReader)
  .DocStr("Dummy")
  .OutputFn([](const OpSpec& spec) { return 1; })
  .NumInput(0)
  .NumOutput(1)
  LOADER_SCHEMA_ARGS;

template <typename Backend>
class ReaderTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(ReaderTest, TestTypes);

TYPED_TEST(ReaderTest, SimpleTest) {
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
