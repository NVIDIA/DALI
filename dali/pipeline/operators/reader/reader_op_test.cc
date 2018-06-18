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
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/test/dali_test.h"
#include "dali/pipeline/operators/reader/reader_op.h"

namespace dali {

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
};

DALI_REGISTER_OPERATOR(DummyDataReader, DummyDataReader, CPU);

DALI_SCHEMA(DummyDataReader)
  .DocStr("Dummy")
  .OutputFn([](const OpSpec& spec) { return 1; })
  .NumInput(0)
  .NumOutput(1)
  .AddParent("LoaderBase");

template <typename Backend>
class ReaderTest : public DALITest {
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

};  // namespace dali
