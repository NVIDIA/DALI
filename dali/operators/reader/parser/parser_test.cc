// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/test/dali_test.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/operators/reader/parser/parser.h"

namespace dali {

struct IntArrayWrapper {
  int *data;
  size_t size;
};

template <typename Backend>
class IntArrayParser : public Parser<IntArrayWrapper> {
 public:
  explicit IntArrayParser(const OpSpec& spec)
    : Parser<IntArrayWrapper>(spec) {}
  void Parse(const IntArrayWrapper& data, SampleWorkspace* ws) override {
    const int *int_data = data.data;

    const int H = int_data[0];
    const int W = int_data[1];
    const int C = int_data[2];

    printf("H: %d, W: %d, C: %d\n", H, W, C);

    Tensor<Backend>& output = ws->template Output<Backend>(0);
    output.Resize({H, W, C}, DALI_INT32);

    int *output_data = output.template mutable_data<int>();

    std::memcpy(output_data, &int_data[3], H*W*C);
  }
};

template <typename Backend>
class ParserTest : public DALITest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_SUITE(ParserTest, TestTypes);

TYPED_TEST(ParserTest, BasicTest) {
  const int H = 64, W = 64, C = 3;
  std::vector<int> data(3 + H*W*C);
  data[0] = H;
  data[1] = W;
  data[2] = C;

  Workspace workspace;
  SampleWorkspace ws;

  MakeSampleView(ws, workspace, 0, 0);

  shared_ptr<Tensor<CPUBackend>> t(new Tensor<CPUBackend>());
  ws.AddOutput(t.get());

  IntArrayParser<CPUBackend> parser(OpSpec("temp"));
  IntArrayWrapper ia_wrapper = {data.data(), data.size()};
  parser.Parse(ia_wrapper, &ws);
}


}  // namespace dali
