// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <gtest/gtest.h>
#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/operators/op_spec.h"
#include "ndll/test/ndll_test.h"
#include "ndll/pipeline/workspace/sample_workspace.h"
#include "ndll/pipeline/operators/reader/parser/parser.h"

namespace ndll {

template <typename Backend>
class IntArrayParser : public Parser {
 public:
  explicit IntArrayParser(const OpSpec& spec)
    : Parser(spec) {}
  void Parse(const uint8_t* data, size_t size, SampleWorkspace* ws) {
    const int *int_data = reinterpret_cast<const int*>(data);

    const int H = int_data[0];
    const int W = int_data[1];
    const int C = int_data[2];

    printf("H: %d, W: %d, C: %d\n", H, W, C);

    Tensor<Backend>* output = ws->template Output<Backend>(0);
    output->Resize({H, W, C});

    int *output_data = output->template mutable_data<int>();

    std::memcpy(output_data, &int_data[3], H*W*C);
  }
};

template <typename Backend>
class ParserTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(ParserTest, TestTypes);

TYPED_TEST(ParserTest, BasicTest) {
  const int H = 64, W = 64, C = 3;
  int *data = new int[3 + H*W*C];
  data[0] = H;
  data[1] = W;
  data[2] = C;

  HostWorkspace workspace;
  SampleWorkspace* ws = new SampleWorkspace;

  workspace.GetSample(ws, 0, 0);

  shared_ptr<Tensor<CPUBackend>> t(new Tensor<CPUBackend>());
  ws->AddOutput(t);

  IntArrayParser<CPUBackend> parser(OpSpec("temp"));
  parser.Parse(reinterpret_cast<uint8_t*>(data), 3 + H*W*C, ws);
}


}  // namespace ndll
