// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_operator_test.h"
#include "dali/pipeline/operators/sequence/seq_rearrange.h"

namespace dali {
class SeqRearrangeTest : public testing::DaliOperatorTest {
  testing::GraphDescr GenerateOperatorsGraph() const noexcept override {
    return {"SequenceRearrange"};
  }

 public:
  // SeqRearrangeTest() : DaliOperatorTest(1, 1) {}
  // TensorList<CPUBackend> getInput() = 0;
};

/*
std::vector<Arguments> reorders = {
      {{"new_order", std::vector<Index>{120}}},
};

std::vector<Arguments> input_reshape = {
      {{"new_shape", std::vector<Index>{-1}}},
      {{}} // no-arg?
};

std::vector<Arguments> wrong_reshape = {
      {{"new_shape", std::vector<Index>{121}}},
      {{"new_shape", std::vector<Index>{3, 60}}},
      {{"new_shape", std::vector<Index>{-1, 3, 20}}},
      {{"new_shape", std::vector<Index>{2, -3, 4, 5}}},
      {{"new_shape", std::vector<Index>{2, -3, -4, 5}}},
};

// TODO unify Verify
void Verify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  auto in = ToTensorListView(input);
  auto out = ToTensorListView(output);
  // for each element:
  ASSERT_EQ(Product(in.shape()), Product(out.shape()));
  EXPECT_EQ(out.shape(), args[new_shape]);
  // compare in.data() and out.data() elementwise
}

void Verify_2arg(std::vector<TensorListWrapper> inputs, TensorListWrapper output, Arguments args) {
  auto in = ToTensorListView(inputs[0]);
  auto shape = ToTensorListView(inputs[1]);
  auto out = ToTensorListView(output);
  // for each element:
  ASSERT_EQ(Product(in.shape()), Product(out.shape()));
  EXPECT_EQ(out.shape(), shape);
  // compare in.data() and out.data() elementwise
}

TEST_P(ReshapeTest, ContigousInTest) {
  auto args = GetParam();
  TensorListWrapper tlout; // todo, whats with that out?
  this->RunTest<CPUBackend>(getInputContigous(), tlout, args, Verify);
}
*/

}  // namespace dali
