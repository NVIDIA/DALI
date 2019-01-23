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
class SequenceRearrangeTest : public testing::DaliOperatorTest {
  testing::GraphDescr GenerateOperatorGraph() const noexcept override {
    return {"SequenceRearrange"};
  }

 public:
  SequenceRearrangeTest() : DaliOperatorTest(1, 1) {}
  TensorList<CPUBackend> getInput() {
    return {};
    std::vector<Index> seq_shape{20, 4, 2, 2};
    // repeat seq_shape for whole batch
    TensorList<CPUBackend> tl;
    //set type to int
    //Resize to shape
    //fil with consecutive numbers for each frame
  }

};

std::vector<testing::Arguments> reorders = {
      {{"new_order", std::vector<int>{0, 1, 2, 3, 4}}},
};
/*

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

*/
// TODO unify Verify
void Verify(const testing::TensorListWrapper& input, const testing::TensorListWrapper& output, const testing::Arguments& args) {
  auto in = input.get<CPUBackend>();
  auto out = output.get<CPUBackend>();
  // for each element:
  // ASSERT_EQ(Product(in.shape()), Product(out.shape()));
  // EXPECT_EQ(out.shape(), args[new_shape]);
  // compare in.data() and out.data() elementwise
  EXPECT_TRUE(true);
}


TEST_P(SequenceRearrangeTest, GoodRearranges) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  testing::TensorListWrapper tlin(getInput()); // TODO(klecki) this is not fun, it was supposed to happend automatically
  this->RunTest<CPUBackend>(tlin, tlout, args, Verify());
}


INSTANTIATE_TEST_CASE_P(GoodRearranges, SequenceRearrangeTest, ::testing::ValuesIn(reorders));

}  // namespace dali
