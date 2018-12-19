// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

namespace dali {

namespace testing {

TensorList<CPUBackend> tlcpu;
TensorList<CPUBackend> tlgpu;


class ExampleOperatorTestCase : public DaliOperatorTest {

 protected:
  ExampleOperatorTestCase() {
    tlcpu.Resize({{1}});
    auto ptrcpu = tlcpu.mutable_data<int>();
    ptrcpu[0] = 666;

    tlgpu.Resize({{1}});
    auto ptrgpu = tlgpu.mutable_data<int>();
    ptrgpu[0] = 69;

    *in_ = TensorListWrapper(&tlcpu);
    *out_ = TensorListWrapper(&tlgpu);
  }


 private:
  GraphDescr GenerateOperatorsGraph() const noexcept override {
    GraphDescr graph("ExampleOp");
    return graph;
  }


 protected:
  TensorListWrapper* in_;  // fill it somewhere
  TensorListWrapper* out_;  // fill it somewhere
};

TEST_F(ExampleOperatorTestCase, ExampleTest) {
  TensorListWrapper in(&tlcpu);
  TensorListWrapper out(&tlgpu);
  Arguments args;

  auto ver = [](TensorListWrapper, TensorListWrapper, Arguments) -> void {
      ASSERT_FALSE(true);
  };

  this->RunTest(in, out, args, ver);
}


std::vector<Arguments> args1 = {{{"arg1", 1.}, {"arg2", 2.}, {"arg3", 3.}}};


INSTANTIATE_TEST_CASE_P(FirstOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, ExamplePTest1) {
  auto ver = [](TensorListWrapper, TensorListWrapper, Arguments) -> void {
      ASSERT_FALSE(true);
  };

  this->RunTest(*in_, *out_, GetParam(), ver);
}


INSTANTIATE_TEST_CASE_P(SecondOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, ExampleMultInpTest) {
  auto ver_in1 = [](TensorListWrapper, TensorListWrapper, Arguments) -> void {
      ASSERT_FALSE(true);
  };

  auto ver_in2 = [](TensorListWrapper, TensorListWrapper, Arguments) -> void {
      ASSERT_FALSE(true);
  };

  this->RunTest({*in_, *in_}, {*out_, *out_}, GetParam(), {ver_in1, ver_in2});
}

}  // namespace testing

}  // namespace dali
