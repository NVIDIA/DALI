// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_DALI_OPERATOR_TEST_H_
#define DALI_TEST_DALI_OPERATOR_TEST_H_

#include <gtest/gtest.h>
#include <vector>
#include "tensor_adapter.h"
#include "operators_graph.h"

namespace dali {

namespace testing {

template<typename InputType, typename OutputType>
class DaliOperatorTest : public ::testing::Test {
  static_assert(std::is_fundamental<InputType>::value, "DaliOperatorTest expects fundamental type as InputType");
  static_assert(std::is_fundamental<OutputType>::value, "DaliOperatorTest expects fundamental type as OutputType");

public:
  using Arguments =  std::map<std::string, int>;  // TODO some generalization (boost::any? tagged union?)

  void RunTest(Arguments operator_arguments, std::vector<TensorAdapter<OutputType>> anticipated_outputs) noexcept {}

private:
  virtual std::vector<TensorAdapter<InputType>> GenerateInputs() const noexcept = 0;

  virtual OperatorsGraph GenerateOperatorsGraph() const noexcept = 0;

  virtual bool Verify(TensorAdapter<OutputType> output, TensorAdapter<OutputType> anticipated_output) const noexcept = 0;

  void SetUp() final {
    inputs_ = GenerateInputs();
  }

  void TearDown() final {}

  std::vector<TensorAdapter<InputType>> inputs_;
};

}  // namespace testing

}  // namespace dali

#endif // DALI_TEST_DALI_OPERATOR_TEST_H_