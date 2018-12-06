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
#include <dali/pipeline/pipeline.h>
#include "operators_graph.h"
#include "operator_arguments.h"

namespace dali {

namespace testing {

template<typename InputBackend, typename OutputBackend>
class DaliOperatorTest : public ::testing::Test {

 public:
  template<typename Backend>
  using DataType = std::unique_ptr<dali::TensorList<Backend>>;

  /**
   * Verify(output, anticipated_output)
   */
  using Verify = std::function<void(const dali::TensorList<OutputBackend> &,
                                    const dali::TensorList<OutputBackend> &)>;


  void
  RunTest(const std::vector<DataType<InputBackend>> &inputs, OperatorArguments operator_arguments,
          const std::vector<DataType<OutputBackend>> &anticipated_outputs, Verify verify) const {

  }


 private:
  virtual OperatorsGraph GenerateOperatorsGraph() const noexcept = 0;


  void SetUp() final {}


  void TearDown() final {}
};

}  // namespace testing

}  // namespace dali

#endif // DALI_TEST_DALI_OPERATOR_TEST_H_

