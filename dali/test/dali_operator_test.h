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

#ifndef DALI_TEST_DALI_OPERATOR_TEST_H_
#define DALI_TEST_DALI_OPERATOR_TEST_H_

#include <gtest/gtest.h>
#include <vector>
#include <dali/pipeline/pipeline.h>

namespace dali {

namespace testing {

using Arguments = std::map<std::string, double>; // TODO: some generalization. boost::any?

using OpDAG = std::string; // temporary

class DaliOperatorTest : public ::testing::Test, public ::testing::WithParamInterface<Arguments> {
public:

  template<typename Backend>
  using Inputs = TensorList<Backend>; // TODO std::tuple<TensorList<Backend>>. But how to deal with heterogeneous backends?

  template<typename Backend>
  using Outputs = TensorList<Backend>;

  template<typename Backend>
  using Verify = std::function<void(Outputs<Backend>, Outputs<Backend>, Arguments)>;

protected:

  template<typename InputBackend, typename OutputBackend>
  void RunTest(const Inputs<InputBackend> &inputs, const Outputs<OutputBackend> &outputs, Arguments arguments,
               Verify<OutputBackend> verify) const {

  }


private:
  virtual OpDAG GenerateOperatorsGraph() const noexcept = 0;


  void SetUp() final {}


  void TearDown() final {}

};

}  // namespace testing

}  // namespace dali

#endif // DALI_TEST_DALI_OPERATOR_TEST_H_

