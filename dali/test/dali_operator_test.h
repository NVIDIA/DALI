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

#include <dali/pipeline/pipeline.h>
#include <dali/test/graph_descr.h>
#include <dali/test/tensor_list_wrapper.h>
#include <dali/test/argument_key.h>
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <map>

namespace dali {
namespace testing {

using Arguments = std::map<ArgumentKey, double>;  // TODO(mszolucha) some generalization boost::any?

class DaliOperatorTest : public ::testing::Test, public ::testing::WithParamInterface<Arguments> {
 public:
  using Verify = std::function<void(TensorListWrapper, TensorListWrapper, Arguments)>;

 protected:
  void RunTest(const TensorListWrapper &input, const TensorListWrapper &output,
               const Arguments &operator_arguments, const Verify &verify) {
  }


  void RunTest(const std::vector<TensorListWrapper> &inputs,
               const std::vector<TensorListWrapper> &outputs,
               const Arguments &operator_arguments, const std::vector<Verify> &verify) {
  }


 private:
  virtual GraphDescr GenerateOperatorsGraph() const noexcept = 0;


  void SetUp() final {
  }


  void TearDown() final {
  }
};

}  // namespace testing

}  // namespace dali

#endif  // DALI_TEST_DALI_OPERATOR_TEST_H_
