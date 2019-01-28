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

#include <gtest/gtest.h>
#include "dali/test/dali_operator_test.h"
#include "dali/pipeline/operators/util/optical_flow.h"

namespace dali {
namespace testing {

class OpticalFlowTest : public DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr graph("BbFlip");
    return graph;
  }
};

Arguments argums = {};

void verify(const TensorListWrapper & /* single input */,
            const TensorListWrapper & /* single output */,
            const Arguments &) {

}

TEST_F(OpticalFlowTest, StubImplementationTest) {
  std::unique_ptr<TensorList<CPUBackend>>tl(new TensorList<CPUBackend>());
  TensorListWrapper  tlout;
  this->RunTest<CPUBackend>(tl.get(), tlout, argums, verify);

}

}  // namespace testing
}  // namespace dali
