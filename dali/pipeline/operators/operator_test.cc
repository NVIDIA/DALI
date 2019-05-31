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
#include <string>
#include "dali/pipeline/operators/operator.h"

namespace dali {
namespace testing {

OpSpec MakeOpSpec(const std::string& operator_name) {
  return OpSpec(operator_name)
    .AddArg("num_threads", 1)
    .AddArg("batch_size", 32)
    .AddArg("device", "cpu");
}

TEST(InstantiateOperator, ValidOperatorName) {
  ASSERT_NE(nullptr,
    InstantiateOperator(
      MakeOpSpec("Crop")));
}

TEST(InstantiateOperator, InvalidOperatorName) {
  ASSERT_THROW(
    InstantiateOperator(
      MakeOpSpec("DoesNotExist")),
    std::runtime_error);
}

TEST(InstantiateOperator, RunMethodIsAccessible) {
  SampleWorkspace ws;
  ws.set_data_idx(0);
  auto op = InstantiateOperator(MakeOpSpec("Crop"));
  // We just want to test that Run method is visible (exported to the so file)
  // It is expected that the call throws as the worspace is empty
  ASSERT_THROW(
    op->Run(&ws),
    std::runtime_error);
}

}  // namespace testing
}  // namespace dali
