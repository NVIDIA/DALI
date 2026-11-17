// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_DALI_PIPELINE_LEAK_TEST_H_
#define DALI_TEST_DALI_PIPELINE_LEAK_TEST_H_

#include <gtest/gtest.h>
#include "dali/c_api_2/pipeline_registry.h"

namespace dali {

/**
 * @brief GoogleTest Event Listener that checks after every test that all pipeline instances
 * created through the C APIs were destroyed.
 *
 * Leaked pipelines are destroyed after reporting the failure, so that a single leaking test
 * doesn't cause all subsequent tests to fail as well.
 */
class PipelineLeakEventListener : public ::testing::EmptyTestEventListener {
  void OnTestEnd(const ::testing::TestInfo &test_info) override {
    size_t outstanding = c_api::GetOutstandingPipelineCount();
    EXPECT_EQ(outstanding, 0u) << outstanding << " pipeline instance(s) created with the C API "
                                  "were not destroyed by the test.";
    if (outstanding)
      c_api::DestroyOutstandingPipelines();
  }
};

}  // namespace dali

#endif  // DALI_TEST_DALI_PIPELINE_LEAK_TEST_H_
