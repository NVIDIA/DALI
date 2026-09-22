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

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include "dali/c_api.h"
#include "dali/c_api_2/pipeline_registry.h"
#include "dali/pipeline/pipeline.h"

namespace dali::test {

namespace {

std::string SerializeSimplePipeline() {
  Pipeline p(MakePipelineParams(1, 1, CPU_ONLY_DEVICE_ID));
  p.AddExternalInput("dummy");
  p.SetOutputDescs({ {"dummy", "cpu"} });
  return p.SerializeToProtobuf();
}

}  // namespace

TEST(CApiPipelineRegistryTest, TracksLegacyCApiPipelines) {
  ASSERT_EQ(c_api::GetOutstandingPipelineCount(), 0u);

  auto proto = SerializeSimplePipeline();
  daliPipelineHandle h;
  daliDeserializeDefault(&h, proto.c_str(), proto.length());
  EXPECT_EQ(c_api::GetOutstandingPipelineCount(), 1u);
  daliDeletePipeline(&h);
  EXPECT_EQ(c_api::GetOutstandingPipelineCount(), 0u);
  // the handle is no longer tracked - it must be rejected instead of being deleted twice
  EXPECT_THROW(daliDeletePipeline(&h), std::invalid_argument);

  daliDeserializeDefault(&h, proto.c_str(), proto.length());
  EXPECT_EQ(c_api::GetOutstandingPipelineCount(), 1u);
  // The pipeline is intentionally not destroyed with daliDeletePipeline
  EXPECT_EQ(c_api::DestroyOutstandingPipelines(), 1u);
  EXPECT_EQ(c_api::GetOutstandingPipelineCount(), 0u);
}

TEST(CApiPipelineRegistryTest, CreateAfterCloseThrows) {
  auto &registry = c_api::PipelineRegistry::instance();
  ASSERT_EQ(registry.Count(), 0u);
  auto proto = SerializeSimplePipeline();
  daliPipelineHandle h = nullptr;

  registry.Close();
  EXPECT_THROW(daliDeserializeDefault(&h, proto.c_str(), proto.length()), std::runtime_error);
  EXPECT_EQ(h, nullptr);
  EXPECT_EQ(registry.Count(), 0u);
  registry.Open();

  daliDeserializeDefault(&h, proto.c_str(), proto.length());
  EXPECT_EQ(registry.Count(), 1u);
  daliDeletePipeline(&h);
  EXPECT_EQ(registry.Count(), 0u);
}

}  // namespace dali::test
