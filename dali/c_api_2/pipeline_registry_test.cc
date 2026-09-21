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
#include <atomic>
#include <string>
#include "dali/c_api_2/error_handling.h"
#include "dali/c_api_2/pipeline_registry.h"
#include "dali/c_api_2/test_utils.h"
#include "dali/c_api.h"
#include "dali/dali.h"
#include "dali/pipeline/pipeline.h"

namespace dali::c_api::test {

namespace {

std::string SerializeSimplePipeline() {
  Pipeline p(MakePipelineParams(1, 1, CPU_ONLY_DEVICE_ID));
  p.AddExternalInput("dummy");
  p.SetOutputDescs({ {"dummy", "cpu"} });
  return p.SerializeToProtobuf();
}

}  // namespace

TEST(CAPI2_PipelineRegistryTest, RegisterUnregister) {
  auto &registry = PipelineRegistry::instance();
  size_t base = registry.Count();

  static std::atomic<int> destroyed;
  destroyed = 0;
  int a = 0, b = 0;
  auto deleter = [](void *) { destroyed++; };

  registry.Register(&a, deleter);
  registry.Register(&b, deleter);
  EXPECT_EQ(registry.Count(), base + 2);

  registry.Register(&a, deleter);  // re-registering the same pointer doesn't duplicate it
  EXPECT_EQ(registry.Count(), base + 2);

  registry.Unregister(&a);
  EXPECT_EQ(registry.Count(), base + 1);
  registry.Unregister(&a);  // unregistering twice is a no-op
  EXPECT_EQ(registry.Count(), base + 1);
  registry.Unregister(&b);
  EXPECT_EQ(registry.Count(), base);
  EXPECT_EQ(destroyed, 0);
}

TEST(CAPI2_PipelineRegistryTest, DestroyAll) {
  auto &registry = PipelineRegistry::instance();
  ASSERT_EQ(registry.Count(), 0u);

  static std::atomic<int> destroyed;
  destroyed = 0;
  int a = 0, b = 0;
  auto deleter = [](void *) { destroyed++; };

  registry.Register(&a, deleter);
  registry.Register(&b, deleter);
  EXPECT_EQ(registry.DestroyAll(), 2u);
  EXPECT_EQ(destroyed, 2);
  EXPECT_EQ(registry.Count(), 0u);
  EXPECT_EQ(registry.DestroyAll(), 0u);
}

TEST(CAPI2_PipelineRegistryTest, Destroy) {
  auto &registry = PipelineRegistry::instance();
  ASSERT_EQ(registry.Count(), 0u);

  static std::atomic<int> destroyed;
  destroyed = 0;
  int a = 0;
  auto deleter = [](void *) { destroyed++; };

  registry.Register(&a, deleter);
  EXPECT_TRUE(registry.Destroy(&a));
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(registry.Count(), 0u);
  EXPECT_FALSE(registry.Destroy(&a));  // already claimed - the deleter must not run again
  EXPECT_EQ(destroyed, 1);
}

TEST(CAPI2_PipelineRegistryTest, CloseRejectsRegistration) {
  auto &registry = PipelineRegistry::instance();
  ASSERT_EQ(registry.Count(), 0u);
  ASSERT_FALSE(registry.IsClosed());

  static std::atomic<int> destroyed;
  destroyed = 0;
  int a = 0, b = 0;
  auto deleter = [](void *) { destroyed++; };

  registry.Register(&a, deleter);
  registry.Close();
  EXPECT_TRUE(registry.IsClosed());
  EXPECT_THROW(registry.Register(&b, deleter), Unloading);
  EXPECT_EQ(registry.Count(), 1u);  // closing doesn't destroy the pipelines by itself
  EXPECT_EQ(registry.DestroyAll(), 1u);
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(registry.Count(), 0u);

  registry.Open();
  EXPECT_FALSE(registry.IsClosed());
  registry.Register(&b, deleter);
  EXPECT_EQ(registry.Count(), 1u);
  registry.Unregister(&b);
}

TEST(CAPI2_PipelineRegistryTest, CreateAfterCloseFails) {
  auto &registry = PipelineRegistry::instance();
  ASSERT_EQ(registry.Count(), 0u);
  CHECK_DALI(daliInit());  // make sure the lazy initialization doesn't reopen the registry

  registry.Close();
  daliPipelineParams_t params{};
  daliPipeline_h h = nullptr;
  EXPECT_EQ(daliPipelineCreate(&h, &params), DALI_ERROR_UNLOADING);
  EXPECT_EQ(h, nullptr);
  EXPECT_EQ(registry.Count(), 0u);
  registry.Open();

  CHECK_DALI(daliPipelineCreate(&h, &params));
  EXPECT_EQ(registry.Count(), 1u);
  CHECK_DALI(daliPipelineDestroy(h));
  EXPECT_EQ(registry.Count(), 0u);
}

TEST(CAPI2_PipelineRegistryTest, TracksCApi2Pipelines) {
  ASSERT_EQ(GetOutstandingPipelineCount(), 0u);

  daliPipelineParams_t params{};
  daliPipeline_h h1 = nullptr, h2 = nullptr;
  CHECK_DALI(daliPipelineCreate(&h1, &params));
  EXPECT_EQ(GetOutstandingPipelineCount(), 1u);

  auto proto = SerializeSimplePipeline();
  CHECK_DALI(daliPipelineDeserialize(&h2, proto.c_str(), proto.length(), &params));
  EXPECT_EQ(GetOutstandingPipelineCount(), 2u);

  CHECK_DALI(daliPipelineDestroy(h1));
  EXPECT_EQ(GetOutstandingPipelineCount(), 1u);
  CHECK_DALI(daliPipelineDestroy(h2));
  EXPECT_EQ(GetOutstandingPipelineCount(), 0u);
}

TEST(CAPI2_PipelineRegistryTest, DestroysLeakedCApi2Pipelines) {
  ASSERT_EQ(GetOutstandingPipelineCount(), 0u);

  auto proto = SerializeSimplePipeline();
  daliPipelineParams_t params{};
  daliPipeline_h h = nullptr;
  CHECK_DALI(daliPipelineDeserialize(&h, proto.c_str(), proto.length(), &params));
  CHECK_DALI(daliPipelineBuild(h));
  EXPECT_EQ(GetOutstandingPipelineCount(), 1u);

  // The pipeline is intentionally not destroyed with daliPipelineDestroy
  EXPECT_EQ(DestroyOutstandingPipelines(), 1u);
  EXPECT_EQ(GetOutstandingPipelineCount(), 0u);
}

TEST(CAPI2_PipelineRegistryTest, TracksLegacyCApiPipelines) {
  ASSERT_EQ(GetOutstandingPipelineCount(), 0u);

  auto proto = SerializeSimplePipeline();
  daliPipelineHandle h;
  daliDeserializeDefault(&h, proto.c_str(), proto.length());
  EXPECT_EQ(GetOutstandingPipelineCount(), 1u);
  daliDeletePipeline(&h);
  EXPECT_EQ(GetOutstandingPipelineCount(), 0u);

  daliDeserializeDefault(&h, proto.c_str(), proto.length());
  EXPECT_EQ(GetOutstandingPipelineCount(), 1u);
  // The pipeline is intentionally not destroyed with daliDeletePipeline
  EXPECT_EQ(DestroyOutstandingPipelines(), 1u);
  EXPECT_EQ(GetOutstandingPipelineCount(), 0u);
}

}  // namespace dali::c_api::test
