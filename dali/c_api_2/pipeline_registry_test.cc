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
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include "dali/c_api_2/error_handling.h"
#include "dali/c_api_2/pipeline_registry.h"
#include "dali/c_api_2/test_utils.h"
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

daliPipelineParams_t MakeCreateParams() {
  daliPipelineParams_t params{};
  params.max_batch_size_present = true;
  params.max_batch_size = 1;
  params.num_threads_present = true;
  params.num_threads = 1;
  return params;  // no device_id - CPU-only pipeline
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
  auto params = MakeCreateParams();
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

  auto params = MakeCreateParams();
  daliPipeline_h h1 = nullptr, h2 = nullptr;
  CHECK_DALI(daliPipelineCreate(&h1, &params));
  EXPECT_EQ(GetOutstandingPipelineCount(), 1u);

  auto proto = SerializeSimplePipeline();
  daliPipelineParams_t deserialize_params{};
  CHECK_DALI(daliPipelineDeserialize(&h2, proto.c_str(), proto.length(), &deserialize_params));
  EXPECT_EQ(GetOutstandingPipelineCount(), 2u);

  CHECK_DALI(daliPipelineDestroy(h1));
  EXPECT_EQ(GetOutstandingPipelineCount(), 1u);
  CHECK_DALI(daliPipelineDestroy(h2));
  EXPECT_EQ(GetOutstandingPipelineCount(), 0u);
}

TEST(CAPI2_PipelineRegistryTest, DestroyUnknownHandleFails) {
  auto params = MakeCreateParams();
  daliPipeline_h h = nullptr;
  CHECK_DALI(daliPipelineCreate(&h, &params));
  CHECK_DALI(daliPipelineDestroy(h));
  // the handle is no longer tracked - the registry must reject it instead of deleting it twice
  EXPECT_EQ(daliPipelineDestroy(h), DALI_ERROR_INVALID_HANDLE);
  EXPECT_EQ(daliPipelineDestroy(nullptr), DALI_ERROR_INVALID_HANDLE);
}

/** Creates and destroys pipelines on several threads while another thread performs the final
 * shutdown. Every call is either admitted (and then must finish before the registry is torn
 * down) or rejected with DALI_ERROR_UNLOADING; no call may crash or leak a pipeline.
 */
TEST(CAPI2_PipelineRegistryTest, ConcurrentCreateVsShutdown) {
  auto &registry = PipelineRegistry::instance();
  ASSERT_EQ(registry.Count(), 0u);
  const int num_threads = 8;
  const int num_rounds = 5;

  for (int round = 0; round < num_rounds; round++) {
    std::atomic<bool> stop{false};
    std::atomic<int> created{0}, destroyed{0}, destroy_rejected{0}, create_rejected{0};
    std::atomic<int> unexpected{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
      workers.emplace_back([&]() {
        auto params = MakeCreateParams();
        while (!stop) {
          daliPipeline_h h = nullptr;
          auto err = daliPipelineCreate(&h, &params);
          if (err == DALI_ERROR_UNLOADING) {
            create_rejected++;
            if (h != nullptr)
              unexpected++;
            continue;
          }
          if (err != DALI_SUCCESS) {
            unexpected++;
            continue;
          }
          created++;
          err = daliPipelineDestroy(h);
          if (err == DALI_SUCCESS) {
            destroyed++;
          } else if (err == DALI_ERROR_UNLOADING) {
            destroy_rejected++;  // the final shutdown has already destroyed the pipeline
          } else {
            unexpected++;
          }
        }
      });
    }

    // let the workers reach a steady state, then shut down for real
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    while (!registry.IsClosed())
      CHECK_DALI(daliShutdown());  // the lazy initialization may have bumped the counter
    // after the final shutdown, nothing may be tracked and no new creation may succeed
    EXPECT_EQ(registry.Count(), 0u);
    daliPipeline_h h = nullptr;
    auto params = MakeCreateParams();
    EXPECT_EQ(daliPipelineCreate(&h, &params), DALI_ERROR_UNLOADING);
    EXPECT_EQ(h, nullptr);

    stop = true;
    for (auto &w : workers)
      w.join();

    EXPECT_EQ(unexpected, 0);
    EXPECT_EQ(created, destroyed + destroy_rejected);
    EXPECT_EQ(registry.Count(), 0u);

    CHECK_DALI(daliInit());  // reopen for the remaining tests (and the next round)
    EXPECT_FALSE(registry.IsClosed());
  }
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

}  // namespace dali::c_api::test
