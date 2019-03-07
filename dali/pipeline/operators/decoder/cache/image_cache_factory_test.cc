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

#include "dali/pipeline/operators/decoder/cache/image_cache_factory.h"
#include <gtest/gtest.h>
#include <memory>

namespace dali {
namespace testing {

struct ImageCacheFactoryTest : public ::testing::Test {
  ImageCacheFactoryTest() {}

  void SetUp() override {}
};

TEST_F(ImageCacheFactoryTest, Init) {
  auto &factory = ImageCacheFactory::Instance();
  ASSERT_FALSE(factory.IsInitialized(0));
  auto cache = factory.Get(0, "threshold", 1*1024*1024, true, 1024);
  EXPECT_NE(nullptr, cache);
  EXPECT_TRUE(factory.IsInitialized(0));
}

TEST_F(ImageCacheFactoryTest, TwoDifferentDevices) {
  auto &factory = ImageCacheFactory::Instance();
  ASSERT_FALSE(factory.IsInitialized(0));
  ASSERT_FALSE(factory.IsInitialized(1));
  auto cache0 = factory.Get(0, "threshold", 1*1024*1024, true, 1024);
  auto cache1 = factory.Get(1, "largest", 2*1024*1024, true, 0);
  EXPECT_NE(nullptr, cache0);
  EXPECT_NE(nullptr, cache1);
  EXPECT_TRUE(factory.IsInitialized(0));
  EXPECT_TRUE(factory.IsInitialized(1));
}

TEST_F(ImageCacheFactoryTest, Lifetime) {
  auto &factory = ImageCacheFactory::Instance();
  ASSERT_FALSE(factory.IsInitialized(0));
  ASSERT_FALSE(factory.IsInitialized(1));
  {
    auto cache0 = factory.Get(0, "threshold", 1*1024*1024, true, 1024);
    EXPECT_NE(nullptr, cache0);
    EXPECT_TRUE(factory.IsInitialized(0));
  }
  EXPECT_FALSE(factory.IsInitialized(0));
  {
    auto cache0 = factory.Get(0, "largest", 2*1024*1024, true, 0);
    EXPECT_NE(nullptr, cache0);
  }
  EXPECT_FALSE(factory.IsInitialized(0));
}

TEST_F(ImageCacheFactoryTest, Collision) {
  auto &factory = ImageCacheFactory::Instance();
  ASSERT_FALSE(factory.IsInitialized(0));
  ASSERT_FALSE(factory.IsInitialized(1));

  auto cache0 = factory.Get(0, "threshold", 1*1024*1024, true, 1024);
  ASSERT_NE(nullptr, cache0);
  ASSERT_TRUE(factory.IsInitialized(0));
  EXPECT_THROW(
    factory.Get(0, "largest", 2*1024*1024, true, 0),
    std::runtime_error);
  EXPECT_TRUE(factory.IsInitialized(0));
  cache0.reset();
  EXPECT_FALSE(factory.IsInitialized(0));
  cache0 = factory.Get(0, "largest", 2*1024*1024, true, 0);
  EXPECT_TRUE(factory.IsInitialized(0));
}

TEST_F(ImageCacheFactoryTest, GetAndRefCount) {
  auto &factory = ImageCacheFactory::Instance();
  ASSERT_FALSE(factory.IsInitialized(0));
  ASSERT_FALSE(factory.IsInitialized(1));

  auto cache0 = factory.Get(0, "threshold", 1*1024*1024, true, 1024);
  ASSERT_NE(nullptr, cache0);
  ASSERT_TRUE(factory.IsInitialized(0));

  // same so ok
  auto cache01 = factory.Get(0, "threshold", 1*1024*1024, true, 1024);

  // different fails
  EXPECT_THROW(
    factory.Get(0, "threshold", 2*1024*1024, true, 1024),
    std::runtime_error);

  // just pick same
  auto cache02 = factory.Get(0);

  EXPECT_TRUE(factory.IsInitialized(0));
  cache0.reset();
  EXPECT_TRUE(factory.IsInitialized(0));
  cache01.reset();
  EXPECT_TRUE(factory.IsInitialized(0));
  cache02.reset();
  EXPECT_FALSE(factory.IsInitialized(0));

  // now we can allocate again
  auto cache03 = factory.Get(0, "threshold", 2*1024*1024, true, 1024);
}

}  // namespace testing
}  // namespace dali
