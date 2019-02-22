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
#include <vector>
#include <utility>
#include <memory>
#include <string>
#include "dali/pipeline/operators/decoder/decoder_cache_largest_only.h"

namespace dali {
namespace testing {

struct DecoderCacheLargestOnlyTest : public ::testing::Test {
  DecoderCacheLargestOnlyTest() {
  }

  void SetUp() override {
    SetUpImpl((1<<9));
  }

  void SetUpImpl(std::size_t cache_size, std::size_t image_size_threshold = 0) {
    cache_.reset(new DecoderCacheLargestOnly(cache_size, image_size_threshold));

    for (std::size_t i = 0; i <= 10; i++) {
      data_.push_back(
        {std::to_string(i), std::vector<uint8_t>(i, i%256)});
    }
  }

  void AddImage(std::size_t i) {
    cache_->Add(
      data_[i].first,
      &data_[i].second[0],
      data_[i].second.size(),
      Dims{static_cast<Index>(data_[i].second.size()), 1, 1});
  }

  bool IsCached(std::size_t i) {
    return cache_->IsCached(data_[i].first);
  }

  std::unique_ptr<DecoderCacheLargestOnly> cache_;

  std::vector<std::pair<std::string, std::vector<uint8_t>>> data_;
};

TEST_F(DecoderCacheLargestOnlyTest, FirstRoundNoCache) {
  AddImage(1);
  AddImage(2);
  AddImage(3);
  AddImage(4);
  EXPECT_FALSE(IsCached(1));
  EXPECT_FALSE(IsCached(2));
  EXPECT_FALSE(IsCached(3));
  EXPECT_FALSE(IsCached(4));
}

TEST_F(DecoderCacheLargestOnlyTest, SecondRoundCache) {
  for (std::size_t i = 0; i < 2; i++) {
    AddImage(1);
    AddImage(2);
    AddImage(3);
    AddImage(4);
  }

  EXPECT_TRUE(IsCached(1));
  EXPECT_TRUE(IsCached(2));
  EXPECT_TRUE(IsCached(3));
  EXPECT_TRUE(IsCached(4));
}

TEST_F(DecoderCacheLargestOnlyTest, OnlySpaceForTheLast) {
  SetUpImpl(4);
  for (std::size_t i = 0; i < 2; i++) {
    AddImage(1);
    AddImage(2);
    AddImage(3);
    AddImage(4);
  }

  EXPECT_FALSE(IsCached(1));
  EXPECT_FALSE(IsCached(2));
  EXPECT_FALSE(IsCached(3));
  EXPECT_TRUE(IsCached(4));
}

TEST_F(DecoderCacheLargestOnlyTest, OnlySpaceForTheOneBeforeTheLast) {
  SetUpImpl(3);
  for (std::size_t i = 0; i < 2; i++) {
    AddImage(1);
    AddImage(2);
    AddImage(3);
    AddImage(4);
  }

  EXPECT_FALSE(IsCached(1));
  EXPECT_FALSE(IsCached(2));
  EXPECT_TRUE(IsCached(3));
  EXPECT_FALSE(IsCached(4));
}

TEST_F(DecoderCacheLargestOnlyTest, OnlySpaceFor4Plus2ButNot3) {
  SetUpImpl(6);
  for (std::size_t i = 0; i < 2; i++) {
    AddImage(1);
    AddImage(2);
    AddImage(3);
    AddImage(4);
  }

  EXPECT_FALSE(IsCached(1));
  EXPECT_TRUE(IsCached(2));
  EXPECT_FALSE(IsCached(3));
  EXPECT_TRUE(IsCached(4));
}

TEST_F(DecoderCacheLargestOnlyTest, CopyDataWorks) {
  SetUpImpl(4);

  AddImage(4);
  EXPECT_FALSE(IsCached(4));
  AddImage(4);
  EXPECT_TRUE(IsCached(4));

  std::vector<uint8_t> dst(4, 0x00);
  cache_->CopyData("4", &dst[0]);
  EXPECT_EQ(data_[4].second, dst);
}

}  // namespace testing
}  // namespace dali
