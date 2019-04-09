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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/pipeline/operators/decoder/cache/image_cache_largest.h"
#include "dali/kernels/common/copy.h"

namespace dali {
namespace testing {

struct ImageCacheLargestTest : public ::testing::Test {
  ImageCacheLargestTest() {}

  void SetUp() override { SetUpImpl((1 << 9)); }

  void SetUpImpl(std::size_t cache_size) {
    cache_.reset(new ImageCacheLargest(cache_size, false));

    for (std::size_t i = 0; i <= 10; i++) {
      data_.push_back({std::to_string(i), std::vector<uint8_t>(i, i % 256)});
    }
  }

  void AddImage(std::size_t i) {
    cache_->Add(data_[i].first, &data_[i].second[0],
                {static_cast<int64_t>(data_[i].second.size()), 1, 1}, 0);
  }

  bool IsCached(std::size_t i) { return cache_->IsCached(data_[i].first); }

  std::unique_ptr<ImageCacheLargest> cache_;

  std::vector<std::pair<std::string, std::vector<uint8_t>>> data_;
};

TEST_F(ImageCacheLargestTest, FirstRoundNoCache) {
  AddImage(1);
  AddImage(2);
  AddImage(3);
  AddImage(4);
  EXPECT_FALSE(IsCached(1));
  EXPECT_FALSE(IsCached(2));
  EXPECT_FALSE(IsCached(3));
  EXPECT_FALSE(IsCached(4));
}

TEST_F(ImageCacheLargestTest, SecondRoundCache) {
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

TEST_F(ImageCacheLargestTest, OnlySpaceForTheLast) {
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

TEST_F(ImageCacheLargestTest, OnlySpaceForTheOneBeforeTheLast) {
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

TEST_F(ImageCacheLargestTest, OnlySpaceFor4Plus2ButNot3) {
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

TEST_F(ImageCacheLargestTest, ReadWorks) {
  SetUpImpl(4);

  AddImage(4);
  EXPECT_FALSE(IsCached(4));
  AddImage(4);
  EXPECT_TRUE(IsCached(4));

  std::vector<uint8_t> dst(4, 0x00);
  cache_->Read("4", &dst[0], 0);
  cudaStreamSynchronize(0);
  EXPECT_EQ(data_[4].second, dst);
}

TEST_F(ImageCacheLargestTest, GetWorks) {
  SetUpImpl(4);

  AddImage(4);
  EXPECT_FALSE(IsCached(4));
  AddImage(4);
  EXPECT_TRUE(IsCached(4));

  std::vector<uint8_t> dst(4, 0x00);
  auto dev = cache_->Get("4");
  kernels::TensorView<kernels::StorageCPU, uint8_t, 3> host(dst.data(), dev.shape);
  kernels::copy(host, dev);
  cudaStreamSynchronize(0);
  EXPECT_EQ(data_[4].second, dst);
}

TEST_F(ImageCacheLargestTest, AllocateMoreThan2000MB) {
  std::size_t one_mb = 1024 * 1024;
  std::size_t size = 3l * 1024 * one_mb;
  SetUpImpl(size);
  std::vector<uint8_t> data_1MB(one_mb, 0xFF);
  std::size_t N = size / one_mb;

  // First observe
  for (std::size_t i = 0; i < N + 10; i++) {
    cache_->Add(std::to_string(i) + "_mb", &data_1MB[0],
                {static_cast<Index>(one_mb), 1, 1}, 0);
  }

  // Now cache
  for (std::size_t i = 0; i < N + 10; i++) {
    cache_->Add(std::to_string(i) + "_mb", &data_1MB[0],
                {static_cast<Index>(one_mb), 1, 1}, 0);
  }

  // Cache is ready here
  for (std::size_t i = 0; i < N; i++) {
    EXPECT_TRUE(cache_->IsCached(std::to_string(i) + "_mb"));
  }
  for (std::size_t i = N; i < N + 10; i++) {
    EXPECT_FALSE(cache_->IsCached(std::to_string(i) + "_mb"));
  }
}

}  // namespace testing
}  // namespace dali
