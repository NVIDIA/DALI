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

#include "dali/pipeline/operators/decoder/cache/image_cache_blob.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace dali {
namespace testing {

const char kKey1[] = "file1.jpg";
const std::vector<uint8_t> kValue1(300, 0xAA);
const ImageCache::ImageShape kShape1{100, 1, 3};

static_assert(sizeof(size_t) > 4, "size_t too small");

struct ImageCacheBlobTest : public ::testing::Test {
  ImageCacheBlobTest() {}

  void SetUp() override { SetUpImpl((1 << 9)); }

  void SetUpImpl(std::size_t cache_size, std::size_t image_size_threshold = 0) {
    cache_.reset(new ImageCacheBlob(cache_size, image_size_threshold, false));
  }

  std::unique_ptr<ImageCacheBlob> cache_;
};

TEST_F(ImageCacheBlobTest, EmptyCache) {
  EXPECT_FALSE(cache_->IsCached(kKey1));
}

TEST_F(ImageCacheBlobTest, Add) {
  EXPECT_FALSE(cache_->IsCached(kKey1));
  cache_->Add(kKey1, &kValue1[0], kShape1, 0);
  EXPECT_TRUE(cache_->IsCached(kKey1));
  std::vector<uint8_t> cachedData(kValue1.size());
  EXPECT_TRUE(cache_->Read(kKey1, &cachedData[0], 0));
  EXPECT_EQ(kValue1, cachedData);
}

TEST_F(ImageCacheBlobTest, ErrorReadNonExistent) {
  std::vector<uint8_t> cachedData(kValue1.size());
  EXPECT_FALSE(cache_->Read(kKey1, &cachedData[0], 0));
}

TEST_F(ImageCacheBlobTest, AddExistingIgnored) {
  cache_->Add(kKey1, &kValue1[0], kShape1, 0);
  cache_->Add(kKey1, &kValue1[0], kShape1, 0);
}

TEST_F(ImageCacheBlobTest, ErrorTooSmallCacheSize) {
  SetUpImpl(kValue1.size() - 1);
  cache_->Add(kKey1, &kValue1[0], kShape1, 0);
  EXPECT_FALSE(cache_->IsCached(kKey1));
}

TEST_F(ImageCacheBlobTest, AllocateMoreThan2000MB) {
  std::size_t one_mb = 1024 * 1024;
  std::size_t size = 3l * 1024 * one_mb;
  SetUpImpl(size);
  std::vector<uint8_t> data_1MB(one_mb, 0xFF);
  std::size_t N = size / one_mb;
  for (std::size_t i = 0; i < N + 10; i++) {
    cache_->Add(std::to_string(i) + "_mb", &data_1MB[0],
                {static_cast<Index>(one_mb), 1, 1}, 0);
  }

  for (std::size_t i = 0; i < N; i++) {
    EXPECT_TRUE(cache_->IsCached(std::to_string(i) + "_mb"));
  }

  for (std::size_t i = N; i < N + 10; i++) {
    EXPECT_FALSE(cache_->IsCached(std::to_string(i) + "_mb"));
  }
}

}  // namespace testing
}  // namespace dali
