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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dali/operators/video/reader/frame_index_cache.h"
#include "dali/core/format.h"

namespace dali {

namespace {

// Unique keys per test: the cache is process-wide
std::string Key(const std::string &test, int i) {
  return make_string("frame_index_cache_test/", test, "/", i);
}

FrameIndex MakeIndex(const std::string &filename, int num_frames) {
  FrameIndex index;
  index.filename = filename;
  index.timebase = {1, 25};
  for (int i = 0; i < num_frames; i++)
    index.index.push_back({i, 0, i == 0, i == num_frames - 1});
  return index;
}

}  // namespace

TEST(FrameIndexCacheTest, BuildsEachIndexOnceUnderConcurrency) {
  constexpr int kThreads = 16, kFiles = 64, kRounds = 4;
  auto &cache = FrameIndexCache::instance();
  std::vector<std::atomic<int>> builds(kFiles);
  std::atomic<int> errors{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([&, t]() {
      for (int r = 0; r < kRounds; r++) {
        for (int f = 0; f < kFiles; f++) {
          int file = (f + t * 7) % kFiles;  // threads start on different files
          auto index = cache.GetOrBuild(Key("concurrent", file), [&]() {
            builds[file]++;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            return MakeIndex(Key("concurrent", file), file + 1);
          });
          if (!index || index->filename != Key("concurrent", file) ||
              static_cast<int>(index->size()) != file + 1)
            errors++;
        }
      }
    });
  }
  for (auto &t : threads)
    t.join();
  EXPECT_EQ(errors, 0);
  for (int f = 0; f < kFiles; f++)
    EXPECT_EQ(builds[f], 1) << "file " << f;
}

TEST(FrameIndexCacheTest, IndexStaysValidAfterOtherInsertions) {
  auto &cache = FrameIndexCache::instance();
  auto first = cache.GetOrBuild(Key("valid", 0), [] { return MakeIndex(Key("valid", 0), 10); });
  for (int i = 1; i < 1000; i++)  // forces rehashing of the underlying map
    cache.GetOrBuild(Key("valid", i), [i] { return MakeIndex(Key("valid", i), 1); });
  ASSERT_EQ(first->size(), 10u);
  EXPECT_EQ(first->filename, Key("valid", 0));
  EXPECT_EQ(cache.Find(Key("valid", 0)), first);
}

TEST(FrameIndexCacheTest, FailedBuildIsReportedAndRetried) {
  auto &cache = FrameIndexCache::instance();
  auto key = Key("failure", 0);
  EXPECT_THROW(
      cache.GetOrBuild(key, []() -> FrameIndex { throw std::runtime_error("cannot index"); }),
      std::runtime_error);
  EXPECT_EQ(cache.Find(key), nullptr);
  auto index = cache.GetOrBuild(key, [&] { return MakeIndex(key, 3); });
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->size(), 3u);
}

TEST(FrameIndexCacheTest, FindDoesNotBuildOrWait) {
  auto &cache = FrameIndexCache::instance();
  auto key = Key("find", 0);
  EXPECT_EQ(cache.Find(key), nullptr);
  std::atomic<bool> building{false}, release{false};
  std::thread builder([&]() {
    cache.GetOrBuild(key, [&] {
      building = true;
      while (!release)
        std::this_thread::yield();
      return MakeIndex(key, 2);
    });
  });
  while (!building)
    std::this_thread::yield();
  EXPECT_EQ(cache.Find(key), nullptr);  // still being built
  release = true;
  builder.join();
  auto index = cache.Find(key);
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->size(), 2u);
}

}  // namespace dali
