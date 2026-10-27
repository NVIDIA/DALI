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
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dali/operators/video/reader/frame_index_cache.h"
#include "dali/core/format.h"

namespace dali {

namespace {

FrameIndex MakeIndex(const std::string &filename, int num_frames) {
  FrameIndex index;
  index.filename = filename;
  index.timebase = {1, 25};
  for (int i = 0; i < num_frames; i++)
    index.index.push_back({i, 0, i == 0, i == num_frames - 1});
  return index;
}

}  // namespace

class FrameIndexCacheTest : public ::testing::Test {
 protected:
  // Unique keys per test run: the cache is process-wide and keeps its entries, so a repeated
  // test (--gtest_repeat) must not find those of the previous run
  std::string Key(const std::string &test, int i) const {
    return make_string("frame_index_cache_test/", run_, "/", test, "/", i);
  }

 private:
  static int NextRun() {
    static std::atomic<int> runs{0};
    return runs++;
  }
  int run_ = NextRun();
};

TEST_F(FrameIndexCacheTest, BuildsEachIndexOnceUnderConcurrency) {
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

TEST_F(FrameIndexCacheTest, IndexStaysValidAfterOtherInsertions) {
  auto &cache = FrameIndexCache::instance();
  auto first = cache.GetOrBuild(Key("valid", 0), [&] { return MakeIndex(Key("valid", 0), 10); });
  for (int i = 1; i < 1000; i++)  // forces rehashing of the underlying map
    cache.GetOrBuild(Key("valid", i), [&] { return MakeIndex(Key("valid", i), 1); });
  ASSERT_EQ(first->size(), 10u);
  EXPECT_EQ(first->filename, Key("valid", 0));
  EXPECT_EQ(cache.Find(Key("valid", 0)), first);
}

TEST_F(FrameIndexCacheTest, FailedBuildIsReportedAndRetried) {
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

TEST_F(FrameIndexCacheTest, FindDoesNotBuildOrWait) {
  auto &cache = FrameIndexCache::instance();
  auto key = Key("find", 0);
  EXPECT_EQ(cache.Find(key), nullptr);
  std::atomic<bool> building{false}, release{false};
  FrameIndexCache::IndexPtr built;
  std::thread builder([&]() {
    built = cache.GetOrBuild(key, [&] {
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
  EXPECT_EQ(index, built);
  EXPECT_EQ(index->size(), 2u);
}

TEST_F(FrameIndexCacheTest, IndexIsKeptAfterItsLastUser) {
  auto &cache = FrameIndexCache::instance();
  auto key = Key("kept", 0);
  int builds = 0;
  auto build = [&] {
    builds++;
    return MakeIndex(key, 4);
  };
  auto index = cache.GetOrBuild(key, build);
  EXPECT_EQ(cache.GetOrBuild(key, build), index);
  EXPECT_EQ(builds, 1);
  const FrameIndex *built = index.get();
  index.reset();  // e.g. the reader is destroyed: a later one reuses the index
  index = cache.GetOrBuild(key, build);
  EXPECT_EQ(index.get(), built);
  EXPECT_EQ(cache.Find(key), index);
  EXPECT_EQ(builds, 1);
}

TEST_F(FrameIndexCacheTest, IndexIsRebuiltWhenTheFileChanges) {
  char path[] = "/tmp/frame_index_cache_test_XXXXXX";
  int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  close(fd);
  auto &cache = FrameIndexCache::instance();
  int builds = 0;
  auto build = [&] {
    builds++;
    return MakeIndex(path, builds);
  };
  {
    std::ofstream(path) << "first";
  }
  auto first = cache.GetOrBuild(path, build);
  EXPECT_EQ(cache.GetOrBuild(path, build), first);
  EXPECT_EQ(builds, 1);
  {
    std::ofstream(path, std::ios::app) << ", second";  // another size
  }
  auto second = cache.GetOrBuild(path, build);
  EXPECT_EQ(builds, 2);
  EXPECT_NE(second, first);
  EXPECT_EQ(second->size(), 2u);
  EXPECT_EQ(first->size(), 1u);  // still valid for its users
  EXPECT_EQ(cache.GetOrBuild(path, build), second);
  unlink(path);
}

TEST_F(FrameIndexCacheTest, UnfinishedBuildDoesNotOverwriteItsReplacement) {
  auto &cache = FrameIndexCache::instance();
  const FrameIndexCache::FileStamp old_stamp{1, 1}, new_stamp{2, 2};
  for (bool old_fails : {false, true}) {
    auto key = Key(old_fails ? "replaced_failing" : "replaced", 0);
    std::atomic<bool> building{false}, release{false};
    FrameIndexCache::IndexPtr old_index;
    bool old_threw = false;
    std::thread old_builder([&]() {
      try {
        old_index = cache.GetOrBuild(key, old_stamp, [&] {
          building = true;
          while (!release)
            std::this_thread::yield();
          if (old_fails)
            throw std::runtime_error("cannot index");
          return MakeIndex(key, 1);
        });
      } catch (const std::runtime_error &) {
        old_threw = true;
      }
    });
    while (!building)
      std::this_thread::yield();
    // the file changed while the old version was being indexed
    auto new_index = cache.GetOrBuild(key, new_stamp, [&] { return MakeIndex(key, 2); });
    ASSERT_EQ(new_index->size(), 2u);
    EXPECT_EQ(cache.Find(key), new_index);
    release = true;
    old_builder.join();
    if (old_fails) {
      EXPECT_TRUE(old_threw);
    } else {
      ASSERT_NE(old_index, nullptr);  // its caller still gets the index it built
      EXPECT_EQ(old_index->size(), 1u);
    }
    EXPECT_EQ(cache.Find(key), new_index);
    EXPECT_EQ(cache.GetOrBuild(key, new_stamp, [&]() -> FrameIndex {
      throw std::logic_error("must not be built again");
    }), new_index);
  }
}

}  // namespace dali
