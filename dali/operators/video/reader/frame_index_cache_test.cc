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
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/test/dali_test_config.h"

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
  // the cache keeps an index only while it is used: every thread keeps the ones it got
  std::vector<std::vector<FrameIndexCache::IndexPtr>> held(kThreads);
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
          held[t].push_back(index);
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

TEST(FrameIndexCacheTest, HandBuiltIndexIsInDecodingOrder) {
  // Without decoding order, the frames are in decoding order and a seek lands on the frame
  auto index = MakeIndex(Key("hand_built", 0), 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(index[i].decode_idx, -1);
    EXPECT_EQ(index.DecodeIdx(i), i);
    EXPECT_EQ(index.SeekDecodeIdx(i), i);
  }
}

TEST(FrameIndexCacheTest, KeepsDecodingOrder) {
  auto path = testing::dali_extra_path() + "/db/video/vfr/test_1_vp9.mp4";
  FramesDecoderCpu decoder(path);
  ASSERT_TRUE(decoder.IsValid());
  auto &cache = FrameIndexCache::instance();
  auto key = Key("decoding_order", 0);
  auto index = cache.GetOrBuild(key, [&] {
    decoder.BuildIndex();
    return decoder.GetIndex();
  });
  FramesDecoderCpu other(path);
  other.SetIndex(*cache.Find(key));
  auto &built = decoder.GetIndex();
  ASSERT_EQ(index->size(), built.size());
  ASSERT_EQ(other.GetIndex().size(), built.size());
  int seek_targets = 0;
  for (size_t i = 0; i < built.size(); i++) {
    EXPECT_GE(built[i].decode_idx, 0);
    seek_targets += built[i].seek_decode_idx >= 0;
    for (auto *copy : {&(*index)[i], &other.GetIndex()[i]}) {
      EXPECT_EQ(copy->decode_idx, built[i].decode_idx) << i;
      EXPECT_EQ(copy->seek_decode_idx, built[i].seek_decode_idx) << i;
    }
  }
  EXPECT_GT(seek_targets, 1);
}

TEST(FrameIndexCacheTest, FindDoesNotBuildOrWait) {
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

TEST(FrameIndexCacheTest, IndexIsReleasedWithItsLastUser) {
  auto &cache = FrameIndexCache::instance();
  auto key = Key("released", 0);
  int builds = 0;
  auto build = [&] {
    builds++;
    return MakeIndex(key, 4);
  };
  auto index = cache.GetOrBuild(key, build);
  auto other = cache.GetOrBuild(key, build);
  EXPECT_EQ(index, other);
  EXPECT_EQ(builds, 1);
  std::weak_ptr<const FrameIndex> weak = index;
  index.reset();
  other.reset();
  EXPECT_TRUE(weak.expired());  // the cache does not keep it
  EXPECT_EQ(cache.Find(key), nullptr);
  index = cache.GetOrBuild(key, build);
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(builds, 2);
}

TEST(FrameIndexCacheTest, IndexIsRebuiltWhenTheFileChanges) {
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

}  // namespace dali
