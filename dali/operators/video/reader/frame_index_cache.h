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

#ifndef DALI_OPERATORS_VIDEO_READER_FRAME_INDEX_CACHE_H_
#define DALI_OPERATORS_VIDEO_READER_FRAME_INDEX_CACHE_H_

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "dali/operators/video/frames_decoder_base.h"

namespace dali {

/**
 * @brief Process-wide cache of frame indices, keyed by file name.
 *
 * Shared by all video readers in the process and safe to use from several threads at once.
 * Every index is built at most once (by the first caller that asks for it), outside the lock,
 * so building the indices of different files can run in parallel. Callers get an immutable,
 * reference-counted index that stays valid regardless of other insertions.
 */
class FrameIndexCache {
 public:
  using IndexPtr = std::shared_ptr<const FrameIndex>;

  static FrameIndexCache &instance() {
    static FrameIndexCache cache;
    return cache;
  }

  /**
   * @brief Returns the index of `filename`, building it with `build()` if it is not cached.
   *
   * `build` returns a FrameIndex. If another thread is building the same index, this call waits
   * for it. If building fails, the exception is rethrown to every caller waiting for that index
   * and the entry is removed, so that a later call can try again.
   */
  template <typename BuildFn>
  IndexPtr GetOrBuild(const std::string &filename, BuildFn &&build) {
    std::promise<IndexPtr> promise;
    std::shared_future<IndexPtr> index;
    bool build_here = false;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto it = cache_.find(filename);
      if (it != cache_.end()) {
        index = it->second;
      } else {
        index = promise.get_future().share();
        cache_.emplace(filename, index);
        build_here = true;
      }
    }
    if (build_here) {
      try {
        promise.set_value(std::make_shared<const FrameIndex>(build()));
      } catch (...) {
        {
          std::lock_guard<std::mutex> guard(mutex_);
          cache_.erase(filename);
        }
        promise.set_exception(std::current_exception());
      }
    }
    return index.get();
  }

  /**
   * @brief Returns the index of `filename` if it is cached and ready, nullptr otherwise.
   *
   * Never builds and never waits.
   */
  IndexPtr Find(const std::string &filename) const {
    std::shared_future<IndexPtr> index;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto it = cache_.find(filename);
      if (it == cache_.end())
        return nullptr;
      index = it->second;
    }
    if (index.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      return nullptr;
    try {
      return index.get();
    } catch (...) {
      return nullptr;
    }
  }

 private:
  FrameIndexCache() = default;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_future<IndexPtr>> cache_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_READER_FRAME_INDEX_CACHE_H_
