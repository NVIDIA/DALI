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

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "dali/core/api_helper.h"
#include "dali/operators/video/frames_decoder_base.h"

namespace dali {

/**
 * @brief Process-wide cache of frame indices, keyed by file name.
 *
 * Shared by all video readers in the process and safe to use from several threads at once.
 * Every index is built at most once while it is in use (by the first caller that asks for it),
 * outside the lock, so building the indices of different files can run in parallel. Callers get
 * an immutable, reference-counted index that stays valid regardless of other insertions.
 *
 * The cache does not own the indices: it keeps an index only while a caller (e.g. a reader)
 * holds it, so the index is freed with the last reader that uses it, and a later reader of the
 * file builds it again. An index is also built again when the file's size or modification time
 * changed since it was built.
 */
class DLL_PUBLIC FrameIndexCache {
 public:
  using IndexPtr = std::shared_ptr<const FrameIndex>;

  /**
   * @brief The process-wide instance. It is defined in a .cc file, so that there is one per
   *        process, not one per shared object that includes this header.
   */
  static FrameIndexCache &instance();

  /**
   * @brief Returns the index of `filename`, building it with `build()` if it is not cached.
   *
   * `build` returns a FrameIndex. If another thread is building the same index, this call waits
   * for it. If building fails, the exception is rethrown to every caller waiting for that index
   * and the entry is removed, so that a later call can try again.
   */
  template <typename BuildFn>
  IndexPtr GetOrBuild(std::string_view filename, BuildFn &&build) {
    auto stamp = FileStamp::Of(filename);
    std::shared_future<IndexPtr> pending;
    {
      std::shared_lock guard(mutex_);
      if (auto index = Lookup(filename, stamp, pending))
        return index;
    }
    std::promise<IndexPtr> promise;
    uint64_t id = 0;
    if (!pending.valid()) {
      std::unique_lock guard(mutex_);
      // looked up again: another thread may have added the entry since the shared lock was
      // released
      if (auto index = Lookup(filename, stamp, pending))
        return index;
      if (!pending.valid()) {
        SweepExpired();
        id = ++last_id_;
        pending = promise.get_future().share();
        cache_.insert_or_assign(std::string(filename), Entry{id, pending, {}, stamp});
      }
    }
    if (!id)
      return pending.get();  // built by another thread
    IndexPtr index;
    try {
      index = std::make_shared<const FrameIndex>(build());
    } catch (...) {
      Finish(filename, id, nullptr);
      promise.set_exception(std::current_exception());
      throw;
    }
    Finish(filename, id, index);
    promise.set_value(index);
    return index;
  }

  /**
   * @brief Returns the index of `filename` if it is cached and ready, nullptr otherwise.
   *
   * Never builds and never waits; does not check the file.
   */
  IndexPtr Find(std::string_view filename) const;

 private:
  FrameIndexCache() = default;

  /**
   * @brief A version of a file: its size and modification time (-1 if the file cannot be
   *        checked, e.g. the name is not a path).
   */
  struct FileStamp {
    int64_t size = -1, mtime_ns = -1;
    static FileStamp Of(std::string_view filename);
    bool operator==(const FileStamp &other) const = default;
  };

  struct Entry {
    uint64_t id = 0;                        // tells this entry from a later one of the same file
    std::shared_future<IndexPtr> pending;   // valid while the index is being built
    std::weak_ptr<const FrameIndex> index;  // set once built
    FileStamp stamp;                        // of the file the index is built from
  };

  /**
   * @brief With the lock held (shared or exclusive): the cached index of this version of the file,
   *        or nullptr; `pending` is set if that index is being built.
   */
  IndexPtr Lookup(std::string_view filename, const FileStamp &stamp,
                  std::shared_future<IndexPtr> &pending) const;

  /**
   * @brief Publishes the index built for entry `id` (or removes the entry if `index` is nullptr),
   *        unless the entry was replaced in the meantime.
   */
  void Finish(std::string_view filename, uint64_t id, const IndexPtr &index);

  /**
   * @brief With the exclusive lock held: removes the entries whose indices are no longer used,
   *        once the map has grown to twice its size after the last such sweep.
   */
  void SweepExpired();

  struct Hash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const {
      return std::hash<std::string_view>{}(s);
    }
  };

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, Entry, Hash, std::equal_to<>> cache_;
  uint64_t last_id_ = 0;
  size_t sweep_size_ = 64;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_READER_FRAME_INDEX_CACHE_H_
