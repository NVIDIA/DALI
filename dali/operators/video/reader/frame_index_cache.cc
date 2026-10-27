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

#include "dali/operators/video/reader/frame_index_cache.h"
#include <sys/stat.h>
#include <string>

namespace dali {

FrameIndexCache &FrameIndexCache::instance() {
  static FrameIndexCache cache;
  return cache;
}

FrameIndexCache::FileStamp FrameIndexCache::FileStamp::Of(std::string_view filename) {
  struct stat st = {};
  if (stat(std::string(filename).c_str(), &st) != 0)
    return {};
  return {static_cast<int64_t>(st.st_size),
          static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec};
}

FrameIndexCache::IndexPtr FrameIndexCache::Lookup(std::string_view filename,
                                                  const FileStamp &stamp,
                                                  std::shared_future<IndexPtr> &pending) const {
  auto it = cache_.find(filename);
  if (it == cache_.end() || it->second.stamp != stamp)
    return nullptr;  // not cached, or built from another version of the file
  if (it->second.pending.valid()) {
    pending = it->second.pending;
    return nullptr;
  }
  return it->second.index;
}

void FrameIndexCache::Finish(std::string_view filename, uint64_t id, const IndexPtr &index) {
  std::unique_lock guard(mutex_);
  auto it = cache_.find(filename);
  if (it == cache_.end() || it->second.id != id)
    return;  // replaced (the file changed) while it was being built
  if (index) {
    it->second.index = index;
    it->second.pending = {};
  } else {
    cache_.erase(it);
  }
}

FrameIndexCache::IndexPtr FrameIndexCache::Find(std::string_view filename) const {
  std::shared_lock guard(mutex_);
  auto it = cache_.find(filename);
  if (it == cache_.end() || it->second.pending.valid())
    return nullptr;
  return it->second.index;
}

}  // namespace dali
