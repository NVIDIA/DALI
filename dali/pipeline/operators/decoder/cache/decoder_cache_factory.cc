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

#include <memory>
#include "dali/pipeline/operators/decoder/cache/decoder_cache_factory.h"
#include "dali/pipeline/operators/decoder/cache/decoder_cache_blob.h"
#include "dali/pipeline/operators/decoder/cache/decoder_cache_largest_only.h"

namespace dali {

std::shared_ptr<DecoderCache> DecoderCacheFactory::Get(int device_id,
                                                       const std::string& cache_policy,
                                                       std::size_t cache_size,
                                                       bool cache_debug,
                                                       std::size_t cache_threshold) {
  std::lock_guard<std::mutex> lock(mutex_);
  const CacheParams params{cache_policy, cache_size, cache_debug, cache_threshold};
  auto &instance = caches_[device_id];
  auto cache = instance.cache.lock();
  if (!cache) {
    if (cache_policy == "threshold") {
      cache.reset(new DecoderCacheBlob(cache_size, cache_threshold, cache_debug));
    } else if (cache_policy == "largest") {
      cache.reset(new DecoderCacheLargestOnly(cache_size, cache_debug));
    } else {
      DALI_FAIL("unexpected cache policy `" + cache_policy + "`");
    }
    caches_[device_id] = {cache, params};
    return cache;
  }
  DALI_ENFORCE(instance.params == params,
     "Cache for device " + std::to_string(device_id)
     + " was already initialized with other parameters");

  cache = instance.cache.lock();
  return cache;
}

std::shared_ptr<DecoderCache> DecoderCacheFactory::Get(int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  CheckWeakPtr(device_id);
  auto it = caches_.find(device_id);
  DALI_ENFORCE(it != caches_.end(), "Cache does not exist");
  return it->second.cache.lock();
}

bool DecoderCacheFactory::IsInitialized(int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  CheckWeakPtr(device_id);
  return caches_.find(device_id) != caches_.end();
}

void DecoderCacheFactory::CheckWeakPtr(int device_id) {
  auto it = caches_.find(device_id);
  if (it != caches_.end() && it->second.cache.expired())
    caches_.erase(it);
}

}  // namespace dali
