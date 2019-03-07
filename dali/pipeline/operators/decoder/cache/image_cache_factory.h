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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_FACTORY_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_FACTORY_H_

#include <memory>
#include <string>
#include <map>
#include <mutex>
#include "dali/pipeline/operators/decoder/cache/image_cache.h"

namespace dali {

class DLL_PUBLIC ImageCacheFactory {
 public:
  DLL_PUBLIC static ImageCacheFactory& Instance() {
    static ImageCacheFactory instance;
    return instance;
  }

  /**
   * @brief Allocate and get cache
   * Will return the previously allocated cached if the parameters
   * are the same.
   * Will fail if the cache was already allocated but with different
   * parameters
   */
  DLL_PUBLIC std::shared_ptr<ImageCache> Get(
    int device_id,
    const std::string& cache_policy,
    std::size_t cache_size,
    bool cache_debug = false,
    std::size_t cache_threshold = 0);

  /**
   * @brief Get the already allocated cache
   * Will fail if cache was not allocated
   */
  DLL_PUBLIC std::shared_ptr<ImageCache> Get(int device_id);

  /**
   * @brief Check whether the cache for a given device id is already initialized
   */
  DLL_PUBLIC bool IsInitialized(int device_id);

 private:
  bool CheckWeakPtr(int device_id);

  mutable std::mutex mutex_;

  struct CacheParams {
    std::string cache_policy;
    std::size_t cache_size;
    bool cache_debug;
    std::size_t cache_threshold;

    inline bool operator==(const CacheParams& oth) const {
      return cache_policy == oth.cache_policy
          && cache_size == oth.cache_size
          && cache_debug == oth.cache_debug
          && cache_threshold == oth.cache_threshold;
    }
  };

  struct CacheInstance {
    std::weak_ptr<ImageCache> cache;
    CacheParams params;
  };
  std::map<int, CacheInstance> caches_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_FACTORY_H_
