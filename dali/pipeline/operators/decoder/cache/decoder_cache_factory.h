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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_FACTORY_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_FACTORY_H_

#include <memory>
#include <string>
#include <map>
#include <mutex>
#include "dali/pipeline/operators/decoder/cache/decoder_cache.h"

namespace dali {

class DLL_PUBLIC DecoderCacheFactory {
 public:
  DLL_PUBLIC static inline DecoderCacheFactory& Instance(int device_id) {
    static std::mutex __mutex;
    using DeviceId = int;
    static std::map<DeviceId, DecoderCacheFactory> __cache_factory_map;
    std::unique_lock<std::mutex> lock(__mutex);
    return __cache_factory_map[device_id];
  }

  DLL_PUBLIC std::shared_ptr<DecoderCache> Init(
    const std::string& cache_policy,
    std::size_t cache_size,
    bool cache_debug = false,
    std::size_t cache_threshold = 0);

  DLL_PUBLIC void Destroy();

  DLL_PUBLIC std::shared_ptr<DecoderCache> Get() const;

  DLL_PUBLIC bool IsInitialized() const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<DecoderCache> cache_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_FACTORY_H_
