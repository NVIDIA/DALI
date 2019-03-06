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

#include "dali/pipeline/operators/decoder/cache/decoder_cache_factory.h"
#include "dali/pipeline/operators/decoder/cache/decoder_cache_blob.h"
#include "dali/pipeline/operators/decoder/cache/decoder_cache_largest_only.h"

namespace dali {

std::shared_ptr<DecoderCache> DecoderCacheFactory::Init(const std::string& cache_policy,
                                                        std::size_t cache_size,
                                                        bool cache_debug,
                                                        std::size_t cache_threshold) {
  std::lock_guard<std::mutex> lock(mutex_);
  DALI_ENFORCE(cache_ == nullptr, "Cache already exists");
  if (cache_policy == "threshold") {
    cache_.reset(new DecoderCacheBlob(cache_size, cache_threshold, cache_debug));
  } else if (cache_policy == "largest") {
    cache_.reset(new DecoderCacheLargestOnly(cache_size, cache_debug));
  } else {
    DALI_FAIL("unexpected cache policy `" + cache_policy + "`");
  }
  return cache_;
}

std::shared_ptr<DecoderCache> DecoderCacheFactory::Get() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_;
}

void DecoderCacheFactory::Destroy() {
  std::lock_guard<std::mutex> lock(mutex_);
  DALI_ENFORCE(cache_ != nullptr, "Cache does not exist");
  cache_.reset();
}

bool DecoderCacheFactory::IsInitialized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_ != nullptr;
}

}  // namespace dali
