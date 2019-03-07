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

#include "dali/pipeline/operators/decoder/cache/cached_decoder_impl.h"
#include "dali/pipeline/operators/decoder/cache/decoder_cache_factory.h"
#include "dali/error_handling.h"

namespace dali {

CachedDecoderImpl::CachedDecoderImpl(const OpSpec& spec, bool should_init_cache)
  : device_id_(spec.GetArgument<int>("device_id")) {
  if (should_init_cache) {
    const std::size_t cache_size_mb = static_cast<std::size_t>(spec.GetArgument<int>("cache_size"));
    const std::size_t cache_size = cache_size_mb * 1024 * 1024;
    const std::size_t cache_threshold =
        static_cast<std::size_t>(spec.GetArgument<int>("cache_threshold"));
    if (cache_size > 0 && cache_size >= cache_threshold) {
      const std::string cache_type = spec.GetArgument<std::string>("cache_type");
      const bool cache_debug = spec.GetArgument<bool>("cache_debug");
      cache_ = DecoderCacheFactory::Instance().Get(device_id_,
        cache_type, cache_size, cache_debug, cache_threshold);
    }
  } else {
    // OpSpec does not provide cache arguments so we will use this cache as a reader only
    // Assuming the cache was already allocated
    // TODO(janton) : check whether the order of creation of operators is deterministic
    //                If not we need to use std::future or something here
    DALI_ENFORCE(DecoderCacheFactory::Instance().IsInitialized(device_id_),
        "Device cache for device id `" + std::to_string(device_id_)
        + "` was not initialized");
    cache_ = DecoderCacheFactory::Instance().Get(device_id_);
  }
}

bool CachedDecoderImpl::CacheLoad(const std::string& file_name,
                                  const DecoderCache::ImageShape& expected_shape,
                                  uint8_t* output_data,
                                  cudaStream_t stream) {
  if (!cache_ || file_name.empty() || !cache_->IsCached(file_name))
    return false;

  DALI_ENFORCE(cache_->GetShape(file_name) == expected_shape,
    "Output shape does not match the dimensions of the cached image");
  cache_->CopyData(file_name, output_data, stream);
  return true;
}

void CachedDecoderImpl::CacheStore(const std::string& file_name, uint8_t* data,
                                   const DecoderCache::ImageShape& data_shape,
                                   cudaStream_t stream) {
  if (!cache_ || file_name.empty() || cache_->IsCached(file_name))
    return;
  cache_->Add(file_name, data, data_shape, stream);
}

}  // namespace dali