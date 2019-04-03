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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_CACHED_DECODER_IMPL_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_CACHED_DECODER_IMPL_H_

#include <cuda_runtime_api.h>
#include <string>
#include <memory>
#include "dali/pipeline/operators/decoder/cache/image_cache.h"
#include "dali/pipeline/operators/op_spec.h"

namespace dali {

class CachedDecoderImpl {
 public:
  /**
   * @params spec: to determine all the cache parameters
   */
  explicit CachedDecoderImpl(const OpSpec& spec);
  virtual ~CachedDecoderImpl() = default;

  virtual bool CacheLoad(
    const std::string& file_name,
    uint8_t *output_data,
    cudaStream_t stream);

  void CacheStore(
    const std::string& file_name,
    uint8_t* data,
    const ImageCache::ImageShape& data_shape,
    cudaStream_t stream);

  ImageCache::ImageShape CacheImageShape(
    const std::string& file_name);

 protected:
  std::shared_ptr<ImageCache> cache_;
  int device_id_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_CACHED_DECODER_IMPL_H_
