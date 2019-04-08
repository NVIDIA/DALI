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
#include "dali/error_handling.h"
#include "dali/pipeline/operators/decoder/cache/cached_decoder_impl.h"
#include "dali/pipeline/operators/decoder/cache/image_cache_factory.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {

// NOTE: has to be in .cc so we can forward-declare ScatterGatherGPU
CachedDecoderImpl::~CachedDecoderImpl() = default;

CachedDecoderImpl::CachedDecoderImpl(const OpSpec& spec)
    : device_id_(spec.GetArgument<int>("device_id")) {
  const std::size_t cache_size_mb = static_cast<std::size_t>(spec.GetArgument<int>("cache_size"));
  const std::size_t cache_size = cache_size_mb * 1024 * 1024;
  const std::size_t cache_threshold =
      static_cast<std::size_t>(spec.GetArgument<int>("cache_threshold"));
  if (cache_size > 0 && cache_size >= cache_threshold) {
    const std::string cache_type = spec.GetArgument<std::string>("cache_type");
    const bool cache_debug = spec.GetArgument<bool>("cache_debug");
    cache_ = ImageCacheFactory::Instance().Get(
      device_id_, cache_type, cache_size, cache_debug, cache_threshold);

    use_batch_copy_kernel_ = spec.GetArgument<bool>("cache_batch_copy");
    auto batch_size = spec.GetArgument<int>("batch_size");
    const size_t kMaxSizePerBlock = 1<<18;  // 256 kB per block
    scatter_gather_.reset(new kernels::ScatterGatherGPU(
      kMaxSizePerBlock, cache_size, batch_size));
  }
}

bool CachedDecoderImpl::CacheLoad(const std::string& file_name,
                                  uint8_t *output_data,
                                  cudaStream_t stream) {
  if (!cache_ || file_name.empty())
    return false;
  return cache_->Read(file_name, output_data, stream);
}


bool CachedDecoderImpl::DeferCacheLoad(const std::string& file_name, uint8_t *output_data) {
  if (!cache_ || file_name.empty())
    return false;
  auto img = cache_->Get(file_name);
  if (!img.data)
    return false;
  scatter_gather_->AddCopy(output_data, img.data, img.num_elements());
  return true;
}

void CachedDecoderImpl::LoadDeferred(cudaStream_t stream) {
  if (scatter_gather_)
    CUDA_CALL((scatter_gather_->Run(stream, true, !use_batch_copy_kernel_), cudaGetLastError()));
}

ImageCache::ImageShape CachedDecoderImpl::CacheImageShape(const std::string& file_name) {
  return cache_ && cache_->IsCached(file_name) ?
    cache_->GetShape(file_name) : ImageCache::ImageShape{};
}

void CachedDecoderImpl::CacheStore(const std::string& file_name, const uint8_t *data,
                                   const ImageCache::ImageShape& data_shape,
                                   cudaStream_t stream) {
  if (!cache_ || file_name.empty() || cache_->IsCached(file_name))
    return;
  cache_->Add(file_name, data, data_shape, stream);
}

DALI_SCHEMA(CachedDecoderAttr)
  .DocStr(R"code(Attributes for cached decoder.)code")
  .AddOptionalArg("cache_size",
      R"code(Total size of the decoder cache in megabytes. When provided, decoded
images bigger than `cache_threshold` will be cached in memory.)code",
      0)
  .AddOptionalArg("cache_threshold",
      R"code(Size threshold (in bytes) for images to be cached.)code",
      0)
  .AddOptionalArg("cache_debug",
      R"code(Print debug information about decoder cache.)code",
      false)
  .AddOptionalArg("cache_batch_copy",
      R"code(If true, multiple images from cache are copied with a single batched copy kernel call;
otherwise, each image is copied using cudaMemcpy unless order in the batch is the same as in the cache)code",
      true)
  .AddOptionalArg("cache_type",
      R"code(Choose cache type:
`threshold`: Caches every image with size bigger than `cache_threshold` until cache is full.
Warm up time for `threshold` policy is 1 epoch.
`largest`: Store largest images that can fit the cache.
Warm up time for `largest` policy is 2 epochs
To take advantage of caching, it is recommended to use the option `stick_to_shard=True` with
the reader operators, to limit the amount of unique images seen by the decoder in a multi node environment)code",
      std::string());

}  // namespace dali
