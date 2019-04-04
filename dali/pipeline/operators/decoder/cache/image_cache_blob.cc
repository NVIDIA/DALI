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

#include <fstream>
#include <mutex>
#include <unordered_map>
#include "dali/error_handling.h"
#include "dali/kernels/alloc.h"
#include "dali/kernels/span.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operators/decoder/cache/image_cache_blob.h"

namespace dali {

ImageCacheBlob::ImageCacheBlob(std::size_t cache_size,
                               std::size_t image_size_threshold,
                               bool stats_enabled)
    : cache_size_(cache_size)
    , image_size_threshold_(image_size_threshold)
    , stats_enabled_(stats_enabled) {
  DALI_ENFORCE(image_size_threshold <= cache_size_, "Cache size should fit at least one image");

  buffer_ = kernels::memory::alloc_unique<uint8_t>(kernels::AllocType::GPU, cache_size_);
  DALI_ENFORCE(buffer_ != nullptr);
  tail_ = buffer_.get();
  buffer_end_ = buffer_.get() + cache_size_;
  LOG_LINE << "cache size is " << cache_size_ / (1024 * 1024) << " MB" << std::endl;
}

ImageCacheBlob::~ImageCacheBlob() {
  if (stats_enabled_ && images_seen() > 0) print_stats();
}

bool ImageCacheBlob::IsCached(const ImageKey& image_key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_.find(image_key) != cache_.end();
}

const ImageCache::ImageShape& ImageCacheBlob::GetShape(const ImageKey& image_key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = cache_.find(image_key);
  DALI_ENFORCE(it != cache_.end(), "cache entry [" + image_key + "] not found");
  return it->second.shape;
}

bool ImageCacheBlob::Read(const ImageKey& image_key,
                          void* destination_buffer,
                          cudaStream_t stream) const {
  DALI_ENFORCE(!image_key.empty());
  DALI_ENFORCE(destination_buffer != nullptr);
  std::lock_guard<std::mutex> lock(mutex_);
  LOG_LINE << "Read: image_key[" << image_key << "]" << std::endl;
  const auto it = cache_.find(image_key);
  if (it == cache_.end())
    return false;
  const auto& data = it->second;
  DALI_ENFORCE(data.data < tail_);
  const auto n = data.num_elements();
  DALI_ENFORCE(data.data + n <= tail_);
  MemCopy(destination_buffer, data.data, n, stream);
  if (stats_enabled_) stats_[image_key].reads++;
  return true;
}

ImageCache::DecodedImage ImageCacheBlob::Get(const ImageKey& image_key) const {
  DALI_ENFORCE(!image_key.empty());
  std::lock_guard<std::mutex> lock(mutex_);
  LOG_LINE << "Get: image_key[" << image_key << "]" << std::endl;
  const auto it = cache_.find(image_key);
  if (it == cache_.end())
    return {};
  if (stats_enabled_) stats_[image_key].reads++;
  auto ret = it->second;  // make a copy _before_ leaving the mutex
  return ret;
}

void ImageCacheBlob::Add(const ImageKey& image_key, const uint8_t* data,
                         const ImageShape& data_shape, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  const std::size_t data_size = volume(data_shape);
  if (stats_enabled_) stats_[image_key].decodes++;
  if (data_size < image_size_threshold_) return;
  DALI_ENFORCE(!image_key.empty());

  if (cache_.find(image_key) != cache_.end())
    return;

  if (bytes_left() < data_size) {
    LOG_LINE << "WARNING: not enough space in cache. Ignore" << std::endl;
    if (stats_enabled_) is_full = true;
    return;
  }
  MemCopy(tail_, data, data_size, stream);
  cache_[image_key] = {tail_, data_shape};
  tail_ += data_size;

  if (stats_enabled_) stats_[image_key].is_cached = true;
}

void ImageCacheBlob::print_stats() const {
  static std::mutex stats_mutex;
  std::lock_guard<std::mutex> lock(stats_mutex);
  std::size_t images_cached = 0;
  for (auto& elem : stats_)
    if (elem.second.is_cached) images_cached++;
  DALI_ENFORCE(images_cached <= images_seen());
  const char* log_filename = std::getenv("DALI_LOG_FILE");
  std::ofstream log_file;
  if (log_filename) log_file.open(log_filename);
  std::ostream& out = log_filename ? log_file : std::cout;
  out << "#################### CACHE STATS ####################" << std::endl;
  out << "cache_size: " << cache_size_ << std::endl;
  out << "cache_threshold: " << image_size_threshold_ << std::endl;
  out << "is_cache_full: " << static_cast<int>(is_full) << std::endl;
  out << "images_seen: " << images_seen() << std::endl;
  out << "images_cached: " << images_cached << std::endl;
  out << "images_not_cached: " << images_seen() - images_cached << std::endl;
  for (auto& elem : stats_) {
    out << "image[" << elem.first << "] : is_cached[" << static_cast<int>(elem.second.is_cached)
        << "] decodes[" << elem.second.decodes << "] reads[" << elem.second.reads << "]";
    if (elem.second.is_cached) {
      auto shape = GetShape(elem.first);
      out << " shape[" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]";
    }
    out << std::endl;
  }
  out << "#################### END   STATS ####################" << std::endl;
}

}  // namespace dali
