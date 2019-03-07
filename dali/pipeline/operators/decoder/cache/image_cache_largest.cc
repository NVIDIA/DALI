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

#include "dali/pipeline/operators/decoder/cache/image_cache_largest.h"
#include <functional>
#include <stack>
#include "dali/error_handling.h"

namespace dali {

ImageCacheLargest::ImageCacheLargest(std::size_t cache_size, bool stats_enabled)
    : ImageCacheBlob(cache_size, 0, stats_enabled) {}

void ImageCacheLargest::Add(const ImageKey& image_key,
                                  const uint8_t *data,
                                  const ImageShape& data_shape,
                                  cudaStream_t stream) {
  const std::size_t data_size = volume(data_shape);
  std::unique_lock<std::mutex> lock(mutex_);
  // If we haven't started caching
  if (!start_caching_) {
    // if we've already seen this image, start caching
    start_caching_ = (images_.find(image_key) != images_.end());
    // if we decided to start caching, prepare the data
    if (start_caching_) {
      // replace images_ with the biggest_images
      // and clean unnecessary data structures
      total_seen_images_ = images_.size();
      images_.clear();
      while (!biggest_images_.empty()) {
        images_.insert(biggest_images_.top().second);
        biggest_images_.pop();
      }
    } else {
      // mark the image as seen
      images_.insert(image_key);

      const bool data_fits = (biggest_images_total_ + data_size <= cache_size_);
      is_full = is_full || !data_fits;
      // if there is enough space, store the image as one of biggest
      if (data_fits) {
        biggest_images_.push({data_size, image_key});
        biggest_images_total_ += data_size;
      } else if (data_size <= cache_size_) {
        // If full, check whether the current image has higher priority
        std::stack<QueueElement> to_be_discarded;
        while (!biggest_images_.empty()
            && biggest_images_total_ + data_size > cache_size_
            && biggest_images_.top().first < data_size) {
          biggest_images_total_ -= biggest_images_.top().first;
          to_be_discarded.push(std::move(biggest_images_.top()));
          biggest_images_.pop();
        }

        // If we have enough space now, push the new image
        if (biggest_images_total_ + data_size <= cache_size_) {
          biggest_images_.push({data_size, image_key});
          biggest_images_total_ += data_size;
        }

        // If there is extra space, push back the images we took out
        while (!to_be_discarded.empty()) {
          if (biggest_images_total_ + to_be_discarded.top().first <= cache_size_) {
            biggest_images_total_ += to_be_discarded.top().first;
            biggest_images_.push(std::move(to_be_discarded.top()));
          }
          to_be_discarded.pop();
        }
      }
    }
  }
  lock.unlock();

  if (start_caching_ && images_.find(image_key) != images_.end()) {
    ImageCacheBlob::Add(image_key, data, data_shape, stream);
  }
}

}  // namespace dali
