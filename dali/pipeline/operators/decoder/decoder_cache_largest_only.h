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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_DECODER_CACHE_LARGEST_ONLY_H_
#define DALI_PIPELINE_OPERATORS_DECODER_DECODER_CACHE_LARGEST_ONLY_H_

#include "dali/pipeline/operators/decoder/decoder_cache_blob.h"
#include <unordered_set>
#include <queue>
#include <utility>
#include <stack>
#include <vector>
#include <functional>
#include "dali/pipeline/data/tensor_list.h"  // needed for Dims
#include "dali/error_handling.h"

namespace dali {

class DecoderCacheLargestOnly : public DecoderCacheBlob{
 public:
    inline DecoderCacheLargestOnly(std::size_t cache_size,
                                   std::size_t cache_threshold,
                                   bool stats_enabled = true )
        : DecoderCacheBlob(cache_size, cache_threshold, stats_enabled) {
    }

    inline ~DecoderCacheLargestOnly() override {
    }

    DISABLE_COPY_MOVE_ASSIGN(DecoderCacheLargestOnly);

    inline void Add(const ImageKey& image_key,
                    const uint8_t *data, std::size_t data_size,
                    const Dims& data_shape,
                    cudaStream_t stream = 0) override {

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

                // if there is enough space, store the image as one of biggest
                if (biggest_images_total_ + data_size <= cache_size_) {
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
            DecoderCacheBlob::Add(image_key, data, data_size, data_shape, stream);
        }
    }

 private:
    using QueueElement = std::pair<std::size_t, ImageKey>;
    std::priority_queue<
        QueueElement,
        std::vector<QueueElement>,
        std::greater<QueueElement>> biggest_images_;
    std::unordered_set<ImageKey> images_;
    bool start_caching_ = false;
    std::size_t biggest_images_total_ = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_DECODER_CACHE_LARGEST_ONLY_H_
