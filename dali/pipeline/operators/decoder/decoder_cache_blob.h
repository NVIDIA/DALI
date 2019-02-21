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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_DECODER_CACHE_BLOB_H_
#define DALI_PIPELINE_OPERATORS_DECODER_DECODER_CACHE_BLOB_H_

#include <string>
#include <mutex>
#include <unordered_map>
#include "dali/pipeline/data/tensor_list.h"  // needed for Dims
#include "dali/kernels/span.h"
#include "dali/error_handling.h"

namespace dali {

class DecoderCacheBlob {
 public:
    using ImageKey = std::string;

    inline DecoderCacheBlob(std::size_t cache_size,
                            std::size_t image_size_threshold,
                            bool stats_enabled = true )
        : cache_size_(cache_size)
        , image_size_threshold_(image_size_threshold)
        , stats_enabled_(stats_enabled) {
        DALI_ENFORCE(image_size_threshold <= cache_size_,
            "Cache size should fit at least one image");
    }

    inline ~DecoderCacheBlob() {
        if (buffer_)
            GPUBackend::Delete(buffer_, 0, false);
        if (stats_enabled_)
            print_stats();
    }

    DISABLE_COPY_MOVE_ASSIGN(DecoderCacheBlob);

    inline bool IsCached(const ImageKey& image_key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.find(image_key) != cache_.end();
    }

    inline const Dims& GetShape(const ImageKey& image_key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = cache_.find(image_key);
        DALI_ENFORCE(it != cache_.end(),
            "cache entry [" + image_key + "] not found");
        return it->second.dims;
    }

    inline void CopyData(const ImageKey& image_key,
                         void* destination_buffer,
                         cudaStream_t stream = 0) const {
        std::lock_guard<std::mutex> lock(mutex_);
        LOG_LINE << "CopyData: image_key[" << image_key << "]" << std::endl;
        DALI_ENFORCE(!image_key.empty());
        DALI_ENFORCE(destination_buffer != nullptr);
        const auto it = cache_.find(image_key);
        DALI_ENFORCE(it != cache_.end(),
            "cache entry [" + image_key + "] not found");
        const auto &data = it->second.data;
        assert(data.data() < tail_);
        assert(data.data() + data.size() <= tail_);
        MemCopy(destination_buffer,
                data.data(),
                data.size(),
                stream);

        if (stats_enabled_)
            stats_[image_key].reads++;
    }

    inline void Add(const ImageKey& image_key,
                    const uint8_t *data, std::size_t data_size,
                    const Dims& data_shape,
                    cudaStream_t stream = 0) {
        if (stats_enabled_)
            stats_[image_key].decodes++;

        if (data_size < image_size_threshold_)
            return;

        if (IsCached(image_key))
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        LOG_LINE << "Add: image_key[" << image_key << "]" << std::endl;
        DALI_ENFORCE(!image_key.empty());

        if (buffer_ == nullptr)
            AllocateBuffer();

        if (bytes_left() < data_size) {
            LOG_LINE << "WARNING: not enough space in cache. Ignore" << std::endl;
            if (stats_enabled_)
                is_full = true;
            return;
        }
        MemCopy(tail_, data, data_size, stream);
        cache_[image_key] = {
            {tail_, static_cast<int64>(data_size)},
            data_shape };
        tail_ += data_size;

        if (stats_enabled_)
            stats_[image_key].is_cached = true;
    }

 private:
    inline void print_stats() const {
        static std::mutex stats_mutex;
        std::lock_guard<std::mutex> lock(stats_mutex);
        std::size_t images_cached = 0;
        std::size_t images_not_cached = 0;
        for (auto &elem : stats_) {
            if (elem.second.is_cached)
                images_cached++;
            else
                images_not_cached++;
        }
        std::cout << "#################### CACHE STATS ####################" << std::endl;
        std::cout << "cache_size: " << cache_size_ << std::endl;
        std::cout << "cache_threshold: " << image_size_threshold_ << std::endl;
        std::cout << "is_cache_full: " << static_cast<int>(is_full) << std::endl;
        std::cout << "images_seen: " << images_cached + images_not_cached << std::endl;
        std::cout << "images_cached: " << images_cached << std::endl;
        std::cout << "images_not_cached: " << images_not_cached << std::endl;
        for (auto &elem : stats_) {
            std::cout << "image[" << elem.first
                    << "] : is_cached[" << static_cast<int>(elem.second.is_cached)
                    << "] decodes[" << elem.second.decodes
                    << "] reads[" << elem.second.reads << "]";
            if (elem.second.is_cached) {
                Dims dims = GetShape(elem.first);
                std::cout << " dims[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
            }
            std::cout << std::endl;
        }
        std::cout << "#################### END   STATS ####################" << std::endl;
    }

    inline std::size_t bytes_left() const {
        assert(buffer_end_ >= tail_);
        return static_cast<std::size_t>(buffer_end_ - tail_);
    }

    inline void AllocateBuffer() {
        buffer_ = static_cast<uint8_t*>(GPUBackend::New(cache_size_, false));
        DALI_ENFORCE(buffer_ != nullptr);
        tail_ = buffer_;
        buffer_end_ = buffer_ + cache_size_;
        LOG_LINE << "cache size is " << cache_size_ / 1000000 << " MB" << std::endl;
    }

    struct DecodedImage {
        span<uint8_t, dynamic_extent> data;
        Dims dims;

        inline bool operator==(const DecodedImage& oth) const {
            return data == oth.data
                && dims == oth.dims;
        }
    };

    std::size_t cache_size_ = 0;
    std::size_t image_size_threshold_ = 0;
    bool stats_enabled_ = false;
    uint8_t* buffer_ = nullptr;
    uint8_t* buffer_end_ = nullptr;
    uint8_t* tail_ = nullptr;

    std::unordered_map<ImageKey, DecodedImage> cache_;
    mutable std::mutex mutex_;

    struct Stats {
        std::size_t decodes = 0;
        std::size_t reads = 0;
        bool is_cached = false;
    };
    mutable std::unordered_map<ImageKey, Stats> stats_;
    bool is_full = false;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_DECODER_CACHE_BLOB_H_
