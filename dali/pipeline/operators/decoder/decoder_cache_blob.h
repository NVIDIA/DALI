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

#include <fstream>
#include <string>
#include <mutex>
#include <unordered_map>
#include "dali/pipeline/data/tensor_list.h"  // needed for Dims
#include "dali/kernels/span.h"
#include "dali/error_handling.h"
#include "dali/kernels/alloc.h"

namespace dali {

class DecoderCacheBlob {
 public:
    using ImageKey = std::string;

    inline DecoderCacheBlob(std::size_t cache_size,
                            std::size_t image_size_threshold,
                            bool stats_enabled = false )
        : cache_size_(cache_size)
        , image_size_threshold_(image_size_threshold)
        , stats_enabled_(stats_enabled) {
        DALI_ENFORCE(image_size_threshold <= cache_size_,
            "Cache size should fit at least one image");

        buffer_ = kernels::memory::alloc_unique<uint8_t>(
            kernels::AllocType::GPU, cache_size_);
        DALI_ENFORCE(buffer_ != nullptr);
        tail_ = buffer_.get();
        buffer_end_ = buffer_.get() + cache_size_;
        LOG_LINE << "cache size is " << cache_size_ / 1000000 << " MB" << std::endl;
    }

    inline virtual ~DecoderCacheBlob() {
        if (stats_enabled_ && images_seen() > 0)
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
        DALI_ENFORCE(data.data() < tail_);
        DALI_ENFORCE(data.data() + data.size() <= tail_);
        MemCopy(destination_buffer,
                data.data(),
                data.size(),
                stream);

        if (stats_enabled_)
            stats_[image_key].reads++;
    }

    inline virtual void Add(const ImageKey& image_key,
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

 protected:
    inline void print_stats() const {
        static std::mutex stats_mutex;
        std::lock_guard<std::mutex> lock(stats_mutex);
        std::size_t images_cached = 0;
        for (auto &elem : stats_)
            if (elem.second.is_cached)
                images_cached++;
        DALI_ENFORCE(images_cached <= images_seen());
        const char* log_filename = std::getenv("DALI_LOG_FILE");
        std::ofstream log_file;
        if (log_filename)
            log_file.open(log_filename);
        std::ostream& out = log_filename ? log_file : std::cout;
        out << "#################### CACHE STATS ####################" << std::endl;
        out << "cache_size: " << cache_size_ << std::endl;
        out << "cache_threshold: " << image_size_threshold_ << std::endl;
        out << "is_cache_full: " << static_cast<int>(is_full) << std::endl;
        out << "images_seen: " << images_seen() << std::endl;
        out << "images_cached: " << images_cached << std::endl;
        out << "images_not_cached: " << images_seen() - images_cached << std::endl;
        for (auto &elem : stats_) {
            out << "image[" << elem.first
                    << "] : is_cached[" << static_cast<int>(elem.second.is_cached)
                    << "] decodes[" << elem.second.decodes
                    << "] reads[" << elem.second.reads << "]";
            if (elem.second.is_cached) {
                Dims dims = GetShape(elem.first);
                out << " dims[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
            }
            out << std::endl;
        }
        out << "#################### END   STATS ####################" << std::endl;
    }

    inline std::size_t images_seen() const {
        return (total_seen_images_ == 0) ?
            stats_.size() : total_seen_images_;
    }

    inline std::size_t bytes_left() const {
        DALI_ENFORCE(buffer_end_ >= tail_);
        return static_cast<std::size_t>(buffer_end_ - tail_);
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
    kernels::memory::KernelUniquePtr<uint8_t> buffer_;
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
    std::size_t total_seen_images_ = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_DECODER_CACHE_BLOB_H_
