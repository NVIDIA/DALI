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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_BLOB_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_BLOB_H_

#include <fstream>
#include <mutex>
#include <unordered_map>
#include "dali/kernels/span.h"
#include "dali/error_handling.h"
#include "dali/kernels/alloc.h"
#include "dali/pipeline/operators/decoder/cache/image_cache.h"

namespace dali {

class DLL_PUBLIC ImageCacheBlob : public ImageCache {
 public:
    DLL_PUBLIC ImageCacheBlob(std::size_t cache_size,
                              std::size_t image_size_threshold,
                              bool stats_enabled = false);

    ~ImageCacheBlob() override;

    DISABLE_COPY_MOVE_ASSIGN(ImageCacheBlob);

    bool IsCached(const ImageKey& image_key) const override;

    bool Read(const ImageKey& image_key,
              void* destination_data,
              cudaStream_t stream) const override;

    const ImageShape& GetShape(const ImageKey& image_key) const override;

    void Add(const ImageKey& image_key,
             const uint8_t *data,
             const ImageShape& data_shape,
             cudaStream_t stream) override;

    DecodedImage Get(const ImageKey &image_key) const override;

 protected:
    void print_stats() const;

    inline std::size_t images_seen() const {
        return (total_seen_images_ == 0) ?
            stats_.size() : total_seen_images_;
    }

    inline std::size_t bytes_left() const {
        DALI_ENFORCE(buffer_end_ >= tail_);
        return static_cast<std::size_t>(buffer_end_ - tail_);
    }

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

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_BLOB_H_
