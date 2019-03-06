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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_BLOB_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_BLOB_H_

#include <fstream>
#include <mutex>
#include <unordered_map>
#include "dali/kernels/span.h"
#include "dali/error_handling.h"
#include "dali/kernels/alloc.h"
#include "dali/pipeline/operators/decoder/cache/decoder_cache.h"

namespace dali {

class DLL_PUBLIC DecoderCacheBlob : public DecoderCache {
 public:
    DLL_PUBLIC DecoderCacheBlob(std::size_t cache_size,
                                std::size_t image_size_threshold,
                                bool stats_enabled = false);

    ~DecoderCacheBlob() override;

    DISABLE_COPY_MOVE_ASSIGN(DecoderCacheBlob);

    bool IsCached(const ImageKey& image_key) const override;

    const Dims& GetShape(const ImageKey& image_key) const override;

    void CopyData(const ImageKey& image_key,
                  void* destination_buffer,
                  cudaStream_t stream = 0) const override;

    void Add(const ImageKey& image_key,
             const uint8_t *data, std::size_t data_size,
             const Dims& data_shape,
             cudaStream_t stream = 0) override;

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

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_BLOB_H_
