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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_LARGEST_ONLY_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_LARGEST_ONLY_H_

#include <functional>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>
#include "dali/pipeline/data/tensor_list.h"  // needed for Dims
#include "dali/pipeline/operators/decoder/cache/decoder_cache_blob.h"

namespace dali {

class DLL_PUBLIC DecoderCacheLargestOnly : public DecoderCacheBlob {
 public:
  DLL_PUBLIC DecoderCacheLargestOnly(std::size_t cache_size, bool stats_enabled = false);

  DISABLE_COPY_MOVE_ASSIGN(DecoderCacheLargestOnly);

  void Add(const ImageKey& image_key, const uint8_t* data, std::size_t data_size,
           const Dims& data_shape, cudaStream_t stream = 0) override;

 private:
  using QueueElement = std::pair<std::size_t, ImageKey>;
  std::priority_queue<QueueElement,
      std::vector<QueueElement>,
      std::greater<QueueElement>> biggest_images_;
  std::unordered_set<ImageKey> images_;
  bool start_caching_ = false;
  std::size_t biggest_images_total_ = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_LARGEST_ONLY_H_
