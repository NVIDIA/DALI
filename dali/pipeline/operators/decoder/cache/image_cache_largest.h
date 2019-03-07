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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_LARGEST_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_LARGEST_H_

#include <functional>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>
#include "dali/pipeline/operators/decoder/cache/image_cache_blob.h"
#include "dali/common.h"

namespace dali {

class DLL_PUBLIC ImageCacheLargest : public ImageCacheBlob {
 public:
  DLL_PUBLIC ImageCacheLargest(std::size_t cache_size, bool stats_enabled = false);

  DISABLE_COPY_MOVE_ASSIGN(ImageCacheLargest);

  void Add(const ImageKey& image_key, const uint8_t* data, const ImageShape& data_shape,
           cudaStream_t stream) override;

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

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_LARGEST_H_
