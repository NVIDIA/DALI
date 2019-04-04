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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_H_

#include <cuda_runtime.h>
#include <string>
#include "dali/api_helper.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/tensor_view.h"

namespace dali {

class DLL_PUBLIC ImageCache {
 public:
  using ImageKey = std::string;
  using ImageShape = kernels::TensorShape<3>;
  using DecodedImage = kernels::TensorView<kernels::StorageGPU, uint8_t, 3>;

  DLL_PUBLIC virtual ~ImageCache() = default;

  /**
   * @brief Check whether an image is present in the cache
   * @param image_key key representing the image in cache
   */
  DLL_PUBLIC virtual bool IsCached(const ImageKey& image_key) const = 0;

  /**
   * @brief Get image dimensions
   * @param image_key key representing the image in cache
   */
  DLL_PUBLIC virtual const ImageShape& GetShape(const ImageKey& image_key) const = 0;

    /**
     * @brief Try to read from cache
     * @param image_key key representing the image in cache
     * @param destination_data destination buffer
     * @param stream cuda stream
     * @returns true if successful cache read, false otherwise
     */
    DLL_PUBLIC virtual bool Read(const ImageKey& image_key,
                                 void* destination_data,
                                 cudaStream_t stream) const = 0;

  /**
   * @brief Try to add entry to cache.
   * @remarks Whether the entry is registered or not depends on the particular implementation
   *          and the state of the cache
   * @param image_key key representing the image in cache
   * @param data buffer
   * @param data_shape dimensions of the data to be stored
   * @param stream cuda stream
   */
  DLL_PUBLIC virtual void Add(const ImageKey& image_key,
                              const uint8_t *data,
                              const ImageShape& data_shape,
                              cudaStream_t stream) = 0;

  /**
   * @brief Get a cache entry describing an image
   * @param image_key key of the cached image
   * @return Pointer and shape of the cached image; if not found, data is null
   * @remarks This function is valid only if the implementation doesn't evict
   *          images from the cache.
   */
  DLL_PUBLIC virtual DecodedImage Get(const ImageKey &image_key) const = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_IMAGE_CACHE_H_
