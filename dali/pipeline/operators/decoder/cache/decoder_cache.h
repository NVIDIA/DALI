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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_H_
#define DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_H_

#include <cuda_runtime.h>
#include <string>
#include "dali/api_helper.h"
#include "dali/kernels/tensor_shape.h"

namespace dali {

class DLL_PUBLIC DecoderCache {
 public:
    using ImageKey = std::string;
    using ImageShape = kernels::TensorShape<3>;

    DLL_PUBLIC virtual ~DecoderCache() = default;

    DLL_PUBLIC virtual bool IsCached(const ImageKey& image_key) const = 0;

    DLL_PUBLIC virtual const ImageShape& GetShape(const ImageKey& image_key) const = 0;

    DLL_PUBLIC virtual void CopyData(const ImageKey& image_key,
                                     void* destination_buffer,
                                     cudaStream_t stream) const = 0;

    DLL_PUBLIC virtual void Add(const ImageKey& image_key,
                                const uint8_t *data,
                                const ImageShape& data_shape,
                                cudaStream_t stream) = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_CACHE_DECODER_CACHE_H_
