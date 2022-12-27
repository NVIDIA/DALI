// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_INFLATE_INFLATE_GPU_H_
#define DALI_OPERATORS_DECODER_INFLATE_INFLATE_GPU_H_

#include "dali/pipeline/data/views.h"

namespace dali {

namespace inflate {

#define INFLATE_SUPPORTED_TYPES \
  (bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, float16)

/**
 * @brief Fills the possibly uninitialized memory with 0s.
 *
 * If the actual size of a chunks after decompression is less than reported by the user, we could
 * end up with returning some uninitialized memory.
 */
void FillTheTails(DALIDataType output_type, int batch_size, void* const* chunks_dev,
                  size_t* actual_sizes_dev, size_t* output_sizes_dev, cudaStream_t stream);

}  // namespace inflate

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_INFLATE_INFLATE_GPU_H_
