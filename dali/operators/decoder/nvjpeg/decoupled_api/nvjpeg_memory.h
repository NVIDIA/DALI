// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_MEMORY_H_
#define DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_MEMORY_H_

#include <nvjpeg.h>
#include <thread>
#include "dali/kernels/alloc_type.h"

namespace dali {

namespace nvjpeg_memory {

/**
 * @brief Returns a buffer of at least the requested size, preferrably from the preallocated pool
 *        If no buffer that satisfies the requested arguments already exists in the pool, an allocation
 *        will take place
 */
void* GetBuffer(std::thread::id thread_id, kernels::AllocType alloc_type, size_t size);

/**
 * @brief Adds a new buffer to the pool for a given thread id, to be consumed later by ``GetBuffer``
 */
void AddBuffer(std::thread::id thread_id, kernels::AllocType alloc_type, size_t size);

/**
 * @brief Deletes all the buffers associated with a given thread id
 */
void DeleteAllBuffers(std::thread::id thread_id);

/**
 * @brief Enables/disables nvJPEG allocation statistics collection
 */
void SetEnableMemStats(bool enabled);

/**
 * @brief Adds an allocation to the statistics
 */
void AddMemStats(kernels::AllocType alloc_type, size_t size);

/**
 * @brief Prints nvJPEG memory allocation statistics
 */
void PrintMemStats();

nvjpegDevAllocator_t GetDeviceAllocator();
nvjpegPinnedAllocator_t GetPinnedAllocator();

}  // namespace nvjpeg_memory

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_MEMORY_H_
