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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_MEMORY_H_
#define DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_MEMORY_H_

#include <nvjpeg.h>
#include <thread>
#include "dali/kernels/alloc_type.h"

namespace dali {

namespace nvjpeg_memory {

void* GetBuffer(std::thread::id thread_id, kernels::AllocType alloc_type, size_t size);
void AddBuffer(std::thread::id thread_id, kernels::AllocType alloc_type, size_t size);
void DeleteAllBuffers(std::thread::id thread_id);

nvjpegDevAllocator_t GetDeviceAllocator();
nvjpegPinnedAllocator_t GetPinnedAllocator();

}  // namespace nvjpeg_memory

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_MEMORY_H_