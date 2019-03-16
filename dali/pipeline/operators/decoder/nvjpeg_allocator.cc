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

#include "dali/pipeline/operators/decoder/nvjpeg_allocator.h"

#include <unordered_map>

namespace dali {

namespace mem {

std::vector<void*> BasicPinnedAllocator::free_buffers_pool_;
size_t BasicPinnedAllocator::element_size_hint_ = 0;
std::unordered_set<void*> BasicPinnedAllocator::allocated_buffers_;
std::mutex BasicPinnedAllocator::m_;

std::vector<ChunkPinnedAllocator::Chunk> ChunkPinnedAllocator::chunks_;
size_t ChunkPinnedAllocator::element_size_hint_;
std::unordered_map<void*, std::pair<size_t, size_t>> ChunkPinnedAllocator::allocated_buffers_;
int ChunkPinnedAllocator::counter_ = 0;
std::mutex ChunkPinnedAllocator::m_;

}  // namespace mem

}  // namespace dali

