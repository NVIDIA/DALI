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

#include "dali/pipeline/operators/decoder/nvjpeg_decoder_cpu.h"

#include <unordered_map>

namespace dali {

DALI_REGISTER_OPERATOR(nvJPEGDecoderCPUStage, nvJPEGDecoderCPUStage, CPU);

DALI_SCHEMA(nvJPEGDecoderCPUStage)
  .DocStr(R"code(This operator is the CPU stage of nvJPEGDecoder, it is not supposed to be called separately.
It is automatically inserted during the pipeline creation.)code")
  .NumInput(1)
  .NumOutput(3)
  .MakeInternal()
  .AddParent("nvJPEGDecoder");

std::vector<void*> mem::BasicPinnedAllocator::free_buffers_pool_;
size_t mem::BasicPinnedAllocator::element_size_hint_ = 0;
std::unordered_set<void*> mem::BasicPinnedAllocator::allocated_buffers_;
std::mutex mem::BasicPinnedAllocator::m_;

std::vector<mem::ChunkPinnedAllocator::Chunk> mem::ChunkPinnedAllocator::chunks_;
size_t mem::ChunkPinnedAllocator::element_size_hint_;
std::unordered_map<void*, std::pair<size_t, size_t>> mem::ChunkPinnedAllocator::allocated_buffers_;
int mem::ChunkPinnedAllocator::counter_ = 0;
std::mutex mem::ChunkPinnedAllocator::m_;
}  // namespace dali
