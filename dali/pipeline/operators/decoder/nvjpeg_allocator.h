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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_ALLOCATOR_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_ALLOCATOR_H_


#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/util/cuda_utils.h"

namespace dali {

namespace mem {
class BasicPinnedAllocator {
 public:
  static void PreallocateBuffers(size_t element_size_hint, size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);

    if (element_size_hint_ == 0) {
      element_size_hint_ = element_size_hint;
    } else if (element_size_hint_ != element_size_hint) {
      DALI_FAIL("All instances of nvJPEGDecoder should have the same host_memory_padding.");
    }
    free_buffers_pool_.reserve(free_buffers_pool_.size() + n_elements_hint);
    for (size_t i = 0; i < n_elements_hint; ++i) {
      void* buffer;
      CUDA_CALL(cudaHostAlloc(&buffer, element_size_hint, 0));
      free_buffers_pool_.push_back(buffer);
    }
  }

  static void FreeBuffers(size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);
    for (size_t i = 0; i < n_elements_hint; ++i) {
      // Will call std::terminate
      DALI_ENFORCE(free_buffers_pool_.size() > 0, "Some buffers were not returned by nvJPEG.");
      CUDA_CALL(cudaFreeHost(free_buffers_pool_.back()));
      free_buffers_pool_.pop_back();
    }
  }

  static int Alloc(void** ptr, size_t size, unsigned int flags) {
    std::lock_guard<std::mutex> l(m_);
    // Non managed buffer
    if (size > element_size_hint_ || free_buffers_pool_.empty()) {
      return cudaHostAlloc(ptr, size, flags) == cudaSuccess ? 0 : 1;
    }

    // Managed buffer, adding it to the allocated set
    void* buffer = free_buffers_pool_.back();
    free_buffers_pool_.pop_back();
    *ptr = buffer;
    allocated_buffers_.insert(buffer);
    return 0;
  }

  static int Free(void* ptr) {
    std::lock_guard<std::mutex> l(m_);
    if (allocated_buffers_.find(ptr) == allocated_buffers_.end()) {
      // Non managed buffer... just free
      return cudaFreeHost(ptr) == cudaSuccess ? 0 : 1;
    }
    // Managed buffer
    allocated_buffers_.erase(ptr);
    free_buffers_pool_.push_back(ptr);
    return 0;
  }

 private:
  static std::vector<void*> free_buffers_pool_;
  static size_t element_size_hint_;
  static std::unordered_set<void*> allocated_buffers_;

  static std::mutex m_;
};

class ChunkPinnedAllocator {
 public:
  static void PreallocateBuffers(size_t element_size_hint, size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);

    counter_++;

    if (element_size_hint_ == 0) {
      element_size_hint_ = element_size_hint;
    } else if (element_size_hint_ != element_size_hint) {
      DALI_FAIL("All instances of nvJPEGDecoder should have the same host_memory_padding.");
    }

    Chunk chunk;
    CUDA_CALL(cudaHostAlloc(&chunk.memory, element_size_hint * n_elements_hint, 0));
    auto& free_b = chunk.free_blocks;
    free_b.resize(n_elements_hint);
    std::iota(free_b.begin(), free_b.end(), 0);
    chunks_.push_back(chunk);
  }

  static void FreeBuffers(size_t n_elements_hint) {
    std::lock_guard<std::mutex> l(m_);
    counter_--;
    if (counter_ == 0) {
      for (auto chunk : chunks_) {
        // Will call std::terminate if CUDA_CALL fails
        CUDA_CALL(cudaFreeHost(chunk.memory));
      }
      chunks_.clear();
    }
  }

  static int Alloc(void** ptr, size_t size, unsigned int flags) {
    std::lock_guard<std::mutex> l(m_);
    // Non managed block
    if (size > element_size_hint_) {
      return cudaHostAlloc(ptr, size, flags) == cudaSuccess ? 0 : 1;
    }

    for (size_t chunk_idx = 0; chunk_idx < chunks_.size(); ++chunk_idx) {
      auto& chunk = chunks_[chunk_idx];
      // This chunk has no free block
      if (chunk.free_blocks.empty())
        continue;

      size_t block_idx = chunk.free_blocks.back();
      *ptr = static_cast<void*>(
              static_cast<uint8_t*>(chunk.memory) + (element_size_hint_ * block_idx));
      allocated_buffers_[*ptr] = std::make_pair(chunk_idx, block_idx);
      chunk.free_blocks.pop_back();
      return 0;
    }

    return cudaHostAlloc(ptr, size, flags) == cudaSuccess ? 0 : 1;
  }

  static int Free(void* ptr) {
    std::lock_guard<std::mutex> l(m_);
    auto it = allocated_buffers_.find(ptr);
    if (it == allocated_buffers_.end()) {
      // Non managed buffer... just free
      return cudaFreeHost(ptr) == cudaSuccess ? 0 : 1;
    }
    size_t chunk_idx, block_idx;
    std::tie(chunk_idx, block_idx) = it->second;
    auto& chunk = chunks_[chunk_idx];
    chunk.free_blocks.push_back(block_idx);
    allocated_buffers_.erase(it);
    return 0;
  }

 private:
  struct Chunk {
    void* memory;
    std::vector<int> free_blocks;
  };
  static std::vector<Chunk> chunks_;
  static size_t element_size_hint_;
  // ptr to (chunk_idx,block_idx)
  static std::unordered_map<void*, std::pair<size_t, size_t>> allocated_buffers_;

  static int counter_;
  static std::mutex m_;
};
}  // namespace mem

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_ALLOCATOR_H_

