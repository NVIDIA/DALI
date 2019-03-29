// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_BACKEND_H_
#define DALI_PIPELINE_DATA_BACKEND_H_

#include <cuda_runtime_api.h>
#include <memory>

#include "dali/error_handling.h"
#include "dali/pipeline/data/allocator.h"

namespace dali {

// Called by "DALIInit" to set up polymorphic pointers
// to user-defined memory allocators
void InitializeBackends(const OpSpec &cpu_allocator,
    const OpSpec &pinned_cpu_allocator,
    const OpSpec &gpu_allocator);

DLL_PUBLIC void SetCPUAllocator(const OpSpec& allocator);
DLL_PUBLIC void SetPinnedCPUAllocator(const OpSpec& allocator);
DLL_PUBLIC void SetGPUAllocator(const OpSpec& allocator);
DLL_PUBLIC void SetGPUAllocator(std::unique_ptr<GPUAllocator> allocator);

GPUAllocator& GetGPUAllocator();
/**
 * @brief Provides access to GPU allocator and other GPU meta-data.
 */
class DLL_PUBLIC GPUBackend final {
 public:
  DLL_PUBLIC static void* New(size_t bytes, bool);
  DLL_PUBLIC static void Delete(void *ptr, size_t bytes, bool);
};

/**
 * @brief Dummy Backend class to differentiate
 * Mixed ops.
 */
class DLL_PUBLIC MixedBackend final {};

/**
 * @brief Dummy Backend class to differentiate
 * Support ops.
 */
class DLL_PUBLIC SupportBackend final {};

/**
 * @brief Provides access to CPU allocator and other cpu meta-data
 */
class DLL_PUBLIC CPUBackend final {
 public:
  DLL_PUBLIC static void* New(size_t bytes, bool pinned);
  DLL_PUBLIC static void Delete(void *ptr, size_t bytes, bool pinned);
};

// Utility to copy between backends
inline void MemCopy(void *dst, const void *src, size_t bytes, cudaStream_t stream = 0) {
  // Copying 0 bytes is no-op anyways
  if (bytes == 0) {
    return;
  }
#ifndef NDEBUG
  DALI_ENFORCE(dst != nullptr);
  DALI_ENFORCE(src != nullptr);
#endif
  CUDA_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_BACKEND_H_
