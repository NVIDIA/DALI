// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_DATA_BACKEND_H_
#define NDLL_PIPELINE_DATA_BACKEND_H_

#include <cuda_runtime_api.h>

#include "ndll/error_handling.h"
#include "ndll/pipeline/data/allocator.h"

namespace ndll {

// Called by "NDLLInit" to set up polymorphic pointers
// to user-defined memory allocators
void InitializeBackends(const OpSpec &cpu_allocator,
    const OpSpec &pinned_cpu_allocator,
    const OpSpec &gpu_allocator);

void SetCPUAllocator(const OpSpec& allocator);
void SetPinnedCPUAllocator(const OpSpec& allocator);
void SetGPUAllocator(const OpSpec& allocator);
/**
 * @brief Provides access to GPU allocator and other GPU meta-data.
 */
class GPUBackend final {
 public:
  static void* New(size_t bytes, bool);
  static void Delete(void *ptr, size_t bytes, bool);
};

/**
 * @brief Dummy Backend class to differentiate
 * Mixed ops.
 */
class Mixed final {};

/**
 * @brief Provides access to CPU allocator and other cpu meta-data
 */
class CPUBackend final {
 public:
  static void* New(size_t bytes, bool pinned);
  static void Delete(void *ptr, size_t bytes, bool pinned);
};

// Utility to copy between backends
inline void MemCopy(void *dst, const void *src, size_t bytes, cudaStream_t stream = 0) {
#ifndef NDEBUG
  NDLL_ENFORCE(dst != nullptr);
  NDLL_ENFORCE(src != nullptr);
#endif
  CUDA_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
}

}  // namespace ndll

#endif  // NDLL_PIPELINE_DATA_BACKEND_H_
