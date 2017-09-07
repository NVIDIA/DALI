#ifndef NDLL_PIPELINE_BACKEND_H_
#define NDLL_PIPELINE_BACKEND_H_

#include <cuda_runtime_api.h>

#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Default GPU allocator for buffer types. User defined 
 * allocators must implement the same interface. The second
 * parameter to the 'Delete' method is the size of the allocation
 * in bytes, which is required for some types/implementations of
 * memory allocators used by the frameworks.
 */
class GPUBackend {
public:
  static void* New(size_t bytes) {
    void *ptr = nullptr;
    CUDA_ENFORCE(cudaMalloc(&ptr, bytes));
    return ptr;
  }
  static void Delete(void *ptr, size_t /* unused */) {
    CUDA_ENFORCE(cudaFree(ptr));
  }
};

class CPUBackend {
public:
  static void* New(size_t bytes) {
    return ::operator new(bytes);
  }
  static void Delete(void *ptr, size_t /* unused */) {
    ::operator delete(ptr);
  }
};

/**
 * @brief Allocates page-locked host memory
 */
class PinnedCPUBackend {
public:
  static void* New(size_t bytes) {
    void *ptr = nullptr;
    CUDA_ENFORCE(cudaMallocHost(&ptr, bytes));
    return ptr;
  }
  static void Delete(void *ptr, size_t /* unused */) {
    CUDA_ENFORCE(cudaFreeHost(ptr));
  }
};

} // namespace ndll

#endif // NDLL_PIPELINE_BACKEND_H_
