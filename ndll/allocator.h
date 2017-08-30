#ifndef NDDL_ALLOCATOR_H_
#define NDDL_ALLOCATOR_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Default Allocator for CPU memory
 */
class CpuAllocator {
public:
  static void* New(size_t bytes) {
    return ::operator new(bytes);
  }

  static void Delete(void *ptr, size_t /* unused */) {
    ::operator delete(ptr);
  }
private:
  CpuAllocator() {}
};

/**
 * @brief Allocator for pinned CPU memory
 */
class PinnedAllocator {
public:
  static void* New(size_t bytes) {
    void *ptr = nullptr;
    CUDA_CALL(cudaMallocHost(&ptr, bytes));
    return ptr;
  }

  static  void Delete(void *ptr, size_t /* unused */) {
    CUDA_CALL(cudaFreeHost(ptr));
  }
private:
  PinnedAllocator() {}
};

/**
 * @brief Default allocator for GPU memory
 */
class GpuAllocator {
public:
  static void* New(size_t bytes) {
    void *ptr = nullptr;
    CUDA_CALL(cudaMalloc(&ptr, bytes));
    return ptr;
  }

  static void Delete(void *ptr, size_t /* unused */) {
    CUDA_CALL(cudaFree(ptr));
  }
private:
  GpuAllocator() {}
};

} // namespace ndll

#endif // NDDL_ALLOCATOR_H_
