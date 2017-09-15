#ifndef NDLL_PIPELINE_BACKEND_H_
#define NDLL_PIPELINE_BACKEND_H_

#include <cuda_runtime_api.h>

#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Base class for all backend types. Defines the 
 * interface that must be implemented by all backends
 */
class BackendBase {
public:
  /**
   * @brief Allocates `bytes` bytes of memory and returns 
   * a pointer to the newly allocated memory
   */
  virtual void* New(size_t bytes) = 0;

  /**
   * @brief Frees the memory pointed to by 'ptr'. Supports `bytes`
   * argument for certain types of allocators.
   */
  virtual void Delete(void *ptr, size_t bytes) = 0;
};

/**
 * @brief Default GPU backend. User defined GPU backends must
 * derive from this class and override 'New' and 'Delete' as
 * they desire.
 */
class GPUBackend : BackendBase {
public:
  void* New(size_t bytes) override {
    void *ptr = nullptr;
    CUDA_CALL(cudaMalloc(&ptr, bytes));
    return ptr;
  }
  void Delete(void *ptr, size_t /* unused */) override{
    CUDA_CALL(cudaFree(ptr));
  }
};

/**
 * @brief Default CPU backend. User defined CPU backends must
 * derive from this class and override 'New' and 'Delete' as
 * they desire.
 */
class CPUBackend : BackendBase {
public:
  void* New(size_t bytes) override {
    return ::operator new(bytes);
  }
  void Delete(void *ptr, size_t /* unused */) override {
    ::operator delete(ptr);
  }
};

/**
 * @brief Allocates page-locked host memory
 */
class PinnedCPUBackend : CPUBackend {
public:
  void* New(size_t bytes) override {
    void *ptr = nullptr;
    CUDA_CALL(cudaMallocHost(&ptr, bytes));
    return ptr;
  }
  void Delete(void *ptr, size_t /* unused */) override {
    CUDA_CALL(cudaFreeHost(ptr));
  }
};

// Utility to copy between backends
inline void MemCopy(void *dst, const void *src, size_t bytes, cudaStream_t stream = 0) {
#ifndef NDEBUG
  NDLL_ENFORCE(dst != nullptr);
  NDLL_ENFORCE(src != nullptr);
#endif
  CUDA_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
}

} // namespace ndll

#endif // NDLL_PIPELINE_BACKEND_H_
