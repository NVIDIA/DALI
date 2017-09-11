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

  /**
   * @brief Indicates if the memory allocated is on GPU
   */
  virtual bool OnDevice() = 0;
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
    CUDA_ENFORCE(cudaMalloc(&ptr, bytes));
    return ptr;
  }
  void Delete(void *ptr, size_t /* unused */) override{
    CUDA_ENFORCE(cudaFree(ptr));
  }
  bool OnDevice() final { return true; }
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
  bool OnDevice() final { return false; }
};

/**
 * @brief Allocates page-locked host memory
 */
class PinnedCPUBackend : CPUBackend {
public:
  void* New(size_t bytes) override {
    void *ptr = nullptr;
    CUDA_ENFORCE(cudaMallocHost(&ptr, bytes));
    return ptr;
  }
  void Delete(void *ptr, size_t /* unused */) override {
    CUDA_ENFORCE(cudaFreeHost(ptr));
  }
};

} // namespace ndll

#endif // NDLL_PIPELINE_BACKEND_H_
