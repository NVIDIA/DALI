#ifndef NDLL_PIPELINE_DATA_ALLOCATOR_H_
#define NDLL_PIPELINE_DATA_ALLOCATOR_H_

#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Base class for all user-defined allocators. Defines
 * the interface that must be implemented by all allocators.
 *
 * Allocators provide methods for allocating and deleting memory
 * located on gpu or cpu. User defined allocators should inherit
 * from {CPU,GPU}Allocator and define the necessary functions. On 
 * startup, the user calls "NDLLInit" to set polymorphic pointer
 * to CPUAllocator & GPUAllocator objects that are to be used by
 * the entire pipeline.
 */
class AllocatorBase {
public:
  /**
   * @brief Allocates `bytes` bytes of memory and sets the 
   * input ptr to the newly allocator memory
   */
  virtual void New(void **ptr, size_t bytes) = 0;

  /**
   * @brief Frees the memory pointed to by 'ptr'. Supports `bytes`
   * argument for certain types of allocators.
   */
  virtual void Delete(void *ptr, size_t bytes) = 0;
};

/**
 * @brief Default GPU memory allocator.
 */
class GPUAllocator : public AllocatorBase {
public:
  void New(void **ptr, size_t bytes) override {
    CUDA_CALL(cudaMalloc(ptr, bytes));
  }
  
  void Delete(void *ptr, size_t /* unused */) override {
    CUDA_CALL(cudaFree(ptr));
  }
};

/**
 * @brief Default CPU memory allocator.
 */
class CPUAllocator : public AllocatorBase {
public:
  void New(void **ptr, size_t bytes) override {
    *ptr = ::operator new(bytes);
  }
  
  void Delete(void *ptr, size_t /* unused */) override {
    ::operator delete(ptr);
  }
};

/**
 * @brief Pinned memory CPU allocator
 */
class PinnedCPUAllocator : public CPUAllocator {
public:
  void New(void **ptr, size_t bytes) override {
    CUDA_CALL(cudaMallocHost(ptr, bytes));
  }
  
  void Delete(void *ptr, size_t /* unused */) override {
    CUDA_CALL(cudaFreeHost(ptr));
  }
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_ALLOCATOR_H_
