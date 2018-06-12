// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_DATA_ALLOCATOR_H_
#define NDLL_PIPELINE_DATA_ALLOCATOR_H_

#include <cuda.h>

#include "ndll/error_handling.h"
#include "ndll/pipeline/operators/operator_factory.h"

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
  explicit AllocatorBase(const OpSpec &) {}
  virtual ~AllocatorBase() = default;

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
  explicit GPUAllocator(const OpSpec &spec) : AllocatorBase(spec) {}
  virtual ~GPUAllocator() = default;

  virtual void New(void **ptr, size_t bytes) {
    CUDA_CALL(cudaMalloc(ptr, bytes));
  }

  virtual void Delete(void *ptr, size_t /* unused */) {
    if (ptr != nullptr) {
      int dev;
      cudaGetDevice(&dev);
      std::cout << "HI " << ptr << " " << dev << std::endl;
    CUdeviceptr cuptr = (CUdeviceptr) ptr;
    CUcontext ctx;
    CUpointer_attribute attr = CU_POINTER_ATTRIBUTE_CONTEXT;
    CUresult result = cuPointerGetAttribute(&ctx, attr, cuptr);
    if (result == CUDA_SUCCESS) {
      std::cout << "HI2 " << ptr << std::endl;
      cuCtxPushCurrent(ctx);
      CUDA_CALL(cudaFree(ptr));
      cuCtxPopCurrent(&ctx);
    }
    }
  }
};

NDLL_DECLARE_OPTYPE_REGISTRY(GPUAllocator, GPUAllocator);

#define NDLL_REGISTER_GPU_ALLOCATOR(OpName, OpType) \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,     \
      ndll::GPUAllocator, ndll::GPUAllocator)


/**
 * @brief Default CPU memory allocator.
 */
class CPUAllocator : public AllocatorBase {
 public:
  explicit CPUAllocator(const OpSpec &spec) : AllocatorBase(spec) {}
  virtual ~CPUAllocator() = default;

  void New(void **ptr, size_t bytes) override {
    *ptr = ::operator new(bytes);
  }

  void Delete(void *ptr, size_t /* unused */) override {
    ::operator delete(ptr);
  }
};

NDLL_DECLARE_OPTYPE_REGISTRY(CPUAllocator, CPUAllocator);

#define NDLL_REGISTER_CPU_ALLOCATOR(OpName, OpType) \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,     \
      ndll::CPUAllocator, ndll::CPUAllocator)

/**
 * @brief Pinned memory CPU allocator
 */
class PinnedCPUAllocator : public CPUAllocator {
 public:
  explicit PinnedCPUAllocator(const OpSpec &spec) : CPUAllocator(spec) {}
  virtual ~PinnedCPUAllocator() = default;

  void New(void **ptr, size_t bytes) override {
    CUDA_CALL(cudaMallocHost(ptr, bytes));
  }

  void Delete(void *ptr, size_t /* unused */) override {
    CUDA_CALL(cudaFreeHost(ptr));
  }
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_DATA_ALLOCATOR_H_
