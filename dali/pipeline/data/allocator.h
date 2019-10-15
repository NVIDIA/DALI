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

#ifndef DALI_PIPELINE_DATA_ALLOCATOR_H_
#define DALI_PIPELINE_DATA_ALLOCATOR_H_

#include "dali/core/cuda_utils.h"
#include "dali/pipeline/operator/operator_factory.h"

namespace dali {

/**
 * @brief Base class for all user-defined allocators. Defines
 * the interface that must be implemented by all allocators.
 *
 * Allocators provide methods for allocating and deleting memory
 * located on gpu or cpu. User defined allocators should inherit
 * from {CPU,GPU}Allocator and define the necessary functions. On
 * startup, the user calls "DALIInit" to set polymorphic pointer
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
  ~GPUAllocator() override = default;

  void New(void **ptr, size_t bytes) override {
    CUDA_CALL(cudaMalloc(ptr, bytes));
  }

  void Delete(void *ptr, size_t /* unused */) override {
    if (ptr != nullptr) {
      CUDA_CALL(cudaFree(ptr));
    }
  }
};

DALI_DECLARE_OPTYPE_REGISTRY(GPUAllocator, GPUAllocator);

#define DALI_REGISTER_GPU_ALLOCATOR(OpName, OpType) \
  DALI_DEFINE_OPTYPE_REGISTERER(OpName, OpType,     \
      dali::GPUAllocator, dali::GPUAllocator, "GPU_Allocator")


/**
 * @brief Default CPU memory allocator.
 */
class CPUAllocator : public AllocatorBase {
 public:
  explicit CPUAllocator(const OpSpec &spec) : AllocatorBase(spec) {}
  ~CPUAllocator() override = default;

  void New(void **ptr, size_t bytes) override {
    *ptr = ::operator new(bytes);
  }

  void Delete(void *ptr, size_t /* unused */) override {
    ::operator delete(ptr);
  }
};

DALI_DECLARE_OPTYPE_REGISTRY(CPUAllocator, CPUAllocator);

#define DALI_REGISTER_CPU_ALLOCATOR(OpName, OpType) \
  DALI_DEFINE_OPTYPE_REGISTERER(OpName, OpType,     \
      dali::CPUAllocator, dali::CPUAllocator, "CPU_Allocator")

/**
 * @brief Pinned memory CPU allocator
 */
class PinnedCPUAllocator : public CPUAllocator {
 public:
  explicit PinnedCPUAllocator(const OpSpec &spec) : CPUAllocator(spec) {}
  ~PinnedCPUAllocator() override = default;

  void New(void **ptr, size_t bytes) override {
    CUDA_CALL(cudaMallocHost(ptr, bytes));
  }

  void Delete(void *ptr, size_t /* unused */) override {
    CUDA_CALL(cudaFreeHost(ptr));
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_ALLOCATOR_H_
