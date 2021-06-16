// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_CU_VM_H_
#define DALI_CORE_MM_CU_VM_H_

#include <cuda.h>
#include <utility>  // This should be in ifdef, but cpplint can't see it there
#if CUDA_VERSION >= 10020
#define DALI_USE_CUDA_VM_MAP 1

#include "dali/core/unique_handle.h"
#include "dali/core/cuda_error.h"
#include "dali/core/mm/detail/align.h"

namespace dali {
namespace mm {
namespace cuvm {

/**
 * @brief A virtual address range
 */
struct CUAddressRange : std::pair<CUdeviceptr, size_t> {
  using std::pair<CUdeviceptr, size_t>::pair;
  CUdeviceptr &ptr() noexcept { return first; }
  CUdeviceptr  ptr() const noexcept { return first; }
  size_t &size() noexcept { return second; }
  size_t  size() const noexcept { return second; }
};

static inline size_t GetAddressGranularity() {
  static auto impl = []() {
    if (!cuInitChecked())
      throw std::runtime_error("Cannot initialize CUDA driver");
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    size_t grain;
    CUDA_CALL(cuMemGetAllocationGranularity(&grain, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return grain;
  };
  static size_t grain = impl();
  return grain;
}

/**
 * @brief Manages a virtual addresses range
 */
class CUMemAddressRange : public UniqueHandle<CUAddressRange, CUMemAddressRange> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(CUAddressRange, CUMemAddressRange);

  static CUMemAddressRange Reserve(size_t size, size_t alignment = 0, CUdeviceptr start = 0) {
    CUdeviceptr ptr = 0;
    size_t grain = GetAddressGranularity();
    size = align_up(size, grain);
    CUDA_CALL(cuMemAddressReserve(&ptr, size, alignment, start, 0));
    return CUMemAddressRange(CUAddressRange{ ptr, size });
  }

  CUdeviceptr ptr() const noexcept {
      return handle_.ptr();
  }
  size_t size() const noexcept {
      return handle_.size();
  }

  static void DestroyHandle(CUAddressRange handle) {
    CUDA_DTOR_CALL(cuMemAddressFree(handle.ptr(), handle.size()));
  }
};

/**
 * @brief Gets default CUmemAllocationProp for allocating memory on given device.
 * @param device_id device ordinal or -1 for current device.
 */
inline CUmemAllocationProp DeviceMemProp(int device_id = -1) {
  CUmemAllocationProp prop = {};
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  return prop;
}

struct CUMemAllocation : std::pair<CUmemGenericAllocationHandle, size_t> {
  using std::pair<CUmemGenericAllocationHandle, size_t>::pair;
  CUmemGenericAllocationHandle &handle() noexcept { return first; }
  CUmemGenericAllocationHandle  handle() const noexcept { return first; }
  size_t &size() noexcept { return second; }
  size_t  size() const noexcept { return second; }
};

/**
 * @brief Manages a backing storage block for a virtual address range
 */
class CUMem : public UniqueHandle<CUMemAllocation, CUMem> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(CUMemAllocation, CUMem);

  static void DestroyHandle(CUMemAllocation handle) {
    CUDA_DTOR_CALL(cuMemRelease(handle.first));
  }

  static CUMem Create(size_t size, int device_id = -1) {
    return Create(size, DeviceMemProp(device_id));
  }

  static CUMem Create(size_t size, const CUmemAllocationProp &prop) {
    size = align_up(size, GetAddressGranularity());
    CUmemGenericAllocationHandle handle;
    CUDA_CALL(cuMemCreate(&handle, size, &prop, 0));
    return CUMem({ handle, size });
  }

  CUmemGenericAllocationHandle handle() const noexcept { return handle_.first; }

  size_t size() const noexcept { return handle_.second; }
};

template <typename T = void>
T *Map(CUdeviceptr virt_addr, CUmemGenericAllocationHandle mem, size_t size, int device_id = -1) {
  CUDA_CALL(cuMemMap(virt_addr, size, 0, mem, 0));

  if (device_id < 0)
    CUDA_CALL(cudaGetDevice(&device_id));
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device_id;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_CALL(cuMemSetAccess(virt_addr, size, &access, 1));
  return reinterpret_cast<T*>(virt_addr);
}

template <typename T = void>
T *Map(CUdeviceptr virt_addr, CUMemAllocation mem, int device_id = -1) {
  return Map<T>(virt_addr, mem.handle(), mem.size(), device_id);
}

inline void Unmap(const void *ptr, size_t size) {
  CUDA_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size));
}


}  // namespace cuvm
}  // namespace mm
}  // namespace dali

#else
#define DALI_USE_CUDA_VM_MAP 0
#endif

#endif  // DALI_CORE_MM_CU_VM_H_
