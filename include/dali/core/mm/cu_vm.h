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
  constexpr CUdeviceptr &ptr() noexcept { return first; }
  constexpr CUdeviceptr  ptr() const noexcept { return first; }
  constexpr size_t &size() noexcept { return second; }
  constexpr size_t  size() const noexcept { return second; }

  constexpr CUdeviceptr end() const noexcept { return first + second; }

  constexpr bool contains(CUdeviceptr ptr) const noexcept {
    return ptr >= this->ptr() && ptr < this->end();
  }
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

inline bool IsSupported() {
  static auto impl = []() {
    if (!cuInitChecked())
      return false;
    CUdeviceptr mem = 0;
    size_t size = 1<<24;  // This should be big enough for any granularity and small enough
                          // to be available.
    CUresult res = cuMemAddressReserve(&mem, size, 0, 0, 0);
    if (mem && res == CUDA_SUCCESS) {
      (void)cuMemAddressFree(mem, size);
    }
    return res != CUDA_ERROR_NOT_SUPPORTED;
  };
  static bool supported = impl();
  return supported;
}

/**
 * @brief Manages a virtual addresses range
 */
class CUMemAddressRange : public UniqueHandle<CUAddressRange, CUMemAddressRange> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(CUAddressRange, CUMemAddressRange);

  /**
   * @brief Starting address of the VA reservation.
   */
  constexpr CUdeviceptr ptr() const noexcept {
    return handle_.ptr();
  }

  /**
   * @brief Actual size, in bytes, of the VA reservation.
   */
  constexpr size_t size() const noexcept {
    return handle_.size();
  }

  /**
   * @brief One past the last address in this VA reservation.
   */
  constexpr CUdeviceptr end() const noexcept {
    return handle_.end();
  }

  /**
   * @brief Tells if the address is in the memory range managed by this handle
   */
  constexpr bool contains(CUdeviceptr ptr) const noexcept {
    return handle_.contains(ptr);
  }

  /**
   * @brief Reserves a virtual address range without physical backing storage
   *
   * This function is a wrapper around cuMemAddressReserve.
   *
   * The virtual address range reserved by this function is not allocated - however,
   * it is guaranteed that physical allocations can be mapped to addresses from this range
   * and that no other allocation functions (like cudaMalloc) will return addresses from this range.
   *
   * The addresses can be made usable by mapping phycial storage to them with cuvm::Map or
   * cuMemMap + cuMemSetAccess.
   *
   * @param size      Size, in bytes - should be a multiple of GetAddressGranularity - if it's not
   *                  it is rounded up to the next multiple
   * @param alignment Mostly ignored
   * @param start     The requested starting address - a hint for a starting address of the range.
   *                  The driver will attempt to allocate the address range at the address given,
   *                  but this is just a hint and the function will map a different range if the
   *                  requested location is not available.
   */
  static CUMemAddressRange Reserve(size_t size, size_t alignment = 0, CUdeviceptr start = 0) {
    CUdeviceptr ptr = 0;
    size_t grain = GetAddressGranularity();
    size = align_up(size, grain);
    CUDA_CALL(cuMemAddressReserve(&ptr, size, alignment, start, 0));
    return CUMemAddressRange(CUAddressRange{ ptr, size });
  }

  static void DestroyHandle(CUAddressRange handle) {
    CUDA_DTOR_CALL(cuMemAddressFree(handle.ptr(), handle.size()));
  }
};

/**
 * @brief Gets default CUmemAllocationProp for allocating memory on given device.
 * @param device_ordinal device ordinal or -1 for current device.
 */
inline CUmemAllocationProp DeviceMemProp(int device_ordinal = -1) {
  CUmemAllocationProp prop = {};
  if (device_ordinal < 0) {
    CUDA_CALL(cudaGetDevice(&device_ordinal));
  }
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_ordinal;
  return prop;
}

struct CUMemAllocation : std::pair<CUmemGenericAllocationHandle, size_t> {
  using std::pair<CUmemGenericAllocationHandle, size_t>::pair;
  constexpr CUmemGenericAllocationHandle &handle() noexcept { return first; }
  constexpr CUmemGenericAllocationHandle  handle() const noexcept { return first; }
  constexpr size_t &size() noexcept { return second; }
  constexpr size_t  size() const noexcept { return second; }
};

/**
 * @brief Manages a backing storage block for a virtual address range
 *
 * This is a handle to a block of physical memory. This memory doesn't have a virtual address
 * until mapped to an address range with cuvm::Map / cuMemMap
 */
class CUMem : public UniqueHandle<CUMemAllocation, CUMem> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(CUMemAllocation, CUMem);

  /**
   * @brief Driver handle to the physical memory block.
   */
  CUmemGenericAllocationHandle handle() const noexcept { return handle_.first; }

  /**
   * @brief Actual size, in bytes, of the allocation.
   */
  size_t size() const noexcept { return handle_.second; }

  /**
   * @brief Allocates `size` bytes of physical storage with specified properties.
   *
   * The memory must be mapped to a virtual address range before being accessible.
   * @see cuvm::Map
   *
   * @param size           Size, in bytes
   * @param device_ordinal The ordinal of the device on which the memory is going to be located.
   */
  static CUMem Create(size_t size, int device_ordinal = -1) {
    return Create(size, DeviceMemProp(device_ordinal));
  }

  /**
   * @brief Allocates `size` bytes of physical storage with specified properties.
   *
   * The memory must be mapped to a virtual address range before being accessible.
   * @see cuvm::Map
   *
   * @param size Size, in bytes
   * @param prop Allocation properties - indicate memory type and location
   */
  static CUMem Create(size_t size, const CUmemAllocationProp &prop) {
    size = align_up(size, GetAddressGranularity());
    CUmemGenericAllocationHandle handle;
    CUDA_CALL(cuMemCreate(&handle, size, &prop, 0));
    return CUMem({ handle, size });
  }

  static void DestroyHandle(CUMemAllocation handle) {
    CUDA_DTOR_CALL(cuMemRelease(handle.first));
  }
};

/**
 * @brief Maps physical memory to a virtual address range and sets R/W access to it
 *
 * @param virt_addr The beginning of the virtual address range to map
 * @param mem       The physical allocation handle
 * @param size      The size, in bytes, of the range to map. Typically, this is the entire physical
 *                  allocation size.
 */
template <typename T = void>
T *Map(CUdeviceptr virt_addr, CUmemGenericAllocationHandle mem, size_t size) {
  CUDA_CALL(cuMemMap(virt_addr, size, 0, mem, 0));

  CUmemAccessDesc access = {};
  CUmemAllocationProp prop = {};
  CUDA_CALL(cuMemGetAllocationPropertiesFromHandle(&prop, mem));
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_CALL(cuMemSetAccess(virt_addr, size, &access, 1));
  return reinterpret_cast<T*>(virt_addr);
}

template <typename T = void>
T *Map(CUdeviceptr virt_addr, CUMemAllocation mem) {
  return Map<T>(virt_addr, mem.handle(), mem.size());
}

inline void Unmap(CUdeviceptr ptr, size_t size) {
  CUDA_DTOR_CALL(cuMemUnmap(ptr, size));
}

inline void Unmap(const void *ptr, size_t size) {
  Unmap(reinterpret_cast<CUdeviceptr>(ptr), size);
}


}  // namespace cuvm
}  // namespace mm
}  // namespace dali

#else
#define DALI_USE_CUDA_VM_MAP 0
#endif

#endif  // DALI_CORE_MM_CU_VM_H_
