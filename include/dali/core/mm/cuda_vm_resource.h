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

#ifndef DALI_CORE_MM_CUDA_VM_RESOURCE_H_
#define DALI_CORE_MM_CUDA_VM_RESOURCE_H_

#include "dali/core/mm/cu_vm.h"
#include <vector>

#if DALI_USE_CUDA_VM_MAP
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"
#include "dali/core/bitmask.h"

namespace dali {
namespace mm {

namespace detail {

class vm_block_pool {
 public:
  void set_params(size_t size, int device_ordinal = -1) {
    if (device_ordinal < 0)
      CUDA_CALL(cudaGetDevice(&device_ordinal));
    if (size == block_size_ || device_ordinal != device_ordinal_)
      return;
    if (blocks_taken_ > 0)
      throw std::logic_error("Cannot change the block size or target device when there have "
                             "been blocks taken from the pool.");
    purge();
    block_size_ = size;
    device_ordinal_ = device_ordinal;
  }

  void purge() {
    for (auto &h : handles_) {
      CUDA_CALL(cuMemRelease(h));
    }
    handles_.clear();
  }

  cuvm::CUMem get() {
    if (handles_.empty()) {
      if (block_size_ < 1 || device_ordinal_ < 0)
        configure();
      return cuvm::CUMem::Create(block_size_, device_ordinal_);
    } else {
      auto ret = cuvm::CUMem({ handles_.back(), block_size_ });
      handles_.pop_back();
      return ret;
    }
  }

  void put(CUmemGenericAllocationHandle handle) {
    handles_.push_back(handle);
  }

  void put(cuvm::CUMem mem) {
    // try to push_back before releasing - if it fails, mem will be properly destroyed
    handles_.push_back(mem.handle());
    // now it's safe to release
    (void)mem.release();
  }

  void configure() {
    if (device_ordinal_ < 0) {
      CUDA_CALL(cudaGetDevice(&device_ordinal_));
    }
    if (block_size_ == 0) {
      CUdevice device;
      CUDA_CALL(cuDeviceGet(&device, device_ordinal_));
      CUDA_CALL(cuDeviceTotalMem(&total_mem_, device));
      size_t grain = cuvm::GetAddressGranularity();
      block_size_ = std::max(grain,  // at least grain, for correctness
                             std::min<size_t>(next_pow2(total_mem_ >> 8),  // capacity-dependent...
                                              64 << 20));  // ...but capped at 64 MiB
    }
  }

  using mem_handle_t = CUmemGenericAllocationHandle;
 protected:
  size_t block_size_ = 0;
  size_t total_mem_ = 0;
  int device_ordinal_ = -1;
 private:
  vector<CUmemGenericAllocationHandle> handles_;
  int blocks_taken_ = 0;
};

}  // namespace detail

template <typename LockType = std::mutex>
class cuda_vm_resource : public memory_resource<memory_kind::device>
                       , private detail::vm_block_pool {
 public:
  explicit cuda_vm_resource(int device_ordinal = -1) {
    device_ordinal_ = device_ordinal;
    configure();
  }
  using mem_handle_t = CUmemGenericAllocationHandle;

 private:
  void adjust_params(size_t &size, size_t &alignment) {
    alignment = std::max(alignment, next_pow2(size >> 11));
    alignment = std::min(alignment, block_size_);
    size = align_up(size, alignment);
  }

  virtual void *do_allocate(size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    char *ptr = static_cast<char*>(free_mapped_.get(size, alignment));
    if (ptr) {
      free_va_.get_specific_block(ptr, ptr + size);
      return ptr;
    }
    ptr = get_va(size, alignment);
    //map_storage(ptr, size);
    ptr = free_mapped_.get(size, alignment);
    return ptr;
  }

  void *get_va(size_t size, size_t alignment) {
    void *ptr = free_va_.get(size, alignment);
    if (ptr)
      return ptr;
    allocate_va_region(size);
    ptr = free_va_.get(size, alignment);
    assert(ptr);
    return ptr;
  }

  void allocate_va_region(size_t min_size) {
    size_t va_size = std::max(next_pow2(min_size), min_va_size);

    CUdeviceptr hint = {};
    if (va_regions.empty()) {
      // Calculate the hint address for the allocations for this device.
      // There's some not very significant base and different devices get
      // address spaces separated by 2^48 - this should be quite enough.
      hint = 0x1aaa000000000000u + device_ordinal_ * 0x1000000000000u;
    } else {
      // Try to allocate after the last VA for this device
      auto &last_va = va_regions.back().va;
      hint = last_va.ptr() + last_va.size();
    }
    cuvm::CUMemAddressRange va = cuvm::CUMemAddressRange::Reserve(va_size, 0, hint);
      std::cerr << "Allocated a virtual memory range at: " << (void*)va.ptr() << " - hint was "
                << (void*)hint;
    va_regions.push_back({ std::move(va), block_size_ });
  }

  virtual void do_deallocate(void *ptr, size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    free_mapped_.put(ptr, size);
    free_va_.put(ptr, size);
  }

  size_t total_used_, total_mapped_;
  size_t min_va_size = 0x100000000u;  // 4GiB

  struct va_region {
    va_region(cuvm::CUMemAddressRange range, size_t block_size)
    : va(std::move(range)), block_size(block_size) {
      size_t size = range.size();
      size_t blocks_in_range = size / block_size;
      mapping.resize(blocks_in_range);
      mapped.resize(blocks_in_range, false);
      used.resize(blocks_in_range, false);
    }
    ~va_region() {
      CUdeviceptr ptr = va.ptr();
      for (size_t i = 0; i < mapping.size(); i++, ptr += block_size)
        if (h) {
          CUDA_DTOR_CALL(cuMemUnmap(ptr, block_size));
          CUDA_DTOR_CALL(cuMemRelease(h));
          h = nullptr;
        }
      }
    }
    cuvm::CUMemAddressRange va;
    size_t                  block_size;
    vector<mem_handle_t>    mapping;
    bitmask                 mapped, used;
  };
  SmallVector<va_region, 8> va_regions;

  coalescing_free_tree free_mapped_, free_va_;
  LockType    pool_lock_;
  std::mutex  mem_lock_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_USE_CUDA_VM_MAP

#endif  // DALI_CORE_MM_CUDA_VM_RESOURCE_H_
