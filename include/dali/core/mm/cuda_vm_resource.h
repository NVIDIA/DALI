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

class cuda_vm_resource : public memory_resource<memory_kind::device> {
 public:
  explicit cuda_vm_resource(int device_ordinal = -1) {
    device_ordinal_ = device_ordinal;
    configure();
  }
  using mem_handle_t = CUmemGenericAllocationHandle;

  ~cuda_vm_resource() {
    purge();
  }

  void purge() {
    for (auto &r : va_regions_)
      r.purge();
  }

 private:
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

  struct va_region {
    va_region(cuvm::CUMemAddressRange range, size_t block_size)
    : address_range(std::move(range)), block_size(block_size) {
      size_t size = address_range.size();
      size_t blocks_in_range = size / block_size;
      mapping.resize(blocks_in_range);
      mapped.resize(blocks_in_range, false);
      available.resize(blocks_in_range, false);
    }

    va_region(va_region &&) = default;
    va_region &operator=(va_region &&) = default;

    ~va_region() {
      purge();
    }

    int num_blocks() const {
      return mapping.size();
    }

    void purge() {
      CUdeviceptr ptr = address_range.ptr();
      for (int i = 0; i < num_blocks(); i++, ptr += block_size) {
        auto &h = mapping[i];
        if (h) {
          cuvm::Unmap(ptr, block_size);
          CUDA_DTOR_CALL(cuMemRelease(h));
          h = {};
        }
      }
      mapped.fill(false);
      available.fill(false);
    }

    CUdeviceptr block_dptr(ptrdiff_t block_index) const noexcept {
      return address_range.ptr() + block_index * block_size;
    }

    template <typename T = void>
    T *block_ptr(ptrdiff_t block_index) const noexcept {
      return reinterpret_cast<T*>(block_dptr(block_index));
    }

    void map_block(int block_idx, cuvm::CUMem mem) {
      assert(mem.size() == block_size);
      cuvm::Map(block_dptr(block_idx), mem);
      mapping[block_idx] = mem.release().first;
      mapped[block_idx] = true;
      available[block_idx] = true;
      available_blocks++;
    }

    cuvm::CUMem unmap_block(ptrdiff_t block_idx) {
      assert(available[block_idx]);
      cuvm::Unmap(block_dptr(block_idx), block_size);
      cuvm::CUMem mem({mapping[block_idx], block_size });
      mapping[block_idx] = 0;
      mapped[block_idx] = false;
      available[block_idx] = false;
      available_blocks--;
      return mem;
    }

    cuvm::CUMemAddressRange address_range;
    vector<mem_handle_t>    mapping;
    bitmask                 mapped, available;
    ptrdiff_t               available_blocks = 0;
    size_t                  block_size = 0;
  };

  SmallVector<va_region, 8> va_regions_;
  size_t total_used_, total_mapped_;
  size_t min_va_size = 0x100000000u;  // 4GiB
  size_t block_size_ = 0;
  size_t total_mem_ = 0;
  int device_ordinal_ = -1;

  void adjust_params(size_t &size, size_t &alignment) {
    alignment = std::max(alignment, next_pow2(size >> 11));
    alignment = std::min(alignment, block_size_);
    size = align_up(size, alignment);
  }

  virtual void *do_allocate(size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    // try to get a free region that's already mapped
    void *ptr = try_get_mapped(size, alignment);
    if (ptr)
      return ptr;
    ptr = get_va(size, alignment);
    map_storage(ptr, align_up(size, block_size_));
    ptr = try_get_mapped(size, alignment);
    assert(ptr != nullptr);
    return ptr;
  }

  void *try_get_mapped(size_t size, size_t alignment) {
    char *ptr = static_cast<char*>(free_mapped_.get(size, alignment));
    if (ptr) {
      free_va_.get_specific_block(ptr, ptr + size);
      //#error TODO mark overlapping blocks as not available
      return ptr;
    } else {
      return nullptr;
    }
  }

  void *get_va(size_t size, size_t alignment) {
    void *ptr = free_va_.get(size, alignment);
    if (ptr)
      return ptr;
    va_allocate(size);
    ptr = free_va_.get(size, alignment);
    assert(ptr);
    return ptr;
  }

  void va_allocate(size_t min_size) {
    size_t va_size = std::max(next_pow2(min_size), min_va_size);

    CUdeviceptr hint = {};
    va_region *last_va = nullptr;
    if (va_regions_.empty()) {
      // Calculate the hint address for the allocations for this device.
      // There's some not very significant base and different devices get
      // address spaces separated by 2^40 - this should be quite enough.
      hint = (device_ordinal_ + 1) * 0x10000000000u;
    } else {
      // Try to allocate after the last VA for this device
      last_va = &va_regions_.back();
      hint = last_va->address_range.ptr() + last_va->address_range.size();
    }
    cuvm::CUMemAddressRange va = cuvm::CUMemAddressRange::Reserve(va_size, 0, hint);
    std::cerr << "Allocated a virtual memory range at: " << (void*)va.ptr() << " - hint was "
              << (void*)hint;
    va_regions_.emplace_back(std::move(va), block_size_);
    auto &region = va_regions_.back();
    free_va_.put(region.block_ptr(0), region.block_size * region.num_blocks());
  }

  virtual void do_deallocate(void *ptr, size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    free_mapped_.put(ptr, size);
    free_va_.put(ptr, size);
    mark_as_available(ptr, size);
  }

  bool is_block_aligned(void *ptr) const noexcept {
    return detail::is_aligned(ptr, block_size_));
  }

  bool is_free_mapped(char *start, char *end) const noexcept {
    return free_mapped_.contains(start, end);
  }

  std::pair<char *, char *> block_bounds(char *ptr) const {
    char *start = detail::align_ptr_down(ptr, block_size_);
    char *end   = start + block_size_;
    return { start, end };
  }

  void mark_as_available(void *ptr, size_t size) {
    char *cptr = static_cast<char *>(ptr);

    char *start = nullptr, *end = nullptr;
    if (!is_block_aligned(ptr)) {
      auto block_bounds = block_bounds(cptr);
      start = free_mapped_.contains(block_bounds.first, block_bounds.second)
              ? block_bounds.first : block_bounds.second;
    } else {
      start = cptr;
    }
    if (!is_block_aligned(ptr + size)) {
      auto block_bounds = block_bounds(cptr);
      end = free_mapped_.contains(block_bounds.first, block_bounds.second)
              ? block_bounds.second : block_bounds.first;
    } else {
      end = cptr + size;
    }

    set_block_avaialbility(start, end, true);
  }

  void set_block_avaialbility(char *start, char *end, bool available) {
    for (char *ptr = start; ptr < end; ) {
      va_region *va = va_find(ptr);
    }
  }

  void map_storage(void *ptr, size_t size) {
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    auto *region = va_find(ptr);
    assert(region);
    CUdeviceptr region_start  = region->address_range.ptr();
    CUdeviceptr region_end    = region->address_range.end();
    ptrdiff_t offset = dptr - region_start;
    int block_idx = offset / block_size_;
    int blocks = size / block_size_;
    struct block_range {
      va_region *region;
      int begin, end;
    };
    SmallVector<block_range, 8> to_map;
    int blocks_to_map = 0;
    while (blocks) {
      int next_unmapped = region->mapped.find(false, block_idx);
      if (next_unmapped - block_idx >= blocks)
        return;  // everything we need is mapped)
      if (next_unmapped == region->num_blocks()) {
        blocks -= next_unmapped - block_idx;
        region = va_find(region_end);
        assert(region);
        region_start = region->address_range.ptr();
        region_end = region->address_range.end();
        block_idx = 0;
      } else {
        int next_mapped = region->mapped.find(true, next_unmapped + 1);
        blocks_to_map += next_mapped - next_unmapped;
        if (blocks_to_map > blocks)
          blocks_to_map = blocks;
        block_idx += blocks_to_map;
        blocks -= blocks_to_map;
        to_map.push_back({ region, next_unmapped, block_idx });
      }
    }

    SmallVector<cuvm::CUMem, 256> free_blocks;
    get_blocks(free_blocks, blocks_to_map);
    assert(static_cast<int>(free_blocks.size()) == blocks_to_map);

    int i = 0;
    for (const block_range &br : to_map) {
      br.region->available.fill(br.begin, br.end, true);
      for (int b = br.begin; b < br.end; b++) {
        assert(i < static_cast<int>(free_blocks.size()));
        br.region->map_block(b, std::move(free_blocks[i++]));
      }
      free_mapped_.put(br.region->block_ptr(br.begin), (br.end - br.begin) * block_size_);
    }
    assert(i == blocks_to_map);
  }

  template <typename Collection>
  void get_blocks(Collection &out, int count) {
    if (count < 1)
      return;
    for (va_region &r : va_regions_) {
      if (r.available_blocks) {
        int block_idx = 0;
        for (int block_idx = r.available.find(true);
             block_idx < r.num_blocks() && count > 0;
             block_idx = r.available.find(true, block_idx+1)) {
          out.push_back(r.unmap_block(block_idx));
          char *ptr = r.block_ptr<char>(block_idx);
          free_mapped_.get_specific_block(ptr, ptr + block_size_);
          count--;
        }
      }
      if (!count)
        return;
    }
    while (count--)
      out.push_back(cuvm::CUMem::Create(block_size_, device_ordinal_));
  }


  va_region *va_find(CUdeviceptr dptr) {
    for (auto &region : va_regions_) {
      if (region.address_range.contains(dptr))
        return &region;
    }
    return nullptr;
  }

  va_region *va_find(void *ptr) {
    return va_find(reinterpret_cast<CUdeviceptr>(ptr));
  }



  coalescing_free_tree free_mapped_, free_va_;
  std::mutex  pool_lock_;
  std::mutex  mem_lock_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_USE_CUDA_VM_MAP

#endif  // DALI_CORE_MM_CUDA_VM_RESOURCE_H_
