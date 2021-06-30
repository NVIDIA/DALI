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

#include <algorithm>
#include <utility>>
#include <vector>
#include "dali/core/mm/cu_vm.h"

#if DALI_USE_CUDA_VM_MAP
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"
#include "dali/core/bitmask.h"

namespace dali {
namespace mm {

namespace test {
class VMResourceTest;
}  // namespace test

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
    va_ranges_.clear();
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
    va_region(cuvm::CUAddressRange range, size_t block_size)
    : address_range(range), block_size(block_size) {
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
      for (size_t i = 0; i < mapping.size(); i++, ptr += block_size) {
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

    void append(va_region &&other) {
      assert(address_range.end() == other.address_range.ptr());
      assert(block_size == other.block_size);
      address_range.size() += other.address_range.size();
      int other_blocks = other.num_blocks();
      int old_blocks = num_blocks();
      mapping.resize(old_blocks + other_blocks);
      for (int i = 0; i < other_blocks; i++) {
        mapping[old_blocks + i] = other.mapping[i];
        other.mapping[i] = {};
      }
      other.mapping = {};

      mapped.append(other.mapped);
      available.append(other.available);

      other.available = {};
      other.mapped = {};
      available_blocks += other.available_blocks;
      other.available_blocks = 0;
      assert(mapping.size() == mapped.size());
      assert(mapping.size() == available.size());
    }

    void resize(int new_num_blocks) {
      if (new_num_blocks < num_blocks()) {
        int no_longer_in_range = 0;
        for (int i = new_num_blocks; i < num_blocks(); i++) {
          if (mapping[i]) {
            cuvm::Unmap(block_ptr(i), block_size);
            CUDA_DTOR_CALL(cuMemRelease(mapping[i]));
            mapping[i] = {};
            if (available[i])
              no_longer_in_range++;
          }
        }
        available_blocks -= no_longer_in_range;
      }
      mapping.resize(new_num_blocks);
      mapped.resize(new_num_blocks);
      available.resize(new_num_blocks);
      address_range.size() = new_num_blocks * block_size;
    }

    void set_block_availability(char *start, char *end, bool availability) {
      int blk_start = (start - block_ptr<char>(0)) / block_size;
      int blk_end   = (end   - block_ptr<char>(0)) / block_size;
      int flipped = 0;
      for (int b = available.find(!availability, blk_start); b < blk_end; ) {
        int first_flipped = available.find(availability, b+1);
        flipped += first_flipped - b;
        b = first_flipped + 1;
      }
      if (availability)
        available_blocks += flipped;
      else
        available_blocks -= flipped;
      available.fill(blk_start, blk_end, availability);
    }

    cuvm::CUAddressRange    address_range;
    vector<mem_handle_t>    mapping;
    bitmask                 mapped, available;
    ptrdiff_t               available_blocks = 0;
    size_t                  block_size = 0;
  };

  SmallVector<va_region, 8> va_regions_;
  SmallVector<cuvm::CUMemAddressRange, 8> va_ranges_;
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

  void *do_allocate(size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    // try to get a free region that's already mapped
    void *ptr = try_get_mapped(size, alignment);
    if (ptr)
      return ptr;
    ptr = get_va(size, alignment);
    map_storage(ptr, size);
    ptr = try_get_mapped(size, alignment);
    assert(ptr != nullptr);
    return ptr;
  }

  void *try_get_mapped(size_t size, size_t alignment) {
    char *ptr = static_cast<char*>(free_mapped_.get(size, alignment));
    if (ptr) {
      free_va_.get_specific_block(ptr, ptr + size);
      mark_as_unavailable(ptr, size);
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

  /**
   * @brief Allocate virtual address space of at at least `min_size` bytes.
   *
   * The function hints the driver to use a distinct address space for each device in the
   * attempt to have a contiguous address spaces for each device. Currently, the spacing
   * between device VA spaces is 1 TiB and initial VA size for each device is 4 GiB.
   */
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
      hint = last_va->address_range.end();
    }
    va_ranges_.push_back(cuvm::CUMemAddressRange::Reserve(va_size, 0, hint));
    auto &va = va_ranges_.back();
    std::cerr << "Allocated a virtual memory range at: " << (void*)va.ptr() << " - hint was "  // NOLINT
              << (void*)hint << std::endl;  // NOLINT

    va_add_region(va);
  }

  /**
   * @brief Add a memory region that spans the given VA range and merge it with adjacent
   *        regions, if found.
   */
  void va_add_region(cuvm::CUAddressRange va) {
    // Try to merge regions
    // 1. Find preceding region
    va_region *region = nullptr;
    for (auto &r : va_regions_)
      if (r.address_range.end() == va.ptr()) {
        region = &r;
        break;
      }
    if (region) {
      // Found! Resize it. The new VA range is unmapped, so we can simply resize,
      // there's nothing more to do.
      region->resize(region->num_blocks() + va.size() / block_size_);
    } else {
      // Not found - create a new region
      va_regions_.emplace_back(va, block_size_);
      region = &va_regions_.back();
    }

    // 2. Find succeeding region
    for (size_t i = 0; i < va_regions_.size(); i++) {
      auto &r = va_regions_[i];
      if (r.address_range.ptr() == region->address_range.end()) {
        // Found - now we append the old range (which follows the new one) to the new one...
        region->append(std::move(r));
        // ...and remove the old one
        va_regions_.erase_at(i);
        break;
      }
    }

    // 3. Place the virtual address range in the VA free tree
    free_va_.put(reinterpret_cast<void*>(va.ptr()), va.size());
  }

  void do_deallocate(void *ptr, size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    free_mapped_.put(ptr, size);
    free_va_.put(ptr, size);
    mark_as_available(ptr, size);
  }

  bool is_block_aligned(void *ptr) const noexcept {
    return detail::is_aligned(ptr, block_size_);
  }

  bool is_free_mapped(char *start, char *end) const noexcept {
    return free_mapped_.contains(start, end);
  }

  std::pair<char *, char *> block_bounds(char *ptr) const {
    char *start = detail::align_ptr_down(ptr, block_size_);
    char *end   = start + block_size_;
    return { start, end };
  }

  /**
   * @brief Mark all memory in the given range as available; check partially covered blocks.
   *
   * All blocks completely covered by the given memory range are marked as available.
   * The blocks which are only partially covered (at the beginning and at the end) are checked
   * in the free tree and marked accordingly. The function assumes that the free tree has
   * already been updated to reflect the availability.
   */
  void mark_as_available(void *ptr, size_t size) {
    char *cptr = static_cast<char *>(ptr);

    char *start = nullptr, *end = nullptr;
    if (!is_block_aligned(cptr)) {
      auto block = block_bounds(cptr);
      start = free_mapped_.contains(block.first, block.second)
              ? block.first : block.second;
    } else {
      start = cptr;
    }
    if (!is_block_aligned(cptr + size)) {
      auto block = block_bounds(cptr + size);
      end = free_mapped_.contains(block.first, block.second)
              ? block.second : block.first;
    } else {
      end = cptr + size;
    }

    set_block_avaialbility(start, end, true);
  }

  /**
   * @brief Mark all blocks that overlap the given range as unavailable
   */
  void mark_as_unavailable(void *ptr, size_t size) {
    char *cptr = static_cast<char *>(ptr);
    char *start = detail::align_ptr_down(cptr, block_size_);
    char *end = detail::align_ptr(cptr + size, block_size_);
    set_block_avaialbility(start, end, false);
  }


  void set_block_avaialbility(char *start, char *end, bool availability) {
    va_region *va = va_find(start);
    assert(va != nullptr);
    assert(va == va_find(end-1));
    va->set_block_availability(start, end, availability);
  }

  void map_storage(void *ptr, size_t size) {
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    auto *region = va_find(ptr);
    assert(region);
    CUdeviceptr region_start  = region->address_range.ptr();
    CUdeviceptr region_end    = region->address_range.end();
    ptrdiff_t offset = dptr - region_start;
    int start_block_idx = offset / block_size_;
    int end_block_idx = (offset + size + block_size_ - 1) / block_size_;
    struct block_range {
      int begin, end;
    };
    SmallVector<block_range, 8> to_map;
    int blocks_to_map = 0;
    for (int block_idx = start_block_idx; block_idx < end_block_idx; ) {
      int next_unmapped = region->mapped.find(false, block_idx);
      if (next_unmapped >= end_block_idx)
        break;  // everything we need is mapped)
      block_idx = next_unmapped;
      int next_mapped = region->mapped.find(true, next_unmapped + 1);
      int next = std::min(end_block_idx, next_mapped);
      blocks_to_map += next - next_unmapped;
      block_idx = next;
      to_map.push_back({ next_unmapped, block_idx });
    }

    SmallVector<cuvm::CUMem, 256> free_blocks;
    get_blocks(free_blocks, blocks_to_map);
    assert(static_cast<int>(free_blocks.size()) == blocks_to_map);

    int i = 0;
    for (const block_range &br : to_map) {
      region->available.fill(br.begin, br.end, true);
      for (int b = br.begin; b < br.end; b++) {
        assert(i < static_cast<int>(free_blocks.size()));
        region->map_block(b, std::move(free_blocks[i++]));
      }
      free_mapped_.put(region->block_ptr(br.begin), (br.end - br.begin) * block_size_);
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

  friend class test::VMResourceTest;

  coalescing_free_tree free_mapped_, free_va_;
  std::mutex  pool_lock_;
  std::mutex  mem_lock_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_USE_CUDA_VM_MAP

#endif  // DALI_CORE_MM_CUDA_VM_RESOURCE_H_
