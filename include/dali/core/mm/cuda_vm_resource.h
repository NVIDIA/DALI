// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/mm/cu_vm.h"

#if DALI_USE_CUDA_VM_MAP
#include "dali/core/bitmask.h"
#include "dali/core/device_guard.h"
#include "dali/core/format.h"
#include "dali/core/spinlock.h"
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/mm/pool_resource_base.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace mm {

namespace test {
class VMResourceTest;
}  // namespace test

class cuda_vm_resource : public memory_resource<memory_kind::device>,
                         public pool_resource_base<memory_kind::device> {
 public:
  explicit cuda_vm_resource(int device_ordinal = -1,
                            size_t block_size = 0,
                            size_t initial_va_size = 0) {
    device_ordinal_ = device_ordinal;
    if (block_size != 0 && !is_pow2(block_size))
      throw std::invalid_argument("block_size must be a power of 2");
    block_size_ = block_size;
    initial_va_size_ = initial_va_size;

    configure();
  }
  using mem_handle_t = CUmemGenericAllocationHandle;

  ~cuda_vm_resource() {
    purge();
  }

  void *try_allocate_from_free(size_t size, size_t alignment) override {
    if (size == 0)
      return nullptr;
    adjust_params(size, alignment, true);
    lock_guard pool_guard(pool_lock_);
    // try to get a free region that's already mapped
    void *ptr = try_get_mapped(size, alignment);
    if (ptr) {
      stat_add_allocation(size);
    }
    return ptr;
  }

  struct Stat {
    int allocated_blocks;
    int peak_allocated_blocks;
    int va_ranges;
    int peak_va_ranges;
    size_t allocated_va;
    size_t peak_va;
    size_t curr_allocated;
    size_t peak_allocated;
    size_t curr_free;
    size_t curr_allocations;
    size_t peak_allocations;
    size_t total_allocations;
    size_t total_deallocations;
    size_t total_unmaps;
  };

  size_t block_size() const noexcept {
    return block_size_;
  }

  int device_ordinal() const noexcept {
    return device_ordinal_;
  }

  Stat get_stat() const {
    lock_guard pool_guard(pool_lock_);
    return stat_;
  }

  void clear_stat() {
    lock_guard pool_guard(pool_lock_);
    stat_ = {};
  }

  void dump_stats(std::ostream &os) {
    print(os, "cuda_vm_resource stat dump:",
      "\ntotal VM size:         ", stat_.allocated_va,
      "\npeak VM size:          ", stat_.peak_va,
      "\n# VM ranges:           ", stat_.va_ranges,
      "\npeak # VM ranges:      ", stat_.peak_va_ranges,
      "\ncurrently allocated:   ", stat_.curr_allocated,
      "\npeak allocated:        ", stat_.peak_allocated,
      "\nallocated_blocks:      ", stat_.allocated_blocks,
      "\nblock size:            ", block_size_,
      "\nnon-freed allocations: ", stat_.curr_allocations,
      "\ntotal allocations:     ", stat_.total_allocations,
      "\ntotal deallocations:   ", stat_.total_deallocations,
      "\ntotal unmapping:       ", stat_.total_unmaps,
      "\nfree pool size:        ", stat_.curr_free,
      "\n");
  }

  void dbg_dump(std::ostream &os) {
    dump_stats(os);
    dump_regions(os);
  }

  void dump_regions(std::ostream &os) {
    print(os, "Pool map:\n");
    auto hex = [](auto x)->string {
      char buf[32];
      snprintf(buf, 32, "%016llX", (unsigned long long)x);  // NOLINT
      return buf;
    };
    for (auto &region : va_regions_) {
      print(os, "================================\nVA region ",
        hex(region.address_range.ptr()), " : ", hex(region.address_range.end()), "\n");
      for (int i = 0; i < region.num_blocks(); i++) {
        print(os, hex(region.block_dptr(i)), "  ",
          region.available[i] || !region.mapped[i] ? "       " : "In use ",
          region.mapped[i]                         ? "Mapped " : "       ",
          hex(region.mapping[i]),
          "\n");
      }
    }
  }

  /**
   * @brief Releases unused physical blocks
   *
   * Releases physical blocks that are currently allocated, but fully available.
   */
  void release_unused() override {
    do_release_unused(false);
  }

 protected:
  std::pair<size_t, size_t> do_release_unused(bool release_va) {
    size_t mem_freed = 0, va_freed = 0;
    std::vector<cuvm::CUMem> blocks_to_free;
    {
      std::unique_lock pool_guard(pool_lock_, std::try_to_lock);
      std::unique_lock mem_guard(mem_lock_, std::try_to_lock);

      ptrdiff_t num_blocks_to_free = 0;
      for (auto &region : va_regions_) {
        ptrdiff_t start = 0, end = 0;
        for (;; start = end + 1) {
          start = region.available.find(true, start);
          end = region.available.find(false, start + 1);
          if (end <= start)
            break;
          num_blocks_to_free += end - start;
        }
      }
      blocks_to_free.reserve(num_blocks_to_free);

      [&]() noexcept {  // this code mustn't throw!
        for (auto &region : va_regions_) {
          ptrdiff_t start = 0, end = 0;
          for (;; start = end + 1) {
            start = region.available.find(true, start);
            end = region.available.find(false, start + 1);
            if (end <= start)
              break;
            auto *start_ptr = region.block_ptr<char>(start);
            auto *end_ptr = region.block_ptr<char>(end);
            for (ptrdiff_t i = start; i < end; i++) {
              blocks_to_free.push_back(region.unmap_block(i));
              stat_.total_unmaps++;
              mem_freed += block_size_;
            }
            free_mapped_.get_specific_block(start_ptr, end_ptr);
          }
        }
      }();
      assert(static_cast<ptrdiff_t>(blocks_to_free.size()) == num_blocks_to_free);
      stat_.allocated_blocks -= blocks_to_free.size();
    }
    blocks_to_free.clear();  // free the physical memory
    if (release_va)
      va_freed = release_unused_va();
    return { mem_freed, va_freed };
  }

 protected:
  void do_deallocate(void *ptr, size_t size, size_t alignment) override {
    deallocate_impl(ptr, size, alignment, true);
  }

  int find_va_region(CUdeviceptr ptr) {
    for (size_t i = 0; i < va_regions_.size(); i++)
      if (va_regions_[i].address_range.contains(ptr))
        return i;
    return -1;
  }

  size_t release_unused_va() {
    struct RangeDesc {
      int range_idx, region_idx;
      int start_block, end_block;
    };
    std::vector<cuvm::CUMem> blocks_to_free;
    std::vector<cuvm::CUMemAddressRange> va_ranges_to_free;
    std::vector<RangeDesc> ranges_to_free;
    blocks_to_free.reserve(stat_.allocated_blocks);
    va_ranges_to_free.reserve(va_ranges_.size());
    ranges_to_free.reserve(va_ranges_.size());
    size_t va_freed = 0;

    {
      std::unique_lock pool_guard(pool_lock_, std::try_to_lock);
      std::unique_lock mem_guard(mem_lock_, std::try_to_lock);

      // the whole block below musn't throw or it'll leave the resource in an inconsistent state
      [&]() noexcept {
        for (int i = 0, n = va_ranges_.size(); i < n; i++) {
          auto start = va_ranges_[i].ptr();
          auto end = va_ranges_[i].end();
          if (free_va_.contains(reinterpret_cast<void*>(start), reinterpret_cast<void*>(end))) {
            int region_idx = find_va_region(start);
            assert(region_idx >= 0);
            va_region &region = va_regions_[region_idx];

            ptrdiff_t offset = start - region.address_range.ptr();
            assert(offset % block_size_ == 0);
            int num_blocks = (end - start) / block_size_;
            int start_block = offset / block_size_;  // index of the first block
            int end_block = start_block + num_blocks;  // index of the last block

            int avail = region.available.find(true, start_block);
            while (avail < end_block) {
              int end_avail = std::min<int>(region.available.find(false, avail + 1), end_block);
              for (int b = avail; b < end_avail; b++) {
                blocks_to_free.push_back(region.unmap_block(b));
                stat_.total_unmaps++;
              }
              auto *start_ptr = region.block_ptr<char>(avail);
              auto *end_ptr = region.block_ptr<char>(end_avail);
              free_mapped_.get_specific_block(start_ptr, end_ptr);

              if (end_avail == end_block)
                break;
              avail = region.available.find(true, end_avail + 1);
            }

            ranges_to_free.push_back({ i, region_idx, start_block, end_block });
          }
        }
        stat_.allocated_blocks -= blocks_to_free.size();
      }();

      for (const RangeDesc &rd : ranges_to_free) {
        auto &region = va_regions_[rd.region_idx];

        // We split the va_region as follows:

        // |<--------------- old region ---------------------->|
        // |<---region-->|<--va_range_to_remove-->|<---tail--->|
        //               ^                        ^
        // start_block---^                        ^---- end_block

        // If start_block is 0 then `region` will become empty and we can either
        // overwrite it with tail (if not empty) or remove it altogether

        va_region tail = region.split(rd.end_block);
        region.resize(rd.start_block);

        auto *start_ptr = region.block_ptr<char>(rd.start_block);
        auto *end_ptr = region.block_ptr<char>(rd.end_block);
        ptrdiff_t range_size = (end_ptr - start_ptr);

        if (region.empty()) {
          if (tail.empty())
            va_regions_.erase(va_regions_.begin() + rd.region_idx);
          else
            region = std::move(tail);
        } else if (!tail.empty()) {
          va_regions_.push_back(std::move(tail));
        }

        va_ranges_to_free.push_back(std::move(va_ranges_[rd.range_idx]));
        va_ranges_.erase(va_ranges_.begin() + rd.range_idx);
        stat_.allocated_va -= range_size;
        va_freed += range_size;
        void *p = free_va_.get_specific_block(start_ptr, end_ptr);
        assert(p != nullptr);
        (void)p;  // for non-debug builds
      }

      stat_.va_ranges -= va_ranges_to_free.size();
    }

    blocks_to_free.clear();
    va_ranges_to_free.clear();
    return va_freed;
  }

  void *do_allocate(size_t size, size_t alignment) override {
    if (size == 0)
      return nullptr;
    if (size > total_mem_)
      throw CUDABadAlloc(size);  // this can't succeed - no need to even try
    adjust_params(size, alignment, true);
    std::unique_lock<pool_lock_t> pool_guard(pool_lock_);
    // try to get a free region that's already mapped
    void *ptr = try_get_mapped(size, alignment);
    if (ptr) {
      stat_add_allocation(size);
      return ptr;
    }
    DeviceGuard dg(device_ordinal_);
    mem_lock_guard mem_guard(mem_lock_);
    void *va = get_va(size, alignment);
    try {
      map_storage(va, size);
    } catch (const CUDABadAlloc &e) {
      free_va_.put(va, size);
      throw CUDABadAlloc(size);
    } catch (...) {
      free_va_.put(va, size);
      throw;
    }
    ptr = free_mapped_.get_specific_block(va, size);
    assert(ptr == va);
    stat_take_free(size);
    mark_as_unavailable(ptr, size);
    stat_add_allocation(size);
    return ptr;
  }

 private:
  void purge() {
    lock_guard pool_guard(pool_lock_);
    mem_lock_guard mem_guard(mem_lock_);
    for (auto &r : va_regions_)
      r.purge();
    va_ranges_.clear();
  }

  void configure() {
    if (device_ordinal_ < 0) {
      CUDA_CALL(cudaGetDevice(&device_ordinal_));
    }
    DeviceGuard dg(device_ordinal_);
    if (total_mem_ == 0) {
      CUdevice device;
      CUDA_CALL(cuDeviceGet(&device, device_ordinal_));
      CUDA_CALL(cuDeviceTotalMem(&total_mem_, device));
    }
    if (block_size_ == 0) {
      size_t grain = cuvm::GetAddressGranularity();
      block_size_ = std::max(grain,  // at least grain, for correctness
                             std::min<size_t>(next_pow2(total_mem_ >> 8),  // capacity-dependent...
                                              64 << 20));  // ...but capped at 64 MiB
    }
    if (initial_va_size_ == 0)
        initial_va_size_ = align_up(2 * total_mem_, block_size_);  // get 2x physical size of VA
  }

  struct va_region {
    va_region(cuvm::CUAddressRange range, size_t block_size)
    : address_range(range), block_size(block_size) {
      size_t size = address_range.size();
      assert(size % block_size == 0);
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

    bool empty() const {
      return num_blocks() == 0;
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
      assert(mapping[block_idx] == CUmemGenericAllocationHandle{});
      assert(!mapped[block_idx]);
      assert(!available[block_idx]);
      cuvm::Map(block_dptr(block_idx), mem);
      mapping[block_idx] = mem.release().first;
      mapped[block_idx] = true;
      available[block_idx] = true;
      available_blocks++;
      assert(available_blocks >= 0 && available_blocks <= num_blocks());
    }

    cuvm::CUMem unmap_block(int block_idx) {
      assert(available[block_idx]);
      cuvm::Unmap(block_dptr(block_idx), block_size);
      cuvm::CUMem mem({mapping[block_idx], block_size });
      mapping[block_idx] = {};
      mapped[block_idx] = false;
      available[block_idx] = false;
      available_blocks--;
      assert(available_blocks >= 0 && available_blocks <= num_blocks());
      return mem;
    }

    /**
     * @brief Adds the contents of `other` at the end of this region.
     *
     * @note This function operates purely on metadata and doesn't affect the process memory map.
     */
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

    /**
     * @brief Trims the current region to `tail_start` blocks and returns the rest as a new region.
     *
     * @note This function operates purely on metadata and doesn't affect the process memory map.
     *
     * @param tail_start  The index of the first block to be moved to `tail`
     * @return va_region  The region starting at `tail_start`.
     */
    va_region split(int tail_start) {
      if (tail_start == 0) {
        va_region ret = std::move(*this);
        available_blocks = 0;
        address_range = {};
        return ret;
      } else if (tail_start == num_blocks()) {
        return va_region({}, block_size);
      }
      int tail_blocks = num_blocks() - tail_start;
      cuvm::CUAddressRange tail_range(block_dptr(tail_start), tail_blocks * block_size);
      va_region tail(tail_range, block_size);
      int n = num_blocks();
      for (int src = tail_start, dst = 0; src < n; src++, dst++) {
        tail.mapping[dst] = std::move(available[src]);
        tail.mapped[dst] = mapped[src];

        bool avail = available[src];
        tail.available[dst] = avail;
        if (avail) {  // update the available block count
          tail.available_blocks++;
          available_blocks--;
        }
      }
      resize(tail_start);
      return tail;
    }

    /**
     * @brief Changes the size of the region.
     *
     * If the new size is smaller than the old one and there are any blocks mapped at indices
     * that would become out of range, they are unmapped and deallocated.
     *
     * @note This function may change the process memory map.
     *
     * @note This function affects just the region and doesn't adjust the free_va / free_mapped
     *       in the enclosing VM resource. These need to be adjusted by the caller.
     */
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
        assert(available_blocks >= 0 && available_blocks <= new_num_blocks);
      }
      mapping.resize(new_num_blocks);
      mapped.resize(new_num_blocks);
      available.resize(new_num_blocks);
      address_range.size() = new_num_blocks * block_size;
    }

    void set_block_availability(char *start, char *end, bool availability) {
      assert(detail::is_aligned(start, block_size));
      assert(detail::is_aligned(end,   block_size));
      int start_blk = (start - block_ptr<char>(0)) / block_size;
      int end_blk   = (end   - block_ptr<char>(0)) / block_size;
      int flipped = 0;
      for (int b = available.find(!availability, start_blk); b < end_blk; ) {
        int first_flipped = std::min<int>(available.find(availability, b+1), end_blk);
        flipped += first_flipped - b;
        if (first_flipped < end_blk)
          b = std::min<int>(available.find(!availability, first_flipped+1), end_blk);
        else
          break;
      }
      if (availability)
        available_blocks += flipped;
      else
        available_blocks -= flipped;
      assert(available_blocks >= 0 && available_blocks <= num_blocks());
      available.fill(start_blk, end_blk, availability);
    }

    cuvm::CUAddressRange    address_range;
    vector<mem_handle_t>    mapping;
    bitmask                 mapped, available;
    ptrdiff_t               available_blocks = 0;
    size_t                  block_size = 0;
  };

  std::vector<va_region> va_regions_;
  std::vector<cuvm::CUMemAddressRange> va_ranges_;
  bitmask available_backup_;  // made a member to avoid unnecessary allocations
  size_t initial_va_size_ = 0;
  size_t block_size_ = 0;
  size_t total_mem_ = 0;
  int device_ordinal_ = -1;

  void adjust_params(size_t &size, size_t &alignment, bool check) {
    alignment = std::max(alignment, next_pow2(size >> 11));
    alignment = std::min(alignment, block_size_);
    size_t aligned = align_up(size, alignment);
    if (check && aligned < size)
      throw std::bad_alloc();
    size = aligned;
  }

  void *try_get_mapped(size_t size, size_t alignment) {
    char *ptr = static_cast<char*>(free_mapped_.get(size, alignment));
    if (ptr) {
      stat_take_free(size);
      auto *va = free_va_.get_specific_block(ptr, size);
      (void)va;
      assert(va == ptr);
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
    try {
      va_allocate(size);
    } catch (std::bad_alloc &) {
      print(std::cerr, "Could not allocate virtual address space of size ", size, " on device ",
        device_ordinal_);
      dbg_dump(std::cerr);
      throw;
    }
    ptr = free_va_.get(size, alignment);
    assert(ptr);
    return ptr;
  }

  /**
   * @brief Allocate virtual address space of at at least `min_size` bytes.
   *
   * The function hints the driver to use a distinct address space for each device in the
   * attempt to have a contiguous address spaces for each device. Currently, the spacing
   * between device VA spaces is 1 TiB and initial VA size for each device is double the physical
   * size.
   * On platforms that restrict VA size, if the reservation, fails there are more attempts to
   * allocate the size that's large enough to accommodate the requested size.
   */
  void va_allocate(size_t min_size) {
    size_t va_size = std::max(next_pow2(min_size), initial_va_size_);

    cuvm::CUMemAddressRange va;

    struct Hint {
      CUdeviceptr address;
      size_t      alignment;
    };
    SmallVector<Hint, 4> hints;

    if (va_regions_.empty()) {
      // Calculate the alignment for the initial allocations for this device - we start from
      // 4 TiB and go down.
      // The address hint is not important.
      hints = {
        { 0_zu, 1_zu << 42 },
        { 0_zu, 1_zu << 40 },
        { 0_zu, 1_zu << 38 },
        { 0_zu, 1_zu << 36 }
      };
    } else {
      // Try to allocate after the last VA for this device or before the first - calculate
      // the address hint but ignore alignment.
      auto &first_va = va_regions_.front();
      auto &last_va = va_regions_.back();
      assert(!va_ranges_.empty());
      va_size = std::max(va_size, 2 * va_ranges_.back().size());
      hints = {
        { last_va.address_range.end(), block_size_ },
        { first_va.address_range.ptr() - va_size, block_size_ }
      };
    }

    while (!va) {
      // Try to allocate at hinted locations...
      for (auto hint : hints) {
        try {
          va = cuvm::CUMemAddressRange::Reserve(va_size, hint.alignment, hint.address);
          break;
        } catch (const CUDAError &) {
        } catch (const std::bad_alloc &) {}
      }
      if (!va) {
        // ...hint failed - allocate anywhere, just align to block_size_
        auto on_error = [&](auto &exception) {
          size_t next_va_size;
          next_va_size = std::max(align_up(min_size, block_size_), va_size >> 1);
          if (next_va_size == va_size) {  // we're already as low as we can - rethrow
            if (!release_unused_va())
              throw exception;
          }
          va_size = next_va_size;
        };

        try {
          va = cuvm::CUMemAddressRange::Reserve(va_size, block_size_, 0);
          break;
        } catch (const CUDAError &e) {
          on_error(e);
        } catch (const std::bad_alloc &e) {
          on_error(e);
        }
      }
    }


    if (va_size < initial_va_size_)
        initial_va_size_ = va_size;

    if (!mm::detail::is_aligned(detail::u2ptr(va.ptr()), block_size_))
      throw std::logic_error("The VA region is not aligned to block size!\n"
        "This should never happen.");

    va_ranges_.push_back(std::move(va));
    va_add_range(va_ranges_.back());
    stat_va_add(va_size);
  }

  /**
   * @brief Add a memory region that spans the given VA range and merge it with adjacent
   *        regions, if found.
   */
  void va_add_range(cuvm::CUAddressRange va) {
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
        va_regions_.erase(va_regions_.begin() + i);
        break;
      }
    }

    // 3. Place the virtual address range in the VA free tree
    free_va_.put(reinterpret_cast<void*>(va.ptr()), va.size());
  }

  void deallocate_impl(void *ptr, size_t size, size_t alignment, bool lock) {
    if (size == 0)
      return;
    adjust_params(size, alignment, false);
    std::unique_lock<pool_lock_t> pool_guard(pool_lock_, std::defer_lock);
    if (lock)
      pool_guard.lock();
    stat_add_deallocation(size);
    free_mapped_.put(ptr, size);
    free_va_.put(ptr, size);
    stat_add_free(size);
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
    assert(free_mapped_.contains(cptr, cptr + size));

    char *start = nullptr, *end = nullptr;
    if (!is_block_aligned(cptr)) {
      auto block = block_bounds(cptr);
      start = free_mapped_.contains(block.first, block.second)
              ? block.first : block.second;
      // If the entire block is free then include it in the range - otherwise point
      // to the beginning of the next block.
    } else {
      start = cptr;
    }
    if (!is_block_aligned(cptr + size)) {
      auto block = block_bounds(cptr + size);
      end = free_mapped_.contains(block.first, block.second)
              ? block.second : block.first;
      // If the entire block is free then include it in this range - otherwise
      // point to the end of the previous block.
    } else {
      end = cptr + size;
    }
    // It's possible that at this point end < start. It can happen if both ends of the range
    // (ptr, ptr + size) lie within one block and that block is not available - in this case
    // `start` will point to the start of next block and `end` will point to end of the
    // previous block, so they will become swapped. This is handled properly by
    // set_block_availability.
    set_block_availability(start, end, true);
  }

  /**
   * @brief Mark all blocks that overlap the given range as unavailable
   */
  void mark_as_unavailable(void *ptr, size_t size) {
    char *cptr = static_cast<char *>(ptr);
    char *start = detail::align_ptr_down(cptr, block_size_);
    char *end = detail::align_ptr(cptr + size, block_size_);
    set_block_availability(start, end, false);
  }


  void set_block_availability(char *start, char *end, bool availability) {
    if (end <= start)  // NOTE it's possible that end < start - see mark_as_available
      return;
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
        break;  // everything we need is mapped
      block_idx = next_unmapped;
      int next_mapped = region->mapped.find(true, next_unmapped + 1);
      int next = std::min(end_block_idx, next_mapped);
      blocks_to_map += next - next_unmapped;
      block_idx = next;
      to_map.push_back({ next_unmapped, block_idx });
    }

    // a list of blocks to unmap
    SmallVector<std::pair<va_region*, int>, 256> to_unmap;
    {
      // Prevent `get_free_blocks` from unmapping blocks that are already mapped to this range.
      available_backup_ = region->available;
      int num_avail_backup = region->available_blocks;
      mark_as_unavailable(ptr, size);

      // Let's find some free blocks which we can reuse.
      get_free_blocks(to_unmap, blocks_to_map);

      // Restore original availability mask
      region->available = available_backup_;
      region->available_blocks = num_avail_backup;
    }

    int new_blocks = blocks_to_map - to_unmap.size();
    SmallVector<cuvm::CUMem, 256> free_blocks;
    free_blocks.resize(blocks_to_map);
    for (int i = to_unmap.size(); i < blocks_to_map; i++) {
      // This can throw, but we should leave the object in a consistent state
      free_blocks[i] = cuvm::CUMem::Create(block_size_, device_ordinal_);
    }

    int i = 0;
    for (auto &region_block : to_unmap) {
      // Each iteration should leave the resource in a consistent state.
      va_region *r = region_block.first;
      int block_idx = region_block.second;
      auto mem = r->unmap_block(block_idx);  // This can (potentially) throw
      stat_.total_unmaps++;
      char *block_ptr = r->block_ptr<char>(block_idx);
      free_mapped_.get_specific_block(block_ptr, block_size_);
      stat_take_free(block_size_);
      free_blocks[i++] = std::move(mem);
    }
    assert(free_blocks.size() == static_cast<size_t>(blocks_to_map));

    stat_allocate_blocks(new_blocks);

    i = 0;
    for (const block_range &br : to_map) {
      for (int b = br.begin; b < br.end; b++) {
        assert(i < static_cast<int>(free_blocks.size()));
        region->map_block(b, std::move(free_blocks[i++]));
      }
      size_t range_size = (br.end - br.begin) * block_size_;
      free_mapped_.put(region->block_ptr(br.begin), range_size);
      stat_add_free(range_size);
    }
    mark_as_available(ptr, size);
    assert(free_mapped_.contains(ptr, static_cast<char*>(ptr) + size));
    assert(i == blocks_to_map);
  }

  /**
   * @brief Obtains up to `count` physical storage blocks from available ones.
   *
   * Obtains `count` physical storage blocks and places their source regions and indices in `out`.
   * The are obtained either by unmapping existing mapped but free blocks.
   */
  template <typename Collection>
  void get_free_blocks(Collection &out, int count) {
    if (count < 1)
      return;
    for (va_region &r : va_regions_) {
      if (r.available_blocks) {
        for (int block_idx = r.available.find(true);
             block_idx < r.num_blocks() && count > 0;
             block_idx = r.available.find(true, block_idx+1)) {
          out.push_back({ &r, block_idx });
          count--;
        }
      }
      if (!count)
        return;
    }
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
  using pool_lock_t = std::mutex;
  using mem_lock_t = std::mutex;
  mutable pool_lock_t pool_lock_;
  mutable mem_lock_t mem_lock_;
  using lock_guard = std::lock_guard<pool_lock_t>;
  using mem_lock_guard = std::lock_guard<mem_lock_t>;

  Stat stat_ = {};
  void stat_add_allocation(size_t size) {
    Stat &s = stat_;
    s.total_allocations++;
    s.curr_allocations++;
    if (s.curr_allocations > s.peak_allocations)
      s.peak_allocations = s.curr_allocations;
    s.curr_allocated += size;
    if (s.curr_allocated > s.peak_allocated)
      s.peak_allocated = s.curr_allocated;
  }

  void stat_add_deallocation(size_t size) {
    stat_.curr_allocations--;
    stat_.total_deallocations++;
    stat_.curr_allocated -= size;
    assert(static_cast<ssize_t>(stat_.curr_allocated) >= 0);
  }

  void stat_add_free(size_t count) {
    stat_.curr_free += count;
    assert(static_cast<ssize_t>(stat_.curr_free) >= 0);
  }

  void stat_take_free(size_t count) {
    stat_.curr_free -= count;
    assert(static_cast<ssize_t>(stat_.curr_free) >= 0);
  }

  void stat_allocate_blocks(int count) {
    stat_.allocated_blocks += count;
    if (stat_.allocated_blocks > stat_.peak_allocated_blocks)
      stat_.peak_allocated_blocks = stat_.allocated_blocks;
  }

  void stat_va_add(size_t size) {
    stat_.allocated_va += size;
    if (stat_.allocated_va > stat_.peak_va)
      stat_.peak_va = stat_.allocated_va;
    stat_.va_ranges++;
    if (stat_.va_ranges > stat_.peak_va_ranges)
      stat_.peak_va_ranges = stat_.va_ranges;
  }
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_USE_CUDA_VM_MAP

#endif  // DALI_CORE_MM_CUDA_VM_RESOURCE_H_
