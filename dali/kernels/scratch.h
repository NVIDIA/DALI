// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SCRATCH_H_
#define DALI_KERNELS_SCRATCH_H_

#include <array>
#include <cassert>
#include <utility>
#include <type_traits>
#include "dali/core/static_switch.h"
#include "dali/core/mm/memory.h"
#include "dali/core/mm/memory_kind.h"
#include "dali/kernels/context.h"

namespace dali {
namespace kernels {

class BumpAllocator {
 public:
  BumpAllocator() = default;
  BumpAllocator(char *mem, size_t total) : memory_(mem), total_(total), used_(0) {}
  BumpAllocator(BumpAllocator &&a) {
    *this = std::move(a);
  }

  ~BumpAllocator() = default;

  inline BumpAllocator &operator=(BumpAllocator &&a) {
    memory_ = a.memory_;
    total_ = a.total_;
    used_ = a.used_;
    a.memory_ = nullptr;
    a.total_ = 0;
    a.used_ = 0;
    return *this;
  }

  inline char *alloc(size_t elements) {
    AssertAvail(elements);
    char *p = next();
    used_ += elements;
    return p;
  }

  inline char *next() const { return memory_ + used_; }

  inline size_t avail() const { return total_ - used_; }
  inline size_t total() const { return total_; }
  inline size_t used() const { return used_; }

  inline void AssertAvail(size_t required) {
    assert(used_ + required <= total_);
  }

  /**
   * @brief Resets the usage counter so the buffer can be reused.
   */
  inline void Clear() {
    used_ = 0;
  }

 private:
  char *memory_ = nullptr;
  size_t total_ = 0;
  size_t used_ = 0;
};

/**
 * @brief Scratchpad with pre-existing buffers
 */
struct PreallocatedScratchpad : Scratchpad {
  PreallocatedScratchpad() = default;

  explicit PreallocatedScratchpad(
      std::array<BumpAllocator, size_t(mm::memory_kind_id::count)> &&allocs)
  : allocs(std::move(allocs)) {}

  void Clear() {
    for (auto &a : allocs) {
      a.Clear();
    }
  }

  void *Alloc(mm::memory_kind_id kind_id, size_t bytes, size_t alignment) override {
    if (bytes == 0)
      return nullptr;
    assert((size_t)kind_id < allocs.size());
    auto &A = allocs[(size_t)kind_id];
    uintptr_t ptr = reinterpret_cast<uintptr_t>(A.next());
    // Calculate the padding needed to satisfy alignmnent requirements
    uintptr_t padding = (alignment-1) & (-ptr);
    (void)A.alloc(padding);
    return A.alloc(bytes);
  }

  std::array<BumpAllocator, size_t(mm::memory_kind_id::count)> allocs;
};


/**
 * @brief Implements an ever-growing scratchpad
 */
class ScratchpadAllocator {
 public:
  static constexpr size_t NumMemKinds = static_cast<size_t>(mm::memory_kind_id::count);

  /**
   * @brief Describes scratch memory allocation policy
   *
   * When reserving `size` memory and the existing capacity is `capacity`
   * then the new allocation will be of size:
   * ```
   * new_capacity = max(size * (1 + Margin), capacity * GrowthRatio)
   * ```
   */
  struct AllocPolicy {
    /**
     * When reserving more memory than available, current capacity will
     * be multiplied by this value.
     */
    float GrowthRatio = 2;

    /**
     * When reserving memory, make sure that at least `(1 + Margin) * size` is
     * actually allocated.
     */
    float Margin = 0.1;
  };

  /**
   * @brief Returns reference to the current allocation policy for a given memory kind.
   */
  template <typename MemoryKind>
  AllocPolicy &Policy() { return Policy(mm::kind2id_v<MemoryKind>); }

  /**
   * @brief Returns reference to the current allocation policy for a given memory kind.
   */
  template <typename MemoryKind>
  const AllocPolicy &Policy() const { return Policy(mm::kind2id_v<MemoryKind>); }


  /**
   * @brief Returns allocation policy for given allocation type
   */
  AllocPolicy &Policy(mm::memory_kind_id kind) {
    return buffers_[static_cast<int>(kind)].policy;
  }

  const AllocPolicy &Policy(mm::memory_kind_id kind) const {
    return buffers_[static_cast<int>(kind)].policy;
  }

  /**
   * @brief Releases any storage allocated by calls to `Reserve`.
   * @remarks Scratchpad returned by `GetScratchpad` is invalid after this call.
   */
  void Free() {
    for (auto &buffer : buffers_) {
      buffer.mem.reset();
      buffer.capacity = 0;
      buffer.padding = 0;
    }
  }

  /**
   * @brief Reserves memory for all allocation types.
   *
   * See `Reserve<MemoryKind>(size_t)` for details.
   */
  void Reserve(std::array<size_t, NumMemKinds> sizes) {
    for (size_t idx = 0; idx < NumMemKinds; idx++) {
      Reserve(mm::memory_kind_id(idx), sizes[idx]);
    }
  }

  /**
   * @brief Ensures that at least `size` bytes of memory are available in storage `type`
   * @remarks If reallocation happens, any `Scratchpad` returned by `GetScratchpad`
   *          is invalidated.
   */
  template <typename MemoryKind>
  void Reserve(size_t size) {
    size_t index = static_cast<size_t>(mm::kind2id_v<MemoryKind>);
    auto &buf = buffers_[index];

    constexpr size_t alignment = 64;
    size_t new_capacity = buf.capacity;
    if (buf.capacity < size) {
      new_capacity = buf.capacity * buf.policy.GrowthRatio;
      size_t size_with_margin = size * (1 + buf.policy.Margin);
      if (size_with_margin > new_capacity)
        new_capacity = size_with_margin;
    }

    if (new_capacity != buf.capacity) {
      buf.mem.reset();
      buf.mem = mm::alloc_raw_unique<char, MemoryKind>(new_capacity + alignment);
      uintptr_t ptr = reinterpret_cast<uintptr_t>(buf.mem.get());
      size_t padding = (alignment-1) & (-ptr);
      buf.capacity = new_capacity + alignment - padding;
      buf.padding = padding;
    }
  }

  /**
   * @brief Ensures that at least `size` bytes of memory are available in storage `type`
   * @remarks For use ONLY inside loops or when the use of run-time dispatch of the memory
   *          kind is justified. Otherwise, use the overload with type argument MemoryKind.
   */
  void Reserve(mm::memory_kind_id id, size_t size) {
    TYPE_SWITCH(id, mm::kind2id, MemoryKind, (mm::memory_kind::host, mm::memory_kind::pinned,
                                              mm::memory_kind::device, mm::memory_kind::managed), (
       Reserve<MemoryKind>(size);
    ), (assert(!"Incorrect memory kind")));  // NOLINT
  }

  /**
   * @brief Returns allocator's capacities for all allocation types
   */
  std::array<size_t, NumMemKinds> Capacities() const noexcept {
    std::array<size_t, NumMemKinds> capacities;
    for (size_t i = 0; i < buffers_.size(); i++)
      capacities[i] = buffers_[i].capacity;
    return capacities;
  }

  /**
   * @brief Returns allocator's capacity for given memory kind
   */
  size_t Capacity(mm::memory_kind_id kind_id) const noexcept {
    return buffers_[static_cast<size_t>(kind_id)].capacity;
  }

  /**
   * @brief Returns allocator's capacity for given memory kind
   */
  template <typename MemoryKind>
  size_t Capacity() const noexcept {
    return Capacity(mm::kind2id_v<MemoryKind>);
  }

  /**
   * @brief Returns a scratchpad.
   * @remarks The returned scratchpad is invalidated by desctruction of this
   *          object or by subsequent calls to `Reserve` or `Free`.
   */
  PreallocatedScratchpad GetScratchpad() {
    PreallocatedScratchpad scratchpad;
    for (size_t idx = 0; idx < NumMemKinds; idx++) {
      auto &buf = buffers_[idx];
      scratchpad.allocs[idx] = { buf.mem.get() + buf.padding, buf.capacity };
    }
    return scratchpad;
  }

 private:
  struct Buffer {
    mm::uptr<char> mem;
    size_t capacity = 0, padding = 0;
    AllocPolicy policy = {};
  };
  std::array<Buffer, NumMemKinds> buffers_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SCRATCH_H_
