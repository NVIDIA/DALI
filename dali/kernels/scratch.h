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

#ifndef DALI_KERNELS_SCRATCH_H_
#define DALI_KERNELS_SCRATCH_H_

#include <array>
#include <utility>
#include <type_traits>
#include "dali/kernels/alloc.h"
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

  /// @brief Resets the usage counter so the buffer can be reused.
  inline void Clear() {
    used_ = 0;
  }

 private:
  char *memory_ = nullptr;
  size_t total_ = 0;
  size_t used_ = 0;
};

struct PreallocatedScratchpad : Scratchpad {
  PreallocatedScratchpad() = default;

  explicit PreallocatedScratchpad(
      std::array<BumpAllocator,
      size_t(AllocType::Count)> &&allocs)
  : allocs(std::move(allocs)) {}

  void Clear() {
    for (auto &a : allocs) {
      a.Clear();
    }
  }

  void *Alloc(AllocType alloc, size_t bytes, size_t alignment) override {
    auto &A = allocs[(size_t)alloc];
    uintptr_t ptr = reinterpret_cast<uintptr_t>(A.next());
    // Calculate the padding needed to satisfy alignmnent requirements
    uintptr_t padding = (alignment-1) & (-ptr);
    (void)A.alloc(padding);
    return A.alloc(bytes);
  }

  std::array<BumpAllocator, size_t(AllocType::Count)> allocs;
};

/// @brief Implements an ever-growing scratchpad
class ScratchpadAllocator {
 public:
  static constexpr size_t NumAllocTypes = static_cast<size_t>(AllocType::Count);

  /// @brief Releases any storage allocated by calls to `Reserve`.
  /// @remarks Scratchpad returned by `GetScratchpad` is invalid after this call.
  void Free() {
    buffers_ = {};
  }

  void Reserve(std::array<size_t, NumAllocTypes> sizes) {
    for (size_t idx = 0; idx < NumAllocTypes; idx++) {
      Reserve(AllocType(idx), sizes[idx]);
    }
  }

  /// @brief Ensures that at least `sizes` bytes of memory are available in storage `type`
  /// @remarks If reallocation happens, any `Scratchpad` returned by `GetScratchpad`
  ///          is invalidated.
  void Reserve(AllocType type, size_t size) {
    size_t index = static_cast<size_t>(type);
    auto &buf = buffers_[index];
    if (buf.capacity < size) {
      constexpr size_t alignment = 64;
      buf.mem = memory::alloc_unique<char>(type, size + alignment);
      uintptr_t ptr = reinterpret_cast<uintptr_t>(buf.mem.get());
      size_t padding = (alignment-1) & (-ptr);
      buffers_[index].capacity = size + alignment - padding;
      buffers_[index].padding = padding;
    }
  }

  /// @brief Returns a scratchpad.
  /// @remarks The returned scratchpad is invalidated by desctruction of this
  ///          object or by subsequent calls to `Reserve` or `Free`.
  PreallocatedScratchpad GetScratchpad() {
    PreallocatedScratchpad scratchpad;
    for (size_t idx = 0; idx < NumAllocTypes; idx++) {
      auto &buf = buffers_[idx];
      scratchpad.allocs[idx] = { buf.mem.get() + buf.padding, buf.capacity };
    }
    return scratchpad;
  }

 private:
  struct Buffer {
    memory::KernelUniquePtr<char> mem;
    size_t capacity = 0, padding = 0;
  };
  std::array<Buffer, NumAllocTypes> buffers_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SCRATCH_H_
