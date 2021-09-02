// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_KERNEL_MANAGER_H_
#define DALI_KERNELS_KERNEL_MANAGER_H_

#include <cassert>
#include <memory>
#include <utility>
#include <atomic>
#include "dali/kernels/scratch.h"
#include "dali/kernels/context.h"
#include "dali/kernels/kernel_req.h"
#include "dali/core/small_vector.h"
#include "dali/core/mm/memory_kind.h"

namespace dali {
namespace kernels {

template <typename T>
T atomic_max(std::atomic<T> &value, const T &store_if_greater) {
  T old = value.load();
  for (;;) {
    if (!(store_if_greater > old))
      return old;

    if (value.compare_exchange_strong(old, store_if_greater))
      return store_if_greater;
  }
}

struct AnyKernelInstance {
  KernelRequirements requirements;
  std::unique_ptr<void, void(*)(void*)> instance = { nullptr, free };

  template <typename Kernel, typename... Args>
  Kernel &create_or_get(Args&&... args) {
    void (*deleter)(void *) = delete_kernel<Kernel>;
    if (!instance || instance.get_deleter() != deleter) {
      instance.reset();
      instance = { new Kernel{std::forward<Args>(args)...}, deleter };
    }
    return *static_cast<Kernel*>(instance.get());
  }

  template <typename Kernel>
  Kernel &get() {
    void (*deleter)(void *) = delete_kernel<Kernel>;
    if (!instance)
      throw std::logic_error("The kernel instance is null");
    if (instance.get_deleter() != deleter)
      throw std::logic_error("The kernel instance is of different type than requested");

    return *static_cast<Kernel*>(instance.get());
  }

  template <typename Kernel>
  static void delete_kernel(void *ptr) {
    delete static_cast<Kernel*>(ptr);
  }

  explicit operator bool() const noexcept { return static_cast<bool>(instance); }
};

/**
 * @brief Manages multiple instances of run-time typed kernels
 *
 * KernelManager provides type erasure for kernels whose type is selected at
 * run-time. Kernel manager also carries out mundane tasks of keeping
 * ScratchpadAllocators and reserving memory according to requirements returned
 * by kernel's Setup method.
 *
 * A scratchpad allocator is created per-thread with thread indexing supported
 * explicitly by the caller.
 */
class DLL_PUBLIC KernelManager {
 public:
  static constexpr size_t NumMemKinds = ScratchpadAllocator::NumMemKinds;
  using ScratchSizes = std::array<size_t, NumMemKinds>;

  /**
   * @brief Creates `num_threads` scratchpads and `num_instances` slots for kernels
   *
   * @param num_threads -    number of threads that can concurrently use the kernels in the
   *                         manager, assuming that each threads uses its unique
   *                         zero-based index
   * @param num_instances -  number of Kernel instances to be created; typically corresponds
   *                         to number of samples (for per-sample kernels) or minibatches
   */
  void Resize(size_t num_threads, size_t num_instances);

  /**
   * @brief Creates `num_threads` scratchpads and `num_instances` kernels of type Kernel
   *        constructed with `args...`.
   *
   * @param num_threads   -  number of threads that can concurrently use the kernels in the
   *                         manager, assuming that each threads uses its unique
   *                         zero-based index
   * @param num_instances -  number of Kernel instances to be created; typically corresponds
   *                         to number of samples (for per-sample kernels) or minibatches
   * @param args           - arguments passed to Kernel's constructor upon creation.
   * @tparam Kernel        - type of the kernel to be created
   */
  template <typename Kernel, typename... Args>
  void Resize(size_t num_threads, size_t num_instances, const Args&... args) {
    Resize(num_threads, num_instances);
    Initialize<Kernel>(args...);
  }

  /**
   * @brief Populates the instance slots with instances of a given Kernel
   *
   * @param args           - arguments passed to Kernel's constructor upon creation.
   * @tparam Kernel        - type of the kernel to be created
   */
  template <typename Kernel, typename... Args>
  void Initialize(const Args&... args) {
    for (size_t i = 0; i < NumInstances(); i++)
      CreateOrGet<Kernel>(i, args...);
  }

  /**
   * @brief Clears kernel instances and scratchpads
   */
  void Reset();

  /**
   * @brief Gets or creates a Kernel instance
   */
  template <typename Kernel, typename... ConstructorArgs>
  Kernel &CreateOrGet(int instance_idx, ConstructorArgs &&...args) {
    return instances[instance_idx].create_or_get<Kernel>(std::forward<ConstructorArgs>(args)...);
  }


  /**
   * @brief Gets a Kernel instance
   *
   * If there's no instance for a given index of the type is different,
   * `std::logic_error` is thrown.
   * @return A reference to a kernel instance at given index
   */
  template <typename Kernel>
  Kernel &Get(int instance_idx) {
    return instances[instance_idx].get<Kernel>();
  }

  /**
   * @brief Gets a reference to an internally maintained copy of KernelRequirements
   */
  KernelRequirements &GetRequirements(int instance_idx) noexcept {
    return instances[instance_idx].requirements;
  }

  /**
   * @brief Gets a const-reference to an internally maintained copy of KernelRequirements
   */
  const KernelRequirements &GetRequirements(int instance_idx) const noexcept {
    return instances[instance_idx].requirements;
  }

  size_t NumInstances() const noexcept { return instances.size(); }
  size_t NumThreads() const noexcept { return scratchpads.size(); }

  /**
   * @brief Gets a scratchpad allocator assigned to a given thread.
   */
  ScratchpadAllocator &GetScratchpadAllocator(int thread_idx) {
    return scratchpads[thread_idx];
  }

  /**
   * @brief Calls setup on specified kernel instance.
   *
   * @param instance_idx   - kernel instance index; typically corresponds
   *                         to sample index (for per-sample kernels) or minibatch index
   * @param context        - context for the kernel
   *                         * should contain valid CUDA stream for GPU kernels;
   * @param in_args        - pack of arguments (inputs, arguments) used in Kernel::Setup
   * @return Reference to internally maintained copy of the kernel requirements.
   * @remarks The copies of KernelRequirements for each instance index are used for allocating
   *          scratch memory. While the function returns non-const reference, please note
   *          that decreasing scratch sizes calculated by Setup will result in undefined
   *          behavior, including memory corruption or illegal access.
   */
  template <typename Kernel, typename... InArgs>
  KernelRequirements &Setup(int instance_idx, KernelContext &context, InArgs &&...in_args) {
    auto &inst = instances[instance_idx];
    inst.requirements = inst.get<Kernel>().Setup(context, std::forward<InArgs>(in_args)...);
    for (size_t i = 0; i < max_scratch_sizes.size(); i++) {
      atomic_max(max_scratch_sizes[i], inst.requirements.scratch_sizes[i]);
    }
    return inst.requirements;
  }

  /**
   * @brief Calls Run on specified kernel instance using Scratchpad for given thread.
   *
   * @param thread_idx     - zero-based thread index
   * @param instance_idx   - kernel instance index; typically corresponds
   *                         to sample index (for per-sample kernels) or minibatch index
   * @param context        - context for the kernel
   *                         * should contain valid CUDA stream for GPU kernels;
   *                         * scratchpad pointer is overriden with a scratchpad
   *                           created for given thread_idx
   * @param out_in_args    - pack of arguments (outputs, inputs, arguments) used in Kernel::Run
   */
  template <typename Kernel, typename... OutInArgs>
  void Run(int thread_idx, int instance_idx, KernelContext &context, OutInArgs &&...out_in_args) {
    assert(static_cast<size_t>(thread_idx) < scratchpads.size());
    auto &sa = GetScratchpadAllocator(thread_idx);
    Run<Kernel>(sa, instance_idx, context, std::forward<OutInArgs>(out_in_args)...);
  }

  /**
   * @brief Calls Run on specified kernel instance using Scratchpad for given thread.
   *
   * @param sa             - scratchpad allocator; memory will be reserved in it to satisfy
   *                         instance's requirements
   * @param instance_idx   - kernel instance index; typically corresponds
   *                         to sample index (for per-sample kernels) or minibatch index
   * @param context        - context for the kernel
   *                         * should contain valid CUDA stream for GPU kernels;
   *                         * scratchpad pointer is overriden with a scratchpad
   *                           created from `sa`
   * @param out_in_args    - pack of arguments (outputs, inputs, arguments) used in Kernel::Run
   */
  template <typename Kernel, typename... OutInArgs>
  void Run(ScratchpadAllocator &sa,
           int instance_idx,
           KernelContext &context,
           OutInArgs &&...out_in_args) {
    assert(instance_idx >= 0 &&
           static_cast<size_t>(instance_idx) < NumInstances() &&
           "Kernel instance index (instance_idx) out of range");
    auto &inst = instances[instance_idx];
    auto scratchpad = ReserveScratchpad(sa, inst.requirements.scratch_sizes);
    auto *old_scratchpad = context.scratchpad;
    context.scratchpad = &scratchpad;
    inst.get<Kernel>().Run(context, std::forward<OutInArgs>(out_in_args)...);
    context.scratchpad = old_scratchpad;
  }

  /**
   * @brief Makes sure ScratchpadAllocator can accommodate `sizes`
   *
   * @param sa     - scratchpad allocator to reserve
   * @param sizes  - requested minimum size
   *
   * The manager maintains a lifetime maximum of sizes requested.
   * If reallocation is necessary, it allocates `sizes` or that maximum
   * whichever is larger.
   */
  auto ReserveScratchpad(ScratchpadAllocator &sa, const ScratchSizes &sizes)->
  decltype(sa.GetScratchpad());

  /**
   * @brief Calls ReserveScratchpad on ScratchpadAllocator associated with given thread_idx
   */
  inline auto ReserveScratchpad(int thread_idx, const ScratchSizes &sizes) {
    return ReserveScratchpad(GetScratchpadAllocator(thread_idx), sizes);
  }

  /**
   * @brief Returns maximum scratchpad size seen so far
   */
  inline ScratchSizes MaxScratchSizes() const {
    ScratchSizes sizes;
    for (size_t i = 0; i < sizes.size(); i++) {
      sizes[i] = max_scratch_sizes[i];
    }
    return sizes;
  }

  /**
   * @brief Reserves scratchpad big enough to accommodate largest scratch area ever seen
   */
  inline auto ReserveMaxScratchpad(int thread_idx) {
    return ReserveScratchpad(thread_idx, MaxScratchSizes());
  }

  /**
   * @brief Sets a memory size hint for allocating scratchpad memory
   *
   * All calls to ScratchpadAllocator::Reserve followint this call will request at least
   * bytes memory for given allocation type.
   */
  template <typename MemoryKind>
  void SetMemoryHint(size_t bytes) {
    size_t alloc_idx = static_cast<size_t>(mm::kind2id_v<MemoryKind>);
    assert(alloc_idx < max_scratch_sizes.size());
    atomic_max(max_scratch_sizes[alloc_idx], bytes);
  }

 private:
  SmallVector<AnyKernelInstance, 1> instances;
  SmallVector<ScratchpadAllocator, 1> scratchpads;
  std::array<std::atomic_size_t, NumMemKinds> max_scratch_sizes{};
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_MANAGER_H_
