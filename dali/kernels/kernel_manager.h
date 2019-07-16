// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <utility>
#include <functional>
#include "dali/kernels/scratch.h"
#include "dali/kernels/context.h"
#include "dali/kernels/kernel_req.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace kernels {

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

  template <typename Kernel, typename... Args>
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

class DLL_PUBLIC KernelManager {
 public:
  void Initialize(size_t num_instances, size_t num_threads);

  void Reset();

  template <typename Kernel, typename... ConstructorArgs>
  Kernel &GetInstance(int instance_idx, ConstructorArgs &&...args) {
    return instances[instance_idx].create_or_get<Kernel>(std::forward<ConstructorArgs>(args)...);
  }

  KernelRequirements &GetRequirements(int index) {
    return instances[index].requirements;
  }

  const KernelRequirements &GetRequirements(int index) const {
    return instances[index].requirements;
  }

  size_t NumInstances() const { return instances.size(); }
  size_t NumThreads() const { return scratchpads.size(); }

  ScratchpadAllocator &GetScratchadAllocator(int thread_idx) {
    return scratchpads[thread_idx];
  }

  template <typename Kernel, typename... InArgs>
  KernelRequirements &Setup(int instance_idx, KernelContext &context, InArgs &&...in_args) {
    auto &inst = instances[instance_idx];
    return inst.requirements = inst.get<Kernel>().Setup(context, std::forward<InArgs>(in_args)...);
  }

  template <typename Kernel, typename... OutInArgs>
  void Run(int thread_idx, int instance_idx, KernelContext &context, OutInArgs &&...out_in_args) {
    assert(static_cast<size_t>(thread_idx) < scratchpads.size());
    auto &inst = instances[instance_idx];
    auto &alloc = GetScratchadAllocator(thread_idx);
    ReserveScratchpad(alloc, inst.requirements.scratch_sizes);
    auto scratchpad = alloc.GetScratchpad();
    auto *old_scratchpad = context.scratchpad;
    context.scratchpad = &scratchpad;
    inst.get<Kernel>().Run(context, std::forward<OutInArgs>(out_in_args)...);
    context.scratchpad = old_scratchpad;
  }

  void ReserveScratchMem(AllocType type, size_t bytes) {
    for (auto &sa : scratchpads)
      sa.Reserve(type, bytes);
  }

 private:
  void ReserveScratchpad(
      ScratchpadAllocator &sa,
      std::array<size_t, ScratchpadAllocator::NumAllocTypes> sizes);

  SmallVector<AnyKernelInstance, 1> instances;
  SmallVector<ScratchpadAllocator, 1> scratchpads;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_MANAGER_H_
