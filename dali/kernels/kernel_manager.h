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

  explicit operator bool() const noexcept { return (bool)instance; }
};

class KernelManager {
 public:
  void Initialize(int num_instances, bool shared_scratchpad) {
    instances.resize(num_instances);
    scratchpads.resize(shared_scratchpad ? 1 : num_instances);
  }

  template <typename Kernel, typename... ConstructorArgs>
  Kernel &GetInstance(int instance_idx, ConstructorArgs &&...args) {
    return instances[instance_idx].create_or_get<Kernel>(std::forward<ConstructorArgs>(args)...);
  }

  AnyKernelInstance &operator[](int index) {
    return instances[index];
  }
  const AnyKernelInstance &operator[](int index) const {
    return instances[index];
  }

  bool IsScratchpadShared() const {
    return scratchpads.size() == 1;
  }

  ScratchpadAllocator &GetScratchadAllocator(int instance_idx) {
    return IsScratchpadShared() ? scratchpads.front() : scratchpads[instance_idx];
  }

  template <typename Kernel, typename... InArgs>
  KernelRequirements &Setup(int instance_idx, KernelContext &context, InArgs &&...in_args) {
    auto &inst = instances[instance_idx];
    scratchpad_dirty = true;
    return inst.requirements = inst.get<Kernel>().Setup(context, std::forward<InArgs>(in_args)...);
  }

  template <typename Kernel, typename... OutInArgs>
  void Run(int instance_idx, KernelContext &context, OutInArgs &&...out_in_args) {
    auto &inst = instances[instance_idx];
    auto &alloc = GetScratchadAllocator(instance_idx);
    alloc.Reserve(inst.requirements.scratch_sizes);
    auto scratchpad = alloc.GetScratchpad();
    auto *old_scratchpad = context.scratchpad;
    context.scratchpad = &scratchpad;
    inst.get<Kernel>().Run(context, std::forward<OutInArgs>(out_in_args)...);
    context.scratchpad = old_scratchpad;
  }

 private:
  SmallVector<AnyKernelInstance, 1> instances;
  SmallVector<ScratchpadAllocator, 1> scratchpads;
  bool scratchpad_dirty = false;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_MANAGER_H_
