// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/core/call_at_exit.h"
#include "dali/core/small_vector.h"
#include "dali/core/mm/memory_kind.h"

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
      Kernel *k = new Kernel{std::forward<Args>(args)...};
      instance = { k, deleter };
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
 * run-time.
 */
class DLL_PUBLIC KernelManager {
 public:
  static constexpr size_t NumMemKinds = ScratchpadAllocator::NumMemKinds;
  using ScratchSizes = std::array<size_t, NumMemKinds>;

  /**
   * @brief Creates `num_instances` slots for kernels
   *
   * @param num_instances -  number of Kernel instances to be created; typically corresponds
   *                         to number of samples (for per-sample kernels) or minibatches
   */
  void Resize(size_t num_instances) { instances.resize(num_instances); }

  /**
   * @brief Creates `num_instances` kernels of type Kernel constructed with `args...`.
   *
   * @param num_instances -  number of Kernel instances to be created; typically corresponds
   *                         to number of samples (for per-sample kernels) or minibatches
   * @param args           - arguments passed to Kernel's constructor upon creation.
   * @tparam Kernel        - type of the kernel to be created
   */
  template <typename Kernel, typename... Args>
  void Resize(size_t num_instances, const Args&... args) {
    Resize(num_instances);
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
   * @brief Clears kernel instances
   */
  void Reset() {
    instances.clear();
  }

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

  /**
   * @brief Calls setup on specified kernel instance.
   *
   * @param instance_idx   - kernel instance index; typically corresponds
   *                         to sample index (for per-sample kernels) or minibatch index
   * @param context        - context for the kernel
   *                         * should contain valid CUDA stream for GPU kernels;
   * @param in_args        - pack of arguments (inputs, arguments) used in Kernel::Setup
   * @return Reference to internally maintained copy of the kernel requirements.
   */
  template <typename Kernel, typename... InArgs>
  KernelRequirements &Setup(int instance_idx, KernelContext &context, InArgs &&...in_args) {
    auto &inst = instances[instance_idx];
    inst.requirements = inst.get<Kernel>().Setup(context, std::forward<InArgs>(in_args)...);
    return inst.requirements;
  }

  /**
   * @brief Calls Run on specified kernel instance
   *
   * @param instance_idx   - kernel instance index; typically corresponds
   *                         to sample index (for per-sample kernels) or minibatch index
   * @param context        - context for the kernel
   *                         * should contain valid CUDA stream for GPU kernels;
   *                         * if scratchpad pointer is null, a temporary dynamic scratchpad is
   *                           created
   * @param out_in_args    - pack of arguments (outputs, inputs, arguments) used in Kernel::Run
   */
  template <typename Kernel, typename... OutInArgs>
  void Run(int instance_idx, KernelContext &context, OutInArgs &&...out_in_args) {
    assert(instance_idx >= 0 &&
           static_cast<size_t>(instance_idx) < NumInstances() &&
           "Kernel instance index (instance_idx) out of range");
    auto &inst = instances[instance_idx];
    if (!context.scratchpad) {
      DynamicScratchpad scratchpad({}, AccessOrder(context.gpu.stream));
      context.scratchpad = &scratchpad;
      auto finally = AtScopeExit([&]() {
        context.scratchpad = nullptr;
      });
      inst.get<Kernel>().Run(context, std::forward<OutInArgs>(out_in_args)...);
    } else {
      inst.get<Kernel>().Run(context, std::forward<OutInArgs>(out_in_args)...);
    }
  }

 private:
  SmallVector<AnyKernelInstance, 1> instances;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_MANAGER_H_
