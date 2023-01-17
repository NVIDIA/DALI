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

#ifndef DALI_CORE_MM_DEFAULT_RESOURCES_H_
#define DALI_CORE_MM_DEFAULT_RESOURCES_H_


#include <memory>
#include "dali/core/api_helper.h"
#include "dali/core/mm/memory_resource.h"

namespace dali {
namespace mm {

template <typename Kind>
struct DefaultMemoryResourceType;

template <>
struct DefaultMemoryResourceType<memory_kind::host> {
  using type = host_memory_resource;
};

template <>
struct DefaultMemoryResourceType<memory_kind::device> {
  using type = device_async_resource;
};

template <>
struct DefaultMemoryResourceType<memory_kind::pinned> {
  using type = pinned_async_resource;
};

template <>
struct DefaultMemoryResourceType<memory_kind::managed> {
  using type = managed_async_resource;
};

template <typename Kind>
using default_memory_resource_t = typename DefaultMemoryResourceType<Kind>::type;


/**
 * @brief Gets a shared pointer to the current memory resource for allocating memory of given kind.
 */
template <typename Kind>
DLL_PUBLIC std::shared_ptr<default_memory_resource_t<Kind>> ShareDefaultResource();

/**
 * @brief Gets a pointer to the current memory resource for allocating memory of given kind.
 */
template <typename Kind>
DLL_PUBLIC default_memory_resource_t<Kind> *GetDefaultResource();

/**
 * @brief Sets current memory resource for allocating memory of given kind,
 *        optionally granting ownership.
 *
 * If `own` is true, the resource will be managed by the library.
 * The ownership is assumed as soon as the function is called - if an exception happens inside,
 * the resource will be deleted.
 */
template <typename Kind>
DLL_PUBLIC void SetDefaultResource(default_memory_resource_t<Kind> *resource, bool own = false);

/**
 * @brief Sets current memory resource for allocating memory of given kind.
 */
template <typename Kind>
DLL_PUBLIC void SetDefaultResource(std::shared_ptr<default_memory_resource_t<Kind>> resource);

/**
 * @brief Gets a shared pointer to a memory resource.
 *
 * @param device_id Device index; if negative, current device is used.
 */
DLL_PUBLIC
std::shared_ptr<device_async_resource> ShareDefaultDeviceResource(int device_id = -1);

/**
 * @brief Gets device memory resource.
 *
 * @param device_id Device index; if negative, current device is used.
 */
DLL_PUBLIC
device_async_resource *GetDefaultDeviceResource(int device_id = -1);

/**
 * @brief Sets the default device memory resource for a specific device,
 *        optionally granting ownership.
 *
 * If `own` is true, the resource will be managed by the library.
 * The ownership is assumed as soon as the function is called - if an exception happens inside,
 * the resource will be deleted.
 */
DLL_PUBLIC
void SetDefaultDeviceResource(int device_id, device_async_resource *resource, bool own = false);

/**
 * @brief Sets the device memory resource for a specific device.
 */
DLL_PUBLIC
void SetDefaultDeviceResource(int device_id, std::shared_ptr<device_async_resource> resource);

/**
 * @brief Releases unused memory from memory pools
 *
 * The memory pools hold memory for future use. This function will attempt to free that memory.
 * Note that memory blocks that are partially used cannot be released.
 *
 * @note If the relevant memory resource doesn't expose pool-like interface or none if its
 *       accessible upstream resources exposes such an interface, then this function is a no-op.
 */
DLL_PUBLIC
void ReleaseUnusedMemory();

/**
 * @brief Preallocates device memory
 *
 * The function ensures that after the call, the amount of memory given in `bytes` can be
 * allocated from the pool (without further requests to the OS).
 *
 * The function works by allocating and then freeing the requested number of bytes.
 * Any outstanding allocations are not taken into account - that is, the peak amount
 * of memory allocated will be the sum of pre-existing allocation and the amount given
 * in `bytes`.
 *
 * @throws std::bad_alloc
 */
DLL_PUBLIC
void PreallocateDeviceMemory(size_t bytes, int device_id = -1);

/**
 * @brief Preallocates host pinned memory
 *
 * The function ensures that after the call, the amount of memory given in `bytes` can be allocated
 * from the pool (without further requests to the OS).
 *
 * The function works by allocating and then freeing the requested number of bytes.
 * Any outstanding allocations are not taken into account - that is, the peak amount
 * of memory allocated will be the sum of pre-existing allocation and the amount given
 * in `bytes`.
 *
 * @throws std::bad_alloc
 */
DLL_PUBLIC
void PreallocatePinnedMemory(size_t bytes);

}  // namespace mm
}  // namespace dali

#endif  //  DALI_CORE_MM_DEFAULT_RESOURCES_H_
