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

#include <stdexcept>
#include <cstring>
#include "dali/core/mm/default_resources.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/device_guard.h"
#include "dali/core/mm/async_pool.h"
#include "dali/core/mm/composite_resource.h"
#include "dali/core/mm/cuda_vm_resource.h"

namespace dali {
namespace mm {

namespace {

template <typename T>
inline std::shared_ptr<T> wrap(T *p, bool own) {
  if (own)
    return std::shared_ptr<T>(p);
  else
    return std::shared_ptr<T>(p, [](T*){});
}

struct DefaultResources {
  ~DefaultResources() {
    ReleasePinned();
    ReleaseDevice();
    ReleaseManaged();
    ReleaseHost();
  }

  static DefaultResources &instance() {
    static DefaultResources dr;
    return dr;
  }

  std::shared_ptr<host_memory_resource> host;
  std::shared_ptr<pinned_async_resource> pinned_async;
  std::shared_ptr<managed_async_resource> managed;
  std::vector<std::shared_ptr<device_async_resource>> device;
  std::mutex mtx;

  void ReleasePinned() {
    Release(pinned_async);
  }

  void ReleaseManaged() {
    Release(managed);
  }

  void ReleaseDevice() {
    Release(device);
  }

  void ReleaseHost() {
    Release(host);
  }

 private:
  template <typename Ptr>
  void Release(Ptr &p) noexcept(false) {
    if (cudaGetLastError() == cudaErrorCudartUnloading) {
      Abandon(p);
    } else {
      p = {};
    }
  }

  template <typename T>
  void Abandon(std::vector<T> &v) {
    for (auto &x : v)
      Abandon(x);
    v.clear();
  }

  template <typename T>
  void Abandon(std::shared_ptr<T> &p) {
    // shared_ptr doesn't have a release() function like one in unique_ptr so we could
    // safely abandon the pointer.
    // What happens here: we create (by placement new) a shared pointer and move the original
    // pointer there - this way the original pointer is released and the new pointer
    // is allocated dynamically on stack and never destroyed.
    // This accomplishes the intentional leaking of the pointer.
    using Ptr = std::shared_ptr<T>;
    std::aligned_storage_t<sizeof(Ptr), alignof(Ptr)> dump;
    new (&dump) Ptr(std::move(p));
  }
};

#define g_resources (DefaultResources::instance())

inline std::shared_ptr<host_memory_resource> CreateDefaultHostResource() {
  static auto rsrc = std::make_shared<malloc_memory_resource>();
  return rsrc;
}

struct CUDARTLoader {
  CUDARTLoader() {
    int device_id = 0;
    CUDA_CALL(cudaGetDevice(&device_id));
    DeviceGuard dg(device_id);
  }
};


template <typename Callable>
struct CallAtExit {
  explicit CallAtExit(Callable &&c) : callable(std::move(c)) {}
  ~CallAtExit() {
    callable();
  }
  Callable callable;
};

template <typename Callable>
CallAtExit<Callable> AtExit(Callable &&c) {
  return CallAtExit<Callable>(std::forward<Callable>(c));
}

bool UseDeviceMemoryPool() {
  static bool value = []() {
    const char *env = std::getenv("DALI_USE_DEVICE_MEM_POOL");
    return !env || atoi(env);
  }();
  return value;
}

bool UsePinnedMemoryPool() {
  static bool value = []() {
    const char *env = std::getenv("DALI_USE_PINNED_MEM_POOL");
    return !env || atoi(env);
  }();
  return value;
}

bool UseVMM() {
  static bool value = []() {
    const char *env = std::getenv("DALI_USE_VMM");
    return !env || atoi(env);
  }();
  return value;
}

bool UseDeferredDealloc() {
  static bool value = []() {
    const char *env = std::getenv("DALI_USE_DEFERRED_DEALLOC");
    return !env || atoi(env);
  }();
  return value;
}

inline std::shared_ptr<device_async_resource> CreateDefaultDeviceResource() {
  static CUDARTLoader CUDAInit;
  if (!UseDeviceMemoryPool()) {
    static auto rsrc = std::make_shared<mm::cuda_malloc_memory_resource>();
    return rsrc;
  }
  #if DALI_USE_CUDA_VM_MAP
  if (cuvm::IsSupported() && UseVMM()) {
    if (UseDeferredDealloc()) {
      using resource_type = mm::async_pool_resource<mm::memory_kind::device, cuda_vm_resource,
                                                    std::mutex, void>;
      return std::make_shared<resource_type>();
    } else {
      using resource_type = mm::async_pool_resource<mm::memory_kind::device, cuda_vm_resource_base,
                                                    std::mutex, void>;
      return std::make_shared<resource_type>();
    }
  }
  #endif  // DALI_USE_CUDA_VM_MAP
  {
    static auto upstream = std::make_shared<mm::cuda_malloc_memory_resource>();
    if (UseDeferredDealloc()) {
      using resource_type = mm::async_pool_resource<mm::memory_kind::device>;
      auto rsrc = std::make_shared<resource_type>(upstream.get());
      return make_shared_composite_resource(std::move(rsrc), upstream);
    } else {
      using resource_type = mm::async_pool_resource<mm::memory_kind::device,
              pool_resource_base<memory_kind::device, any_context, coalescing_free_tree, spinlock>>;
      auto rsrc = std::make_shared<resource_type>(upstream.get());
      return make_shared_composite_resource(std::move(rsrc), upstream);
    }
  }
}

inline std::shared_ptr<pinned_async_resource> CreateDefaultPinnedResource() {
  if (!UsePinnedMemoryPool()) {
    static auto upstream = std::make_shared<mm::pinned_malloc_memory_resource>();
    return upstream;
  }
  static auto upstream = std::make_shared<pinned_malloc_memory_resource>();
  if (UseDeferredDealloc()) {
    using resource_type = mm::async_pool_resource<mm::memory_kind::pinned>;
    auto rsrc = std::make_shared<resource_type>(upstream.get());
    return make_shared_composite_resource(std::move(rsrc), upstream);
  } else {
    using resource_type = mm::async_pool_resource<mm::memory_kind::pinned,
        pool_resource_base<memory_kind::pinned, any_context, coalescing_free_tree, spinlock>>;
    auto rsrc = std::make_shared<resource_type>(upstream.get());
    return make_shared_composite_resource(std::move(rsrc), upstream);
  }
}

inline std::shared_ptr<managed_async_resource> CreateDefaultManagedResource() {
  static auto rsrc = std::make_shared<mm::managed_malloc_memory_resource>();
  return rsrc;
}


template <typename Kind>
const std::shared_ptr<default_memory_resource_t<Kind>> &ShareDefaultResourceImpl();

template <>
const std::shared_ptr<host_memory_resource> &ShareDefaultResourceImpl<memory_kind::host>() {
  if (!g_resources.host) {
    std::lock_guard<std::mutex> lock(g_resources.mtx);
    if (!g_resources.host)
      g_resources.host = CreateDefaultHostResource();
  }
  return g_resources.host;
}

template <>
const std::shared_ptr<pinned_async_resource> &ShareDefaultResourceImpl<memory_kind::pinned>() {
  if (!g_resources.pinned_async) {
    std::lock_guard<std::mutex> lock(g_resources.mtx);
    if (!g_resources.pinned_async) {
      static CUDARTLoader init_cuda;  // force initialization of CUDA before creating the resource
      static auto cleanup = AtExit([] {
        g_resources.ReleasePinned();
      });
      g_resources.pinned_async = CreateDefaultPinnedResource();
    }
  }
  return g_resources.pinned_async;
}

template <>
const std::shared_ptr<managed_async_resource> &ShareDefaultResourceImpl<memory_kind::managed>() {
  if (!g_resources.managed) {
    std::lock_guard<std::mutex> lock(g_resources.mtx);
    if (!g_resources.managed) {
      static CUDARTLoader init_cuda;  // force initialization of CUDA before creating the resource
      static auto cleanup = AtExit([] {
        g_resources.ReleaseManaged();
      });
      g_resources.managed = CreateDefaultManagedResource();
    }
  }
  return g_resources.managed;
}

const std::shared_ptr<device_async_resource> &ShareDefaultDeviceResourceImpl(int device_id) {
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }
  if (g_resources.device.empty()) {
    std::lock_guard<std::mutex> lock(g_resources.mtx);
    int ndevs = 0;
    CUDA_CALL(cudaGetDeviceCount(&ndevs));
    g_resources.device.resize(ndevs);
  }
  if (static_cast<size_t>(device_id) >= g_resources.device.size()) {
    throw std::out_of_range(make_string(device_id, " is not a valid CUDA device index. "
      "Shoud be 0 <= device_id < ", g_resources.device.size(), " or negative for current device."));
  }
  if (!g_resources.device[device_id]) {
    std::lock_guard<std::mutex> lock(g_resources.mtx);
    if (!g_resources.device[device_id]) {
      DeviceGuard devg(device_id);
      static CUDARTLoader init_cuda;  // force initialization of CUDA before creating the resource
      static auto cleanup = AtExit([] {
        g_resources.ReleaseDevice();
      });
      g_resources.device[device_id] = CreateDefaultDeviceResource();
    }
  }
  return g_resources.device[device_id];
}

template <>
const std::shared_ptr<device_async_resource> &ShareDefaultResourceImpl<memory_kind::device>() {
  return ShareDefaultDeviceResourceImpl(-1);
}

}  // namespace


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::host>(std::shared_ptr<host_memory_resource> resource) {
  std::lock_guard<std::mutex> lock(g_resources.mtx);
  g_resources.host = std::move(resource);
}


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::host>(host_memory_resource *resource, bool own) {
  SetDefaultResource<memory_kind::host>(wrap(resource, own));
}


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::pinned>(std::shared_ptr<pinned_async_resource> resource) {
  std::lock_guard<std::mutex> lock(g_resources.mtx);
  g_resources.pinned_async = std::move(resource);
}


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::pinned>(pinned_async_resource *resource, bool own) {
  SetDefaultResource<memory_kind::pinned>(wrap(resource, own));
}


void SetDefaultDeviceResource(int device_id, std::shared_ptr<device_async_resource> resource) {
  std::lock_guard<std::mutex> lock(g_resources.mtx);
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }
  int ndevs = g_resources.device.size();
  if (ndevs == 0) {
    CUDA_CALL(cudaGetDeviceCount(&ndevs));
    g_resources.device.resize(ndevs);
  }
  if (device_id < 0 || device_id >= ndevs) {
    throw std::out_of_range(make_string(device_id, " is not a valid CUDA device index. "
      "Shoud be 0 <= device_id < ", g_resources.device.size(), " or negative for current device."));
  }
  g_resources.device[device_id] = std::move(resource);
}

void SetDefaultDeviceResource(int device_id, device_async_resource *resource, bool own) {
  SetDefaultDeviceResource(device_id, wrap(resource, own));
}

// This function is for testing purposes only - it must be visible
DLL_PUBLIC void _Test_FreeDeviceResources() {
  // clear does not deallocate - we need something stronger
  decltype(g_resources.device) empty;
  g_resources.device.swap(empty);
}

template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::device>(std::shared_ptr<device_async_resource> resource) {
  int dev = 0;
  CUDA_CALL(cudaGetDevice(&dev));
  SetDefaultDeviceResource(dev, std::move(resource));
}

template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::device>(device_async_resource *resource, bool own) {
  SetDefaultResource<memory_kind::device>(wrap(resource, own));
}

template <typename Kind> DLL_PUBLIC
std::shared_ptr<default_memory_resource_t<Kind>> ShareDefaultResource() {
  return ShareDefaultResourceImpl<Kind>();
}

template <typename Kind> DLL_PUBLIC
default_memory_resource_t<Kind> *GetDefaultResource() {
  return ShareDefaultResourceImpl<Kind>().get();
}

#define INSTANTIATE_DEFAULT_RESOURCE_GETTERS(Kind) \
template DLL_PUBLIC std::shared_ptr<default_memory_resource_t<Kind>> ShareDefaultResource<Kind>(); \
template DLL_PUBLIC default_memory_resource_t<Kind> *GetDefaultResource<Kind>();

INSTANTIATE_DEFAULT_RESOURCE_GETTERS(memory_kind::host);
INSTANTIATE_DEFAULT_RESOURCE_GETTERS(memory_kind::pinned);
INSTANTIATE_DEFAULT_RESOURCE_GETTERS(memory_kind::device);
INSTANTIATE_DEFAULT_RESOURCE_GETTERS(memory_kind::managed);

DLL_PUBLIC
std::shared_ptr<device_async_resource> ShareDefaultDeviceResource(int device_id) {
  return ShareDefaultDeviceResourceImpl(device_id);
}

DLL_PUBLIC
device_async_resource *GetDefaultDeviceResource(int device_id) {
  return ShareDefaultDeviceResourceImpl(device_id).get();
}

}  // namespace mm
}  // namespace dali
