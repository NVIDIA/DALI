// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "rmm/mr/device/owning_wrapper.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"

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
    ReleaseHost();
  }

  static DefaultResources &instance() {
    static DefaultResources dr;
    return dr;
  }

  std::shared_ptr<host_memory_resource> host;
  std::shared_ptr<pinned_async_resource> pinned_async;
  std::vector<std::shared_ptr<device_async_resource>> device;

  void ReleasePinned() {
    Release(pinned_async);
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
      abandon(p);
    } else {
      p = {};
    }
  }

  template <typename T>
  void abandon(std::vector<T> &v) {
    for (auto &x : v)
      abandon(x);
    v.clear();
  }

  template <typename T>
  void abandon(std::shared_ptr<T> &p) {
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
  return std::make_shared<malloc_memory_resource>();
}

struct CUDARTLoader {
  CUDARTLoader() {
    int device_id = 0;
    CUDA_CALL(cudaGetDevice(&device_id));
    CUDA_CALL(cudaSetDevice(device_id));
  }
};


template <class Callable>
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

inline std::shared_ptr<device_async_resource> CreateDefaultDeviceResource() {
  static CUDARTLoader CUDAInit;
#ifdef DALI_USE_RMM_DEVICE_RESOURCE
  auto upstream = std::make_shared<rmm::mr::cuda_memory_resource>();
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(std::move(upstream));
#else
  using resource_type = mm::async_pool_resource<mm::memory_kind::device>;
  static auto upstream = std::make_shared<mm::cuda_malloc_memory_resource>();
  auto rsrc = std::make_shared<resource_type>(upstream.get());
  return make_shared_composite_resource(std::move(rsrc), std::move(upstream));
#endif
}

inline std::shared_ptr<pinned_async_resource> CreateDefaultPinnedResource() {
  using resource_type = mm::async_pool_resource<mm::memory_kind::pinned>;
  auto upstream = std::make_shared<rmm::mr::pinned_memory_resource>();
  auto rsrc = std::make_shared<resource_type>(upstream.get());
  return make_shared_composite_resource(std::move(rsrc), std::move(upstream));
}

template <memory_kind kind>
const std::shared_ptr<default_memory_resource_t<kind>> &ShareDefaultResourceImpl();

template <>
const std::shared_ptr<host_memory_resource> &ShareDefaultResourceImpl<memory_kind::host>() {
  if (!g_resources.host)
    g_resources.host = CreateDefaultHostResource();
  return g_resources.host;
}

template <>
const std::shared_ptr<pinned_async_resource> &ShareDefaultResourceImpl<memory_kind::pinned>() {
  if (!g_resources.pinned_async) {
    static CUDARTLoader init_cuda;  // force initialization of CUDA before creating the resource
    static auto cleanup = AtExit([] {
      g_resources.ReleasePinned();
    });
    g_resources.pinned_async = CreateDefaultPinnedResource();
  }
  return g_resources.pinned_async;
}

const std::shared_ptr<device_async_resource> &ShareDefaultDeviceResourceImpl(int device_id) {
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }
  if (g_resources.device.empty()) {
    int ndevs = 0;
    CUDA_CALL(cudaGetDeviceCount(&ndevs));
    g_resources.device.resize(ndevs);
  }
  if (static_cast<size_t>(device_id) >= g_resources.device.size())
    throw std::out_of_range(make_string(device_id, " is not a valid CUDA device index."));
  if (!g_resources.device[device_id]) {
    DeviceGuard devg(device_id);
    static CUDARTLoader init_cuda;  // force initialization of CUDA before creating the resource
    static auto cleanup = AtExit([] {
      g_resources.ReleaseDevice();
    });
    g_resources.device[device_id] = CreateDefaultDeviceResource();
  }
  return g_resources.device[device_id];
}

template <>
const std::shared_ptr<device_async_resource> &ShareDefaultResourceImpl<memory_kind::device>() {
  return ShareDefaultDeviceResourceImpl(-1);
}

}  // namespace


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::host>(shared_ptr<host_memory_resource> resource) {
  g_resources.host = std::move(resource);
}


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::host>(host_memory_resource *resource, bool own) {
  SetDefaultResource<memory_kind::host>(wrap(resource, own));
}


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::pinned>(shared_ptr<pinned_async_resource> resource) {
  g_resources.pinned_async = std::move(resource);
}


template <> DLL_PUBLIC
void SetDefaultResource<memory_kind::pinned>(pinned_async_resource *resource, bool own) {
  SetDefaultResource<memory_kind::pinned>(wrap(resource, own));
}


void SetDefaultDeviceResource(int device_id, std::shared_ptr<device_async_resource> resource) {
  int ndevs = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndevs));
  if (device_id < 0 || device_id >= ndevs)
    throw std::out_of_range(make_string(device_id, " is not a valid CUDA device index."));
  g_resources.device.resize(ndevs);
  g_resources.device[device_id] = std::move(resource);
}

void SetDefaultDeviceResource(int device_id, device_async_resource *resource, bool own) {
  SetDefaultDeviceResource(device_id, wrap(resource, own));
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

template <memory_kind kind> DLL_PUBLIC
std::shared_ptr<default_memory_resource_t<kind>> ShareDefaultResource() {
  return ShareDefaultResourceImpl<kind>();
}

template <memory_kind kind> DLL_PUBLIC
default_memory_resource_t<kind> *GetDefaultResource() {
  return ShareDefaultResourceImpl<kind>().get();
}

#define INSTANTIATE_DEFAULT_RESOURCE_GETTERS(kind) \
template DLL_PUBLIC std::shared_ptr<default_memory_resource_t<kind>> ShareDefaultResource<kind>(); \
template DLL_PUBLIC default_memory_resource_t<kind> *GetDefaultResource<kind>();

INSTANTIATE_DEFAULT_RESOURCE_GETTERS(memory_kind::host);
INSTANTIATE_DEFAULT_RESOURCE_GETTERS(memory_kind::pinned);
INSTANTIATE_DEFAULT_RESOURCE_GETTERS(memory_kind::device);

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
