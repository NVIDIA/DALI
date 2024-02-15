// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <mutex>
#include <unordered_map>
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/device_guard.h"
#include "dali/core/spinlock.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/detail/align.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif  // NVML_ENABLED

namespace dali {
namespace mm {

#if CUDA_VERSION >= 11020

namespace {

class aligned_alloc_helper {
 public:
  static constexpr size_t kCudaMallocAlignment = 256;

  void *alloc(size_t size, size_t alignment, cudaStream_t stream) {
    void *ptr;
    if (alignment > kCudaMallocAlignment) {
      CUDA_CALL(cudaMallocAsync(&ptr, size + alignment - kCudaMallocAlignment, stream));
      if (detail::is_aligned(ptr, alignment))
        return ptr;

      void *aligned = detail::align_ptr(ptr, alignment);
      std::lock_guard guard(mtx_);
      orig_ptrs_[aligned] = ptr;
      return aligned;
    } else {
      CUDA_CALL(cudaMallocAsync(&ptr, size, stream));
    }
    return ptr;
  }

  void free(void *ptr, size_t size, size_t alignment, cudaStream_t stream) {
    (void)size;
    if (alignment > kCudaMallocAlignment) {
      std::lock_guard guard(mtx_);
      auto it = orig_ptrs_.find(ptr);
      if (it != orig_ptrs_.end()) {
        ptr = it->second;
        orig_ptrs_.erase(it);
      }
    }
    CUDA_DTOR_CALL(cudaFreeAsync(ptr, stream));
  }

  static aligned_alloc_helper &instance() {
    static aligned_alloc_helper inst;
    return inst;
  }

 private:
  spinlock mtx_;
  std::unordered_map<void *, void*> orig_ptrs_;
};

}  // namespace

cuda_malloc_async_memory_resource::cuda_malloc_async_memory_resource(int device_id) {
  // Construct the helper.
  // Calling this in the constructor guarantees that the construction of the object returned by
  // `aligned_alloc_helper::instance` is complete before the construction of the first instance
  // of `cuda_malloc_async_memory_resource` - a natural destruction order would be for all
  // instances of `cuda_malloc_async_memory_resource` to be destroyed before the helper
  // is destroyed.
  (void)aligned_alloc_helper::instance();

  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }

  device_id_ = device_id;
  DeviceGuard dg(device_id_);
  dummy_host_stream_ = CUDAStreamPool::instance().Get(device_id_);
#if NVML_ENABLED
  static const float driverVersion = []() {
    auto nvml_handle = nvml::NvmlInstance::CreateNvmlInstance();
    auto ret = nvml::GetDriverVersion();
    return ret;
  }();
  if (driverVersion < 470.60) {
    cudaMemPool_t memPool;
    CUDA_CALL(cudaDeviceGetDefaultMemPool(&memPool, device_id_));
    int val = 0;
    CUDA_CALL(cudaMemPoolSetAttribute(memPool, cudaMemPoolReuseAllowOpportunistic, &val));
  }
#endif  // NVML_ENABLED
}

bool cuda_malloc_async_memory_resource::is_supported(int device_id) {
  static const int num_devices = []() {
    int ndev;
    CUDA_CALL(cudaGetDeviceCount(&ndev));
    return ndev;
  }();
  enum Support {
    unintialized = 0,
    unsupported = -1,
    supported = 1
  };
  static vector<Support> support(num_devices);
  if (device_id < 0)
    CUDA_CALL(cudaGetDevice(&device_id));

  if (!support[device_id]) {
    auto stream = CUDAStreamPool::instance().Get(device_id);
    try {
    void *ptr;
      CUDA_CALL(cudaMallocAsync(&ptr, 16, stream));
      CUDA_CALL(cudaFreeAsync(ptr, stream));
      support[device_id] = supported;
    } catch (const CUDAError &e) {
      if (e.rt_error() == cudaErrorNotSupported)
        support[device_id] = unsupported;
      else
        throw;
    }
  }
  return support[device_id] == supported;
}

void *cuda_malloc_async_memory_resource::do_allocate(size_t size, size_t alignment) {
  DeviceGuard dg(device_id_);
  void *ptr = aligned_alloc_helper::instance().alloc(size, alignment, dummy_host_stream_);
  CUDA_CALL(cudaStreamSynchronize(dummy_host_stream_));
  return ptr;
}

void cuda_malloc_async_memory_resource::do_deallocate(void *ptr, size_t size, size_t alignment) {
  DeviceGuard dg(device_id_);
  aligned_alloc_helper::instance().free(ptr, size, alignment, dummy_host_stream_);
}

void *cuda_malloc_async_memory_resource::do_allocate_async(size_t size,
                                                           size_t alignment,
                                                           stream_view stream)  {
  DeviceGuard dg(device_id_);
  return aligned_alloc_helper::instance().alloc(size, alignment, stream.get());
}

void cuda_malloc_async_memory_resource::do_deallocate_async(void *ptr,
                                                            size_t size,
                                                            size_t alignment,
                                                            stream_view stream) {
  DeviceGuard dg(device_id_);
  aligned_alloc_helper::instance().free(ptr, size, alignment, stream.get());
}

#endif

}  // namespace mm
}  // namespace dali
