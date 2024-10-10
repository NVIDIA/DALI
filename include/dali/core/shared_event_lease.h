// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_SHARED_EVENT_LEASE_H_
#define DALI_CORE_SHARED_EVENT_LEASE_H_

#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_error.h"

namespace dali {

class SharedEventLease {
 public:
  SharedEventLease() = default;
  explicit SharedEventLease(CUDAEvent &&event, int device_id = -1, CUDAEventPool *owner = nullptr) {
    if (device_id < 0)
      CUDA_CALL(cudaGetDevice(&device_id));
    if (!owner)
      owner = &CUDAEventPool::instance();
    event_ = std::shared_ptr<void>(event.get(), [device_id, owner](void *e) {
      owner->Put(CUDAEvent(static_cast<cudaEvent_t>(e)), device_id);
    });
    event.release();
  }

  static SharedEventLease Get(CUDAEventPool &pool, int device_id = -1) {
    if (device_id < 0)
      CUDA_CALL(cudaGetDevice(&device_id));
    CUDAEvent event = pool.Get(device_id);
    return SharedEventLease(std::move(event), device_id, &pool);
  }

  static SharedEventLease Get(int device_id = -1) {
    return Get(CUDAEventPool::instance(), device_id);
  }

  void reset() noexcept {
    event_.reset();
  }

  cudaEvent_t get() const noexcept {
    return static_cast<cudaEvent_t>(event_.get());
  }

  long use_count() const noexcept {  // NOLINT(runtime/int)
    return event_.use_count();
  }

  explicit operator bool() const noexcept {
    return static_cast<bool>(event_);
  }

  operator cudaEvent_t() const noexcept {
    return get();
  }

  bool operator==(const SharedEventLease &other) const noexcept {
    return get() == other.get();
  }

  bool operator!=(const SharedEventLease &other) const noexcept {
    return get() != other.get();
  }

  bool operator==(cudaEvent_t event) const noexcept {
    return get() == event;
  }

  bool operator!=(cudaEvent_t event) const noexcept {
    return get() != event;
  }


  bool operator==(std::nullptr_t) const noexcept {
    return get() == nullptr;
  }

  bool operator!=(std::nullptr_t) const noexcept {
    return get() != nullptr;
  }

 private:
  // Hack: use shared_ptr<void> to store a CUDA event - shared_ptr doesn't care whether the pointer
  // it manages is a real pointer or something else as long as:
  // - null value is equivalent to nullptr
  // - the provided deleter can free the object.
  std::shared_ptr<void> event_;
};

}  // namespace dali

#endif  // DALI_CORE_SHARED_EVENT_LEASE_H_
