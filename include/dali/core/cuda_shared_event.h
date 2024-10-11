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

#ifndef DALI_CORE_CUDA_SHARED_EVENT_H_
#define DALI_CORE_CUDA_SHARED_EVENT_H_

#include <cuda_runtime_api.h>
#include <memory>
#include <utility>
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_error.h"

namespace dali {

/** A reference counting wrapper around cudaEvent_t.
 *
 * This class wraps a cudaEvent_t in a shared_ptr-like interface.
 * Internally it uses a std::shared_ptr to manage the handle.
 *
 * The class provides convenience functions for getting the event from CUDAEventPool.
 */
class CUDASharedEvent {
 public:
  CUDASharedEvent() = default;

  template <typename EventDeleter>
  CUDASharedEvent(cudaEvent_t event, EventDeleter &&deleter)
  : event_{
    event,
    [del = std::move(deleter)](void *handle) mutable {
      del(static_cast<cudaEvent_t>(handle));
    }} {}

  explicit CUDASharedEvent(CUDAEvent event)
  : CUDASharedEvent(event.get(), CUDAEvent::DestroyHandle)  {
    (void)event.release();
  }

  template <typename EventDeleter>
  explicit CUDASharedEvent(CUDAEvent event, EventDeleter &&deleter)
  : CUDASharedEvent(event.get(), std::forward<EventDeleter>(deleter)) {
    (void)event.release();
  }

  static CUDASharedEvent GetFromPool(CUDAEventPool &pool, int device_id = -1) {
    if (device_id < 0)
      CUDA_CALL(cudaGetDevice(&device_id));
    CUDAEvent &&event = pool.Get(device_id);
    return CUDASharedEvent(
        std::move(event),
        [device_id, owner = &pool](void *e) {
          owner->Put(CUDAEvent(static_cast<cudaEvent_t>(e)), device_id);
        });
  }

  static CUDASharedEvent GetFromPool(int device_id = -1) {
    return GetFromPool(CUDAEventPool::instance(), device_id);
  }

  static CUDASharedEvent Create(int device_id = -1) {
    return CUDASharedEvent(CUDAEvent::Create(device_id));
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

  bool operator==(const CUDASharedEvent &other) const noexcept {
    return get() == other.get();
  }

  bool operator!=(const CUDASharedEvent &other) const noexcept {
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

#endif  // DALI_CORE_CUDA_SHARED_EVENT_H_
