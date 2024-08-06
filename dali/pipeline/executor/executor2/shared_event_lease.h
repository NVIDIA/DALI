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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_SHARED_EVENT_LEASE_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_SHARED_EVENT_LEASE_H_

#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_error.h"

namespace dali {
namespace exec2 {

class SharedEventLease {
 public:
  SharedEventLease() = default;
  explicit SharedEventLease(CUDAEvent &&event, int device_id = -1) {
    if (device_id < 0)
      CUDA_CALL(cudaGetDevice(&device_id));

    event_ = std::shared_ptr<void>(event.get(), [device_id](void *e) {
      CUDAEventPool::instance().Put(CUDAEvent(static_cast<cudaEvent_t>(e)), device_id);
    });

    event.release();
  }

  static SharedEventLease Get(int device_id = -1) {
    if (device_id < 0)
      CUDA_CALL(cudaGetDevice(&device_id));
    CUDAEvent event = CUDAEventPool::instance().Get(device_id);
    return SharedEventLease(std::move(event), device_id);
  }

  void reset() {
    event_.reset();
  }

  cudaEvent_t get() const {
    return static_cast<cudaEvent_t>(event_.get());
  }

  explicit operator bool() const {
    return event_ != nullptr;
  }

  operator cudaEvent_t() const {
    return get();
  }

 private:
  // Hack: use shared_ptr<void> to store a CUDA event - shared_ptr doesn't care whether the pointer
  // it manages is a real pointer or something else as long as:
  // - null value is equivalent to nullptr
  // - the provided deleter can free the object.
  std::shared_ptr<void> event_;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_SHARED_EVENT_LEASE_H_
