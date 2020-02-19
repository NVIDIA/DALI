// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_CUDA_EVENT_H_
#define DALI_CORE_CUDA_EVENT_H_

#include <driver_types.h>
#include <utility>
#include "dali/core/unique_handle.h"

namespace dali {

/**
 * @brief A wrapper class for CUDA event handle (cudaEvent_t),
 *
 * The purpose of this class is to provide safe ownership and lifecycle management
 * for CUDA event handles.
 * The event object may be created using the factory functions @ref Create and @ref CreateWithFlags.
 *
 * The object may also assume ownership of a pre-existing handle via constructor or
 * @link UniqueHandle::reset(handle_type) reset @endlink function.
 */
class DLL_PUBLIC CUDAEvent : public UniqueHandle<cudaEvent_t, CUDAEvent> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(cudaEvent_t, CUDAEvent)
  constexpr CUDAEvent() = default;

  /// @brief Creates an event on specified device (or current device, if device_id < 0)
  static CUDAEvent Create(int device_id = -1);

  /// @brief Creates an event event with specific flags on the device specified
  ///        (or current device, if device_id < 0)
  static CUDAEvent CreateWithFlags(unsigned flags, int device_id = -1);

  /// @brief Calls cudaEventDestroy on the handle.
  static void DestroyHandle(cudaEvent_t);
};

}  // namespace dali

#endif  // DALI_CORE_CUDA_EVENT_H_
