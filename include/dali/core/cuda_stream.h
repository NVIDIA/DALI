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

#ifndef DALI_CORE_CUDA_STREAM_H_
#define DALI_CORE_CUDA_STREAM_H_

#include <driver_types.h>
#include <utility>
#include "dali/core/unique_handle.h"

namespace dali {

/**
 * @brief A wrapper class for CUDA stream handle (cudaStream_t),
 *
 * The purpose of this class is to provide safe ownership and lifecycle management
 * for CUDA stream handles.
 * The stream object may be created using the factory functions @ref Create and
 * @ref CreateWithPriority.
 *
 * The object may also assume ownership of a pre-existing handle via constructor or
 * @link UniqueHandle::reset(handle_type) reset @endlink function.
 */
class DLL_PUBLIC CUDAStream : public UniqueHandle<cudaStream_t, CUDAStream>{
 public:
  DALI_INHERIT_UNIQUE_HANDLE(cudaStream_t, CUDAStream)

  /// @brief Creates a stream on specified device (or current device, if device_id < 0)
  static CUDAStream Create(bool non_blocking, int device_id = -1);

  /// @brief Creates a stream with given priority on specified device
  ///        (or current device, if device_id < 0)
  static CUDAStream CreateWithPriority(bool non_blocking, int priority, int device_id = -1);

  /// @brief Calls cudaStreamDestroy on the handle.
  static void DestroyHandle(cudaStream_t stream);
};

}  // namespace dali

#endif  // DALI_CORE_CUDA_STREAM_H_
