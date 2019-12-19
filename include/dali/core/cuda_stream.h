// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/common.h"

namespace dali {

class DLL_PUBLIC CUDAStream {
 public:
  constexpr CUDAStream() = default;

  /// @brief Creates a on specified device (or current device, if device_id < 0)
  static CUDAStream Create(bool nonBlocking, int device_id = -1);

  /// @brief Creates a non-blocking stream with given priority on specified device
  ///        (or current device, if device_id < 0)
  static CUDAStream CreateWithPriority(bool nonBlocking, int priority, int device_id = -1);

  constexpr explicit CUDAStream(cudaStream_t stream) : stream_(stream) {}

  inline ~CUDAStream() { reset(); }

  CUDAStream(const CUDAStream &) = delete;

  CUDAStream &operator=(const CUDAStream &) = delete;

  inline CUDAStream(CUDAStream &&other) : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  inline CUDAStream &operator=(CUDAStream &&other) {
    std::swap(stream_, other.stream_);
    other.reset();
    return *this;
  }

  void reset();

  inline void reset(cudaStream_t stream) {
    if (stream != stream_) {
      reset();
      stream_ = stream;
    }
  }

  inline constexpr operator cudaStream_t() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = 0;
};

}  // namespace dali

#endif  // DALI_CORE_CUDA_STREAM_H_
