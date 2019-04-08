// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_USER_STREAM_H_
#define DALI_UTIL_USER_STREAM_H_

#include <cuda_runtime_api.h>

#include <mutex>
#include <unordered_map>

#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/backend.h"
#include "dali/error_handling.h"

// This file contains utilities helping inspection and interaction with DALI GPU buffers
// without forcing synchronization of all pipelines.
// This functionality is NOT supposed to be used in any
// performance-oriented portions of the code!

namespace dali {
class DLL_PUBLIC UserStream {
 public:
  DLL_PUBLIC static UserStream* Get() {
    std::lock_guard<std::mutex> lock(m_);
    if (us_ == nullptr) {
      us_ = new UserStream();
    }
    return us_;
  }

  DLL_PUBLIC cudaStream_t GetStream(const dali::Buffer<GPUBackend> &b) {
    size_t dev = GetDeviceForBuffer(b);
    std::lock_guard<std::mutex> lock(m_);
    auto it = streams_.find(dev);
    if (it != streams_.end()) {
      return it->second;
    } else {
      constexpr int kDefaultStreamPriority = 0;
      CUDA_CALL(cudaStreamCreateWithPriority(&streams_[dev], cudaStreamNonBlocking,
                                             kDefaultStreamPriority));
      return streams_.at(dev);
    }
  }

  DLL_PUBLIC void WaitForDevice(const dali::Buffer<GPUBackend> &b) {
    GetDeviceForBuffer(b);
    // GetDeviceForBuffer sets proper current device
    CUDA_CALL(cudaDeviceSynchronize());
  }

  DLL_PUBLIC void Wait(const dali::Buffer<GPUBackend> &b) {
    size_t dev = GetDeviceForBuffer(b);
    DALI_ENFORCE(streams_.find(dev) != streams_.end(),
        "Can only wait on user streams");
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  DLL_PUBLIC void Wait() {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    DALI_ENFORCE(streams_.find(dev) != streams_.end(),
        "Can only wait on user streams");
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  DLL_PUBLIC void WaitAll() {
    for (const auto &dev_pair : streams_) {
      CUDA_CALL(cudaSetDevice(dev_pair.first));
      CUDA_CALL(cudaStreamSynchronize(dev_pair.second));
    }
  }

 private:
  UserStream() {}

  size_t GetDeviceForBuffer(const dali::Buffer<GPUBackend> &b) {
    int dev = b.device_id();
    DALI_ENFORCE(dev != -1,
        "Used a pointer from unknown device");
    CUDA_CALL(cudaSetDevice(dev));
    return dev;
  }

  static std::mutex m_;
  static UserStream * us_;

  std::unordered_map<int, cudaStream_t> streams_;
};
}  // namespace dali

#endif  // DALI_UTIL_USER_STREAM_H_
