// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/device_guard.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"


// This file contains utilities helping inspection and interaction with DALI GPU buffers
// without forcing synchronization of all pipelines.
// This functionality is NOT supposed to be used in any
// performance-oriented portions of the code!

namespace dali {
class DLL_PUBLIC UserStream {
 public:
  /**
   * @brief Gets UserStream instance
   */
  DLL_PUBLIC static UserStream *Get() {
    static UserStream us;
    return &us;
  }

  /**
   * @brief Obtains cudaStream_t for provided Tensor. If there is no for given device,
   * new one is created and stored in the internal map
   */
  DLL_PUBLIC cudaStream_t GetStream(const dali::Tensor<GPUBackend> &t) {
    return GetStream(GetDeviceForBuffer(t));
  }

  /**
   * @brief Obtains cudaStream_t for provided TensorList. If there is no for given device,
   * new one is created and stored in the internal map
   */
  DLL_PUBLIC cudaStream_t GetStream(const dali::TensorList<GPUBackend> &tl) {
    return GetStream(GetDeviceForBuffer(tl));
  }

  /**
   * @brief Synchronizes on the device where given Tensor t exists
   */
  DLL_PUBLIC void WaitForDevice(const dali::Tensor<GPUBackend> &t) {
    WaitForDevice(GetDeviceForBuffer(t));
  }

  /**
   * @brief Synchronizes on the device where given TensorList tl exists
   */
  DLL_PUBLIC void WaitForDevice(const dali::TensorList<GPUBackend> &tl) {
    WaitForDevice(GetDeviceForBuffer(tl));
  }

  /**
   * @brief Synchronizes on the the stream where Tensor t was created
   */
  DLL_PUBLIC void Wait(const dali::Tensor<GPUBackend> &t) {
    Wait(GetDeviceForBuffer(t));
  }

  /**
   * @brief Synchronizes on the the stream where TensorList tl was created
   */
  DLL_PUBLIC void Wait(const dali::TensorList<GPUBackend> &tl) {
    Wait(GetDeviceForBuffer(tl));
  }

  /**
   * @brief Synchronizes stream connected with the current device
   */
  DLL_PUBLIC void Wait() {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    DALI_ENFORCE(streams_.find(dev) != streams_.end(), "Can only wait on user streams");
    DeviceGuard g(dev);
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  /**
   * @brief Synchronizes all tracked streams
   */
  DLL_PUBLIC void WaitAll() {
    for (const auto &dev_pair : streams_) {
      DeviceGuard g(dev_pair.first);
      CUDA_CALL(cudaStreamSynchronize(dev_pair.second));
    }
  }

 private:
  UserStream() = default;

  size_t GetDeviceForBuffer(const dali::Tensor<GPUBackend> &t) {
    int dev = t.device_id();
    DALI_ENFORCE(dev != -1, "Used a pointer from unknown device");
    return dev;
  }

  size_t GetDeviceForBuffer(const dali::TensorList<GPUBackend> &tl) {
    int dev = tl.device_id();
    DALI_ENFORCE(dev != -1, "Used a pointer from unknown device");
    return dev;
  }


  /**
   * @brief Obtains cudaStream_t for for given device, if there is none new one is created and
   * stored in the internal map
   */
  DLL_PUBLIC cudaStream_t GetStream(size_t dev) {
    std::lock_guard<std::mutex> lock(m_);
    auto it = streams_.find(dev);
    if (it != streams_.end()) {
      return it->second;
    } else {
      DeviceGuard g(dev);
      constexpr int kDefaultStreamPriority = 0;
      CUDA_CALL(cudaStreamCreateWithPriority(&streams_[dev], cudaStreamNonBlocking,
                                             kDefaultStreamPriority));
      return streams_.at(dev);
    }
  }

  /**
   * @brief Synchronizes given device
   */
  DLL_PUBLIC void WaitForDevice(size_t dev) {
    DeviceGuard g(dev);
    CUDA_CALL(cudaDeviceSynchronize());
  }

  /**
   * @brief Synchronizes stream that was created for given device
   */
  void Wait(size_t dev) {
    DALI_ENFORCE(streams_.find(dev) != streams_.end(), "Can only wait on user streams");
    DeviceGuard g(dev);
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  static std::mutex m_;

  std::unordered_map<int, cudaStream_t> streams_;
};
}  // namespace dali

#endif  // DALI_UTIL_USER_STREAM_H_
