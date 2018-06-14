// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_USER_STREAM_H_
#define NDLL_UTIL_USER_STREAM_H_

#include <cuda_runtime_api.h>

#include <mutex>
#include <vector>

#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/error_handling.h"

// This file contains utilities helping inspection and interaction with NDLL GPU buffers
// without forcing synchronization of all pipelines.
// This functionality is NOT supposed to be used in any
// performance-oriented portions of the code!

namespace ndll {
class UserStream {
 public:
  static UserStream* Get() {
    std::unique_lock<std::mutex> lock(m_);
    if (us_ == nullptr) {
      us_ = new UserStream();
    }
    return us_;
  }

  cudaStream_t GetStream(const ndll::Buffer<GPUBackend> &b) {
    size_t dev = GetDeviceForBuffer(b);
    NDLL_ENFORCE(dev < streams_.size(), "Requested stream for unknown device");
    return streams_[dev];
  }

  void WaitForDevice(const ndll::Buffer<GPUBackend> &b) {
    GetDeviceForBuffer(b);
    // GetDeviceForBuffer sets proper current device
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void Wait(const ndll::Buffer<GPUBackend> &b) {
    size_t dev = GetDeviceForBuffer(b);
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  void Wait() {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  void WaitAll() {
    for (size_t i = 0; i < streams_.size(); ++i) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_CALL(cudaStreamSynchronize(streams_[i]));
    }
  }

 private:
  UserStream() {
    int gpu_count = 0;
    CUDA_CALL(cudaGetDeviceCount(&gpu_count));
    streams_.resize(gpu_count);
    for (size_t i = 0; i < streams_.size(); ++i) {
      CUDA_CALL(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
    }
  }

  size_t GetDeviceForBuffer(const ndll::Buffer<GPUBackend> &b) {
    const void* ptr = b.raw_data();
    cudaPointerAttributes attr;
    CUDA_CALL(cudaPointerGetAttributes(&attr, ptr));
    CUDA_CALL(cudaSetDevice(attr.device));
    return attr.device;
  }

  static std::mutex m_;
  static UserStream * us_;

  std::vector<cudaStream_t> streams_;
};
}  // namespace ndll

#endif  // NDLL_UTIL_USER_STREAM_H_
