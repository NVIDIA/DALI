// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_USER_STREAM_H_
#define NDLL_UTIL_USER_STREAM_H_

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <mutex>
#include <unordered_map>

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
    std::unique_lock<std::mutex> lock(m_);
    auto it = streams_.find(dev);
    if (it != streams_.end()) {
      return it->second;
    } else {
      CUDA_CALL(cudaStreamCreate(&streams_[dev]));
      return streams_.at(dev);
    }
  }

  void WaitForDevice(const ndll::Buffer<GPUBackend> &b) {
    GetDeviceForBuffer(b);
    // GetDeviceForBuffer sets proper current device
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void Wait(const ndll::Buffer<GPUBackend> &b) {
    size_t dev = GetDeviceForBuffer(b);
    NDLL_ENFORCE(streams_.find(dev) != streams_.end(),
        "Can only wait on user streams");
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  void Wait() {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    NDLL_ENFORCE(streams_.find(dev) != streams_.end(),
        "Can only wait on user streams");
    CUDA_CALL(cudaStreamSynchronize(streams_[dev]));
  }

  void WaitAll() {
    for (const auto &dev_pair : streams_) {
      CUDA_CALL(cudaSetDevice(dev_pair.first));
      CUDA_CALL(cudaStreamSynchronize(dev_pair.second));
    }
  }

 private:
  UserStream() {}

  size_t GetDeviceForBuffer(const ndll::Buffer<GPUBackend> &b) {
    const void* ptr = b.raw_data();
    CUdeviceptr cuptr = (const CUdeviceptr) ptr;
    CUcontext ctx;
    CUpointer_attribute attr = CU_POINTER_ATTRIBUTE_CONTEXT;
    CUresult result = cuPointerGetAttribute(&ctx, attr, cuptr);
    NDLL_ENFORCE(result == CUDA_SUCCESS,
        "Used pointer from unknown CUDA context");
    cuCtxSetCurrent(ctx);
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    return dev;
  }

  static std::mutex m_;
  static UserStream * us_;

  std::unordered_map<int, cudaStream_t> streams_;
};
}  // namespace ndll

#endif  // NDLL_UTIL_USER_STREAM_H_
