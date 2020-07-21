// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_CUDA_ERROR_H_
#define DALI_CORE_CUDA_ERROR_H_

#include <cuda_runtime_api.h>
#include <cstring>
#include <cstdio>
#include <string>
#include "dali/core/dynlink_cuda.h"
#include "dali/core/error_handling.h"

namespace dali {

class CUDAError : public std::runtime_error {
 public:
  explicit CUDAError(cudaError_t status) : std::runtime_error(get_message(status)), rt_err(status) {
  }
  explicit CUDAError(CUresult status) : std::runtime_error(get_message(status)), drv_err(status) {
  }

  CUresult drv_error() const noexcept { return drv_err; }
  cudaError_t rt_error() const noexcept { return rt_err; }

  bool is_drv_api() const noexcept { return drv_err != CUDA_SUCCESS; }
  bool is_rt_api() const noexcept { return rt_err != cudaSuccess; }

 private:
  CUresult drv_err = CUDA_SUCCESS;
  cudaError_t rt_err = cudaSuccess;

  static std::string get_message(CUresult status) {
    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &desc);
    std::ostringstream ss;
    if (!name) name = "<unknown error>";
    ss << "CUDA driver API error "
       << name << " (" << static_cast<unsigned>(status) << ")";
    if (desc && *desc) ss << ":\n" << desc;
    return ss.str();
  }

  static std::string get_message(cudaError_t status) {
    const char *name  = cudaGetErrorName(status);
    const char *desc = cudaGetErrorString(status);
    if (!name) name = "<unknown error>";
    std::ostringstream ss;
    ss << "CUDA runtime API error "
       << name << " (" << static_cast<unsigned>(status) << ")";
    if (desc && *desc) ss << ":\n" << desc;
    return ss.str();
  }
};

class CUDABadAlloc : public std::bad_alloc {
 public:
  CUDABadAlloc() {
    std::strncpy(message, "CUDA allocation failed", sizeof(message));
  }
  explicit CUDABadAlloc(size_t requested_size, bool host = false) {
    if (host) {
      std::snprintf(message, sizeof(message),
        "Can't allocate %zu bytes on host.", requested_size);
    } else {
      int dev = -1;
      cudaGetDevice(&dev);
      std::snprintf(message, sizeof(message),
        "Can't allocate %zu bytes on device %d.", requested_size, dev);
    }
  }
  const char *what() const noexcept override {
    return message;
  }
 private:
  char message[64];
};

template <typename Code>
inline void cudaResultCheck(Code code) {
  static_assert(!std::is_same<Code, Code>::value,
    "cudaResultCheck not implemented for this type of status code");
}

template <>
inline void cudaResultCheck<cudaError_t>(cudaError_t status) {
  switch (status) {
  case cudaSuccess:
    return;
  case cudaErrorMemoryAllocation:
    cudaGetLastError();  // clear the last error
    throw dali::CUDABadAlloc();
  default:
    cudaGetLastError();  // clear the last error
    throw dali::CUDAError(status);
  }
}

template <>
inline void cudaResultCheck<CUresult>(CUresult status) {
  switch (status) {
  case CUDA_SUCCESS:
    return;
  case CUDA_ERROR_OUT_OF_MEMORY:
    throw dali::CUDABadAlloc();
  default:
    throw dali::CUDAError(status);
  }
}

template <typename Code>
inline void cudaResultDestructorCheck(Code status) {
  cudaResultCheck(status);
}

template <>
inline void cudaResultDestructorCheck<cudaError_t>(cudaError_t status) {
  switch (status) {
  case cudaErrorCudartUnloading:
    return;
  default:
    cudaResultCheck(status);
  }
}

template <>
inline void cudaResultDestructorCheck<CUresult>(CUresult status) {
  switch (status) {
  case CUDA_ERROR_DEINITIALIZED:
    return;
  default:
    cudaResultCheck(status);
  }
}

}  // namespace dali

template <typename T>
inline void CUDA_CALL(T status) {
  return dali::cudaResultCheck(status);
}

template <typename T>
inline void CUDA_DTOR_CALL(T status) {
  return dali::cudaResultDestructorCheck(status);
}


#endif  // DALI_CORE_CUDA_ERROR_H_
