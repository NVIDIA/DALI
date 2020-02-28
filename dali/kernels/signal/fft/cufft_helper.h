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

#ifndef DALI_KERNELS_SIGNAL_FFT_CUFFT_HELPER_H_
#define DALI_KERNELS_SIGNAL_FFT_CUFFT_HELPER_H_

#include <cufft.h>
#include <utility>
#include <string>
#include "dali/core/cuda_error.h"
#include "dali/core/unique_handle.h"
#include "dali/core/format.h"

namespace dali {

class CUFFTError : public std::runtime_error {
 public:
  explicit CUFFTError(cufftResult result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(cufftResult result) {
    switch (result) {
      case CUFFT_SUCCESS:
        return "cuFFT operation was successful";
      case CUFFT_INVALID_PLAN:
        return "cuFFT was passed an invalid plan handle";
      case CUFFT_ALLOC_FAILED:
        return "cuFFT failed to allocate memory";
      case CUFFT_INVALID_TYPE:
        return "Invalid type";
      case CUFFT_INVALID_VALUE:
        return "Invalid pointer or parameter";
      case CUFFT_INTERNAL_ERROR:
        return "Driver or internal cuFFT library error";
      case CUFFT_EXEC_FAILED:
        return "Failed to execute an FFT on the GPU";
      case CUFFT_SETUP_FAILED:
        return "The cuFFT library failed to initialize";
      case CUFFT_INVALID_SIZE:
        return "Invalid transform size specified";
      case CUFFT_UNALIGNED_DATA:
        return "Unaligned data";
      case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "Missing parameters in call";
      case CUFFT_INVALID_DEVICE:
        return "Execution of a plan was on different GPU than plan creation";
      case CUFFT_PARSE_ERROR:
        return "Internal plan database error";
      case CUFFT_NO_WORKSPACE:
        return "No workspace has been provided prior to plan execution";
      case CUFFT_NOT_IMPLEMENTED:
        return "Function does not implement functionality for parameters given";
      case CUFFT_LICENSE_ERROR:
        return "License error";
      case CUFFT_NOT_SUPPORTED:
        return "Operation is not supported for parameters given";
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(cufftResult result, const char *details) {
    if (details && *details) {
      return make_string("CUFFT error: ", result, " ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("CUFFT error: ", result, " ", ErrorString(result));
    }
  }


  cufftResult result() const { return result_; }

 private:
  cufftResult result_;
};

class CUFFTBadAlloc : public CUDABadAlloc {};

template <>
inline void cudaResultCheck<cufftResult>(cufftResult status) {
  switch (status) {
  case CUFFT_SUCCESS:
    return;
  case CUFFT_ALLOC_FAILED:
    throw dali::CUDABadAlloc();
  default:
    throw dali::CUFFTError(status);
  }
}

struct CUFFTHandle : public UniqueHandle<cufftHandle, CUFFTHandle> {
  DALI_INHERIT_UNIQUE_HANDLE(cufftHandle, CUFFTHandle)

  static void DestroyHandle(cufftHandle handle) {
    CUDA_CALL(cufftDestroy(handle));
  }
};

}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_CUFFT_HELPER_H_
