// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_DECODERS_NVJPEG2K_NVJPEG2K_HELPER_H_
#define DALI_IMGCODEC_DECODERS_NVJPEG2K_NVJPEG2K_HELPER_H_

#include <nvjpeg2k.h>
#include <string>
#include <memory>
#include "dali/core/error_handling.h"
#include "dali/core/unique_handle.h"
#include "dali/core/format.h"
#include "dali/core/common.h"
#include "dali/core/cuda_error.h"

namespace dali {
namespace imgcodec {

class NvJpeg2kError : public std::runtime_error {
 public:
  explicit NvJpeg2kError(nvjpeg2kStatus_t result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(nvjpeg2kStatus_t result) {
    switch (result) {
      case NVJPEG2K_STATUS_SUCCESS:
        return "The API call has finished successfully. Note that many of the calls are "
                "asynchronous and some of the errors may be seen only after synchronization.";
      case NVJPEG2K_STATUS_NOT_INITIALIZED :
        return "The library handle was not initialized.";
      case NVJPEG2K_STATUS_INVALID_PARAMETER:
        return "Wrong parameter was passed. For example, a null pointer as input data, or "
               "an invalid enum value";
      case NVJPEG2K_STATUS_BAD_JPEG:
        return "Cannot parse the JPEG2000 stream. Likely due to a corruption that cannot "
               "be handled";
      case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED:
        return "Attempting to decode a JPEG2000 stream that is not supported by "
               "the nvJPEG2000 library.";
      case NVJPEG2K_STATUS_ALLOCATOR_FAILURE:
        return "The user-provided allocator functions, for either memory allocation or "
               "for releasing the memory, returned a non-zero code.";
      case NVJPEG2K_STATUS_EXECUTION_FAILED:
        return "Error during the execution of the device tasks.";
      case NVJPEG2K_STATUS_ARCH_MISMATCH:
        return "The device capabilities are not enough for the set of input parameters provided.";
      case NVJPEG2K_STATUS_INTERNAL_ERROR:
        return "Unknown error occurred in the library.";
      case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return "API is not supported by the backend.";
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(nvjpeg2kStatus_t result, const char *details) {
    if (details && *details) {
      return make_string("nvJPEG2000 error (", result, "): ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("nvJPEG2000 error (", result, "): ", ErrorString(result));
    }
  }

  nvjpeg2kStatus_t result() const { return result_; }

 private:
  nvjpeg2kStatus_t result_;
};

struct NvJpeg2kHandle : public UniqueHandle<nvjpeg2kHandle_t, NvJpeg2kHandle> {
  DALI_INHERIT_UNIQUE_HANDLE(nvjpeg2kHandle_t, NvJpeg2kHandle);

  NvJpeg2kHandle() = default;

  NvJpeg2kHandle(nvjpeg2kDeviceAllocator_t *dev_alloc, nvjpeg2kPinnedAllocator_t *pin_alloc) {
    if (nvjpeg2kCreate(NVJPEG2K_BACKEND_DEFAULT, dev_alloc, pin_alloc, &handle_) !=
        NVJPEG2K_STATUS_SUCCESS) {
      handle_ = null_handle();
    }
  }

  static constexpr nvjpeg2kHandle_t null_handle() { return nullptr; }

  static void DestroyHandle(nvjpeg2kHandle_t handle) {
    nvjpeg2kDestroy(handle);
  }
};

struct NvJpeg2kStream : public UniqueHandle<nvjpeg2kStream_t, NvJpeg2kStream> {
  DALI_INHERIT_UNIQUE_HANDLE(nvjpeg2kStream_t, NvJpeg2kStream);

  static NvJpeg2kStream Create() {
    nvjpeg2kStream_t handle{};
    CUDA_CALL(nvjpeg2kStreamCreate(&handle));
    return NvJpeg2kStream(handle);
  }

  static constexpr nvjpeg2kStream_t null_handle() { return nullptr; }

  static void DestroyHandle(nvjpeg2kStream_t handle) {
    nvjpeg2kStreamDestroy(handle);
  }
};

struct NvJpeg2kDecodeState : public UniqueHandle<nvjpeg2kDecodeState_t, NvJpeg2kDecodeState> {
  DALI_INHERIT_UNIQUE_HANDLE(nvjpeg2kDecodeState_t, NvJpeg2kDecodeState);

  explicit NvJpeg2kDecodeState(nvjpeg2kHandle_t nvjpeg2k_handle) {
    CUDA_CALL(nvjpeg2kDecodeStateCreate(nvjpeg2k_handle, &handle_));
  }

  static constexpr nvjpeg2kDecodeState_t null_handle() { return nullptr; }

  static void DestroyHandle(nvjpeg2kDecodeState_t handle) {
    nvjpeg2kDecodeStateDestroy(handle);
  }
};

struct NvJpeg2kDecodeParams : public UniqueHandle<nvjpeg2kDecodeParams_t, NvJpeg2kDecodeParams> {
  DALI_INHERIT_UNIQUE_HANDLE(nvjpeg2kDecodeParams_t, NvJpeg2kDecodeParams);

  NvJpeg2kDecodeParams() {
    CUDA_CALL(nvjpeg2kDecodeParamsCreate(&handle_));
  }

  static constexpr nvjpeg2kDecodeParams_t null_handle() { return nullptr; }

  static void DestroyHandle(nvjpeg2kDecodeParams_t handle) {
    nvjpeg2kDecodeParamsDestroy(handle);
  }
};

}  // namespace imgcodec

template <>
inline void cudaResultCheck<nvjpeg2kStatus_t>(nvjpeg2kStatus_t status) {
  switch (status) {
  case NVJPEG2K_STATUS_SUCCESS:
    return;
  default:
    throw dali::imgcodec::NvJpeg2kError(status);
  }
}

template <>
inline void cudaResultCheck<nvjpeg2kStatus_t>(nvjpeg2kStatus_t status, const string &extra) {
  switch (status) {
  case NVJPEG2K_STATUS_SUCCESS:
    return;
  default:
    throw dali::imgcodec::NvJpeg2kError(status, extra.c_str());
  }
}

}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG2K_NVJPEG2K_HELPER_H_
