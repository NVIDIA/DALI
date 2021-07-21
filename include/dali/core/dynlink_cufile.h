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

#ifndef DALI_CORE_DYNLINK_CUFILE_H_
#define DALI_CORE_DYNLINK_CUFILE_H_

#include <cufile.h>
#include <string>
#include "dali/core/cuda_utils.h"
#include "dali/util/nvml_wrap.h"
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"

namespace dali {

class CufileError : public std::runtime_error {
 public:
  explicit CufileError(CUfileError_t result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(CUfileError_t result) {
    switch (result.err) {
      case CU_FILE_SUCCESS:
        return "cufile: operation was successful";
      case CU_FILE_DRIVER_NOT_INITIALIZED:
        return "nvidia-fs driver is not loaded";
      case CU_FILE_DRIVER_INVALID_PROPS:
        return "cufile: invalid property";
      case CU_FILE_DRIVER_UNSUPPORTED_LIMIT:
        return "cufile: property range error";
      case CU_FILE_DRIVER_VERSION_MISMATCH:
        return "nvidia-fs driver version mismatch";
      case CU_FILE_DRIVER_VERSION_READ_ERROR:
        return "nvidia-fs driver version read error";
      case CU_FILE_DRIVER_CLOSING:
        return "cufile: driver shutdown in progress";
      case CU_FILE_PLATFORM_NOT_SUPPORTED:
        return "GPUDirect Storage not supported on current platform";
      case CU_FILE_IO_NOT_SUPPORTED:
        return "GPUDirect Storage not supported on current file";
      case CU_FILE_DEVICE_NOT_SUPPORTED:
        return "GPUDirect Storage not supported on current GPU";
      case CU_FILE_NVFS_DRIVER_ERROR:
        return "nvidia-fs driver ioctl error";
      case CU_FILE_CUDA_DRIVER_ERROR:
        return "cufile: CUDA Driver API error";
      case CU_FILE_CUDA_POINTER_INVALID:
        return "cufile: invalid device pointer";
      case CU_FILE_CUDA_MEMORY_TYPE_INVALID:
        return "cufile: invalid pointer memory type";
      case CU_FILE_CUDA_POINTER_RANGE_ERROR:
        return "cufile: pointer range exceeds allocated address range";
      case CU_FILE_CUDA_CONTEXT_MISMATCH:
        return "cufile: cuda context mismatch";
      case CU_FILE_INVALID_MAPPING_SIZE:
        return "cufile: access beyond maximum pinned size";
      case CU_FILE_INVALID_MAPPING_RANGE:
        return "cufile: access beyond mapped size";
      case CU_FILE_INVALID_FILE_TYPE:
        return "cufile: unsupported file type";
      case CU_FILE_INVALID_FILE_OPEN_FLAG:
        return "cufile: unsupported file open flags";
      case CU_FILE_DIO_NOT_SET:
        return "cufile: fd direct IO not set";
      case CU_FILE_INVALID_VALUE:
        return "cufile: invalid arguments";
      case CU_FILE_MEMORY_ALREADY_REGISTERED:
        return "cufile: device pointer already registered";
      case CU_FILE_MEMORY_NOT_REGISTERED:
        return "cufile: device pointer lookup failure";
      case CU_FILE_PERMISSION_DENIED:
        return "cufile: driver or file access error";
      case CU_FILE_DRIVER_ALREADY_OPEN:
        return "cufile: driver is already open";
      case CU_FILE_HANDLE_NOT_REGISTERED:
        return "cufile: file descriptor is not registered";
      case CU_FILE_HANDLE_ALREADY_REGISTERED:
        return "cufile: file descriptor is already registered";
      case CU_FILE_DEVICE_NOT_FOUND:
        return "cufile: GPU device not found";
      case CU_FILE_INTERNAL_ERROR:
        return "cufile: internal error";
      case CU_FILE_GETNEWFD_FAILED:
        return "cufile: failed to obtain new file descriptor";
      case CU_FILE_NVFS_SETUP_ERROR:
        return "NVFS driver initialization error";
      case CU_FILE_IO_DISABLED:
        return "GPUDirect Storage disabled by config on current file";
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(CUfileError_t result, const char *details) {
    if (details && *details) {
      return make_string("cufile error: ", result.err, " ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("cufile error: ", result.err, " ", ErrorString(result));
    }
  }


  CUfileError_t result() const { return result_; }

 private:
  CUfileError_t result_;
};

template <>
inline void cudaResultCheck<CUfileError_t>(CUfileError_t status) {
  switch (status.err) {
  case CU_FILE_SUCCESS:
    return;
  default:
    throw dali::CufileError(status);
  }
}

}  // namespace dali

#endif  // DALI_CORE_DYNLINK_CUFILE_H_
