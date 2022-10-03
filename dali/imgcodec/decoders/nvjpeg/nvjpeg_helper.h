// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_DECODERS_NVJPEG_NVJPEG_HELPER_H_
#define DALI_IMGCODEC_DECODERS_NVJPEG_NVJPEG_HELPER_H_

#include <nvjpeg.h>

#include <string>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"

namespace dali {
namespace imgcodec {

class NvjpegError : public std::runtime_error {
 public:
  explicit NvjpegError(nvjpegStatus_t result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(nvjpegStatus_t result) {
    switch (result) {
      case NVJPEG_STATUS_SUCCESS:
        return "The API call has finished successfully. Note that many of the calls are "
               "asynchronous and some of the errors may be seen only after synchronization.";
      case NVJPEG_STATUS_NOT_INITIALIZED:
        return "The library handle was not initialized.";
      case NVJPEG_STATUS_INVALID_PARAMETER:
        return "Wrong parameter was passed. For example, a null pointer as input data, or "
               "an image index not in the allowed range.";
      case NVJPEG_STATUS_BAD_JPEG:
        return "Cannot parse the JPEG stream. Check that the encoded JPEG stream and its "
               "size parameters are correct.";
      case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return "Attempting to decode a JPEG stream that is not supported by the nvJPEG library.";
      case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return "The user-provided allocator functions, for either memory allocation or for "
               "releasing the memory, returned a non-zero code.";
      case NVJPEG_STATUS_EXECUTION_FAILED:
        return "Error during the execution of the device tasks.";
      case NVJPEG_STATUS_ARCH_MISMATCH:
        return "The device capabilities are not enough for the set of input parameters "
               "provided (input parameters such as backend, encoded stream parameters, "
               "output format).";
      case NVJPEG_STATUS_INTERNAL_ERROR:
        return "Error during the execution of the device tasks.";
      case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return "Not supported.";
#if NVJPEG_VER_MAJOR > 11 || \
    (NVJPEG_VER_MAJOR == 11 && NVJPEG_VER_MINOR >= 6)
      case NVJPEG_STATUS_INCOMPLETE_BITSTREAM :
        return "Bitstream input data incomplete.";
#endif
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(nvjpegStatus_t result, const char *details) {
    if (details && *details) {
      return make_string("nvJPEG error (", result, "): ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("nvJPEG error (", result, "): ", ErrorString(result));
    }
  }


  nvjpegStatus_t result() const { return result_; }

 private:
  nvjpegStatus_t result_;
};

inline nvjpegOutputFormat_t GetFormat(DALIImageType type) {
  switch (type) {
    case DALI_RGB:
    case DALI_ANY_DATA:
    case DALI_YCbCr:  // purposedly not using NVJPEG_OUTPUT_YUV, as we want to control the
                      // definition of YCbCr to be consistent with the host backend
      return NVJPEG_OUTPUT_RGBI;
    case DALI_BGR:
      return NVJPEG_OUTPUT_BGRI;
    case DALI_GRAY:
      return NVJPEG_OUTPUT_Y;
    default:
      return NVJPEG_OUTPUT_FORMAT_MAX;  // doesn't matter (will fallback to host decoder)
  }
}

// TODO(spanev): Replace when it is available in the nvJPEG API
}  // namespace imgcodec

template <>
inline void cudaResultCheck<nvjpegStatus_t>(nvjpegStatus_t status) {
  switch (status) {
  case NVJPEG_STATUS_SUCCESS:
    return;
  default:
    throw imgcodec::NvjpegError(status);
  }
}

template <>
inline void cudaResultCheck<nvjpegStatus_t>(nvjpegStatus_t status, const string &extra) {
  switch (status) {
  case NVJPEG_STATUS_SUCCESS:
    return;
  default:
    throw imgcodec::NvjpegError(status, extra.c_str());
  }
}

}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG_NVJPEG_HELPER_H_
