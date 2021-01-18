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

#ifndef DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_TURING_OF_UTILS_H_
#define DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_TURING_OF_UTILS_H_

#include <string>
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"

namespace dali {

class NvofError : public std::runtime_error {
 public:
  explicit NvofError(NV_OF_STATUS result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(NV_OF_STATUS result) {
    switch (result) {
      case NV_OF_SUCCESS:
        return "NVIDIA Optical flow operation was successful";
      case NV_OF_ERR_OF_NOT_AVAILABLE:
        return "NVIDIA HW Optical flow functionality is not supported";
      case NV_OF_ERR_UNSUPPORTED_DEVICE:
        return "The device passed by the client to NVIDIA HW Optical flow is not supported";
      case NV_OF_ERR_DEVICE_DOES_NOT_EXIST:
        return "The device passed to the NVIDIA HW Optical flow API call is no longer available "
                "and needs to be reinitialized";
      case NV_OF_ERR_INVALID_PTR:
        return "One or more of the pointers passed to the NVIDIA HW Optical flow "
               "API call is invalid";
      case NV_OF_ERR_INVALID_PARAM:
        return "One or more of the parameters passed to the NVIDIA Optical flow API call "
               "are invalid";
      case NV_OF_ERR_INVALID_CALL:
        return "Nvidia Optical flow API call was made in wrong sequence/order";
      case NV_OF_ERR_INVALID_VERSION:
        return "An invalid struct version was used by the NVIDIA Optical flow client";
      case NV_OF_ERR_OUT_OF_MEMORY:
        return "NVIDIA Optical flow API call failed because it was unable to allocate  "
               "enough memory to perform the requested operation";
      case NV_OF_ERR_NOT_INITIALIZED:
        return "Nvidia Optical flow session has not been initialized or initialization has failed";
      case NV_OF_ERR_UNSUPPORTED_FEATURE:
        return "Unsupported parameter was passed by the client to NVIDIA Optical flow";
      case NV_OF_ERR_GENERIC:
        // fallthrough
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(NV_OF_STATUS result, const char *details) {
    if (details && *details) {
      return make_string("NVIDIA Optical flow error: ", result, " ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("NVIDIA Optical flow error: ", result, " ", ErrorString(result));
    }
  }


  NV_OF_STATUS result() const { return result_; }

 private:
  NV_OF_STATUS result_;
};

template <>
inline void cudaResultCheck<NV_OF_STATUS>(NV_OF_STATUS status) {
  switch (status) {
  case NV_OF_SUCCESS:
    return;
  default:
    throw dali::NvofError(status);
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_TURING_OF_UTILS_H_
