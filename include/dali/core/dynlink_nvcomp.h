// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_DYNLINK_NVCOMP_H_
#define DALI_CORE_DYNLINK_NVCOMP_H_

#include <nvcomp/lz4.h>
#include <string>
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"

bool nvCompIsSymbolAvailable(const char *symbol);

namespace dali {

class NvCompError : public std::runtime_error {
 public:
  explicit NvCompError(nvcompStatus_t  result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(nvcompStatus_t result) {
    switch (result) {
      case nvcompSuccess:
        return "The API call has finished successfully. Note that many of the calls are "
               "asynchronous and some of the errors may be seen only after synchronization.";
      case nvcompErrorInvalidValue:
        return "Invalid value provided to the API.";
      case nvcompErrorNotSupported:
        return "Operation not supported.";
      case nvcompErrorCannotDecompress:
        return "Cannot decompress provided input.";
      case nvcompErrorBadChecksum:
        return "Wrong checksum of the provided data.";
      case nvcompErrorCannotVerifyChecksums:
        return "Cannot verify checksum of the provided data.";
      case nvcompErrorOutputBufferTooSmall:
        return "Provided output buffer is too small.";
      case nvcompErrorWrongHeaderLength:
        return "Wrong header length of the provided data.";
      case nvcompErrorAlignment:
        return "Wrong alignment of the provided data.";
      case nvcompErrorChunkSizeTooLarge:
        return "Chunk size of the decoded data is too large.";
      case nvcompErrorCudaError:
        return "Unknown CUDA error.";
      case nvcompErrorInternal:
        return "Unknown nvCOMP error.";
#if NVCOMP_VER_MAJOR > 5 || (NVCOMP_VER_MAJOR == 5 && NVCOMP_VER_MINOR >= 1)
      case nvcompErrorBatchSizeTooLarge:
        return "Batch size is too large.";
#endif
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(nvcompStatus_t result, const char *details) {
    if (details && *details) {
      return make_string("nvComp error (", result, "): ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("nvComp error (", result, "): ", ErrorString(result));
    }
  }

  nvcompStatus_t result() const { return result_; }

 private:
  nvcompStatus_t result_;
};

template <>
inline void cudaResultCheck<nvcompStatus_t>(nvcompStatus_t status) {
  switch (status) {
  case nvcompSuccess:
    return;
  default:
    throw NvCompError(status);
  }
}

template <>
inline void cudaResultCheck<nvcompStatus_t>(nvcompStatus_t status, const string &extra) {
  switch (status) {
  case nvcompSuccess:
    return;
  default:
    throw NvCompError(status, extra.c_str());
  }
}

}  // namespace dali

#endif  // DALI_CORE_DYNLINK_NVCOMP_H_
