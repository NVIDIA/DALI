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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_NVJPEG2K_HELPER_H_
#define DALI_OPERATORS_DECODER_NVJPEG_NVJPEG2K_HELPER_H_

#if NVJPEG2K_ENABLED

#include <nvjpeg2k.h>
#include <string>
#include <memory>
#include "dali/core/error_handling.h"
#include "dali/core/unique_handle.h"
#include "dali/core/format.h"

namespace dali {

#define NVJPEG2K_CALL(code)                             \
  do {                                                  \
    nvjpeg2kStatus_t status = code;                     \
    if (status != NVJPEG2K_STATUS_SUCCESS) {            \
      auto error = make_string("NVJPEG2K error \"",     \
        static_cast<int>(status), "\"");                \
      DALI_FAIL(error);                                 \
    }                                                   \
  } while (0)

#define NVJPEG2K_CALL_EX(code, extra)                   \
  do {                                                  \
    nvjpeg2kStatus_t status = code;                     \
    std::string extra_info = extra;                     \
    if (status != NVJPEG2K_STATUS_SUCCESS) {            \
      auto error = make_string("NVJPEG2K error \"",     \
        static_cast<int>(status), "\" : ", extra_info); \
      DALI_FAIL(error);                                 \
    }                                                   \
  } while (0)

struct NvJPEG2KHandle : public UniqueHandle<nvjpeg2kHandle_t, NvJPEG2KHandle> {
  DALI_INHERIT_UNIQUE_HANDLE(nvjpeg2kHandle_t, NvJPEG2KHandle);

  NvJPEG2KHandle() = default;

  NvJPEG2KHandle(nvjpeg2kDeviceAllocator_t *dev_alloc, nvjpeg2kPinnedAllocator_t *pin_alloc) {
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

struct NvJPEG2KStream : public UniqueHandle<nvjpeg2kStream_t, NvJPEG2KStream> {
  DALI_INHERIT_UNIQUE_HANDLE(nvjpeg2kStream_t, NvJPEG2KStream);

  static NvJPEG2KStream Create() {
    nvjpeg2kStream_t handle{};
    NVJPEG2K_CALL(nvjpeg2kStreamCreate(&handle));
    return NvJPEG2KStream(handle);
  }

  static constexpr nvjpeg2kStream_t null_handle() { return nullptr; }

  static void DestroyHandle(nvjpeg2kStream_t handle) {
    nvjpeg2kStreamDestroy(handle);
  }
};

struct NvJPEG2KDecodeState : public UniqueHandle<nvjpeg2kDecodeState_t, NvJPEG2KDecodeState> {
  DALI_INHERIT_UNIQUE_HANDLE(nvjpeg2kDecodeState_t, NvJPEG2KDecodeState);

  explicit NvJPEG2KDecodeState(nvjpeg2kHandle_t nvjpeg2k_handle) {
    NVJPEG2K_CALL(nvjpeg2kDecodeStateCreate(nvjpeg2k_handle, &handle_));
  }

  static constexpr nvjpeg2kDecodeState_t null_handle() { return nullptr; }

  static void DestroyHandle(nvjpeg2kDecodeState_t handle) {
    nvjpeg2kDecodeStateDestroy(handle);
  }
};

}  // namespace dali

#endif  // NVJPEG2K_ENABLED

#endif  // DALI_OPERATORS_DECODER_NVJPEG_NVJPEG2K_HELPER_H_
