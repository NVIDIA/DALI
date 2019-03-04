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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_

#include <nvjpeg.h>

#include <string>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/util/crop_window.h"

namespace dali {

#define NVJPEG_CALL(code)                                    \
  do {                                                       \
    nvjpegStatus_t status = code;                            \
    if (status != NVJPEG_STATUS_SUCCESS) {                   \
      dali::string error = dali::string("NVJPEG error \"") + \
        std::to_string(static_cast<int>(status)) + "\"";     \
      DALI_FAIL(error);                                      \
    }                                                        \
  } while (0)

#define NVJPEG_CALL_EX(code, extra)                          \
  do {                                                       \
    nvjpegStatus_t status = code;                            \
    string extra_info = extra;                               \
    if (status != NVJPEG_STATUS_SUCCESS) {                   \
      dali::string error = dali::string("NVJPEG error \"") + \
        std::to_string(static_cast<int>(status)) + "\"" +    \
        " " + extra_info;                                    \
      DALI_FAIL(error);                                      \
    }                                                        \
  } while (0)

struct StateNvJPEG {
  nvjpegBackend_t nvjpeg_backend;
  nvjpegBufferPinned_t pinned_buffer;
  nvjpegJpegState_t decoder_host_state;
  nvjpegJpegState_t decoder_hybrid_state;
  nvjpegJpegStream_t jpeg_stream;
};

struct EncodedImageInfo {
  bool nvjpeg_support;
  int c;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  nvjpegChromaSubsampling_t subsampling;
  CropWindow crop_window[NVJPEG_MAX_COMPONENT];
};

inline nvjpegJpegState_t GetNvjpegState(const StateNvJPEG& state) {
  switch (state.nvjpeg_backend) {
    case NVJPEG_BACKEND_HYBRID:
      return state.decoder_host_state;
    case NVJPEG_BACKEND_GPU_HYBRID:
      return state.decoder_hybrid_state;
    default:
      DALI_FAIL("Unknown nvjpegBackend_t "
                + std::to_string(state.nvjpeg_backend));
  }
}

inline nvjpegOutputFormat_t GetFormat(DALIImageType type) {
  switch (type) {
    case DALI_RGB:
      return NVJPEG_OUTPUT_RGBI;
    case DALI_BGR:
      return NVJPEG_OUTPUT_BGRI;
    case DALI_GRAY:
      return NVJPEG_OUTPUT_Y;
    default:
      DALI_FAIL("Unknown output format");
  }
}
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_
