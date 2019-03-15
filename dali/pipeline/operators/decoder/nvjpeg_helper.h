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
#include <memory>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/util/crop_window.h"
#include "dali/kernels/backend_tags.h"
#include "dali/kernels/common/copy.h"
#include "dali/image/image_factory.h"

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
  StateNvJPEG(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder_host, nvjpegJpegDecoder_t decoder_hybrid) {
    NVJPEG_CALL(nvjpegBufferPinnedCreate(handle, nullptr, &pinned_buffer));
    NVJPEG_CALL(nvjpegDecoderStateCreate(handle, decoder_host, &decoder_host_state));
    NVJPEG_CALL(nvjpegDecoderStateCreate(handle, decoder_hybrid, &decoder_hybrid_state));
    NVJPEG_CALL(nvjpegJpegStreamCreate(handle, &jpeg_stream));
  }

  ~StateNvJPEG() {
    nvjpegJpegStreamDestroy(jpeg_stream);
    nvjpegJpegStateDestroy(decoder_hybrid_state);
    nvjpegJpegStateDestroy(decoder_host_state);
    nvjpegBufferPinnedDestroy(pinned_buffer);
  }
  nvjpegBackend_t nvjpeg_backend;
  nvjpegBufferPinned_t pinned_buffer;
  nvjpegJpegState_t decoder_host_state;
  nvjpegJpegState_t decoder_hybrid_state;
  nvjpegJpegStream_t jpeg_stream;
};

template <typename T>
struct EncodedImageInfo {
  bool nvjpeg_support;
  T c;
  T widths[NVJPEG_MAX_COMPONENT];
  T heights[NVJPEG_MAX_COMPONENT];
  nvjpegChromaSubsampling_t subsampling;
  CropWindow crop_window;
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

template <typename StorageType>
void HostFallback(const uint8_t *data, int size, DALIImageType image_type, uint8_t *output_buffer,
                  cudaStream_t stream, std::string file_name, CropWindow crop_window) {
  std::unique_ptr<Image> img;
  try {
    img = ImageFactory::CreateImage(data, size, image_type);
    img->SetCropWindow(crop_window);
    img->Decode();
  } catch (std::runtime_error &e) {
    DALI_FAIL(e.what() + ". File: " + file_name);
  }
  const auto decoded = img->GetImage();
  const auto hwc = img->GetImageDims();
  const auto h = std::get<0>(hwc);
  const auto w = std::get<1>(hwc);
  const auto c = std::get<2>(hwc);

  kernels::copy<StorageType, kernels::StorageCPU>(output_buffer, decoded.get(), h * w * c, stream);
}


inline void WarmUpNvJPEG(nvjpegHandle_t handle, StateNvJPEG& state, nvjpegJpegDecoder_t decoder,
                  nvjpegOutputFormat_t output_format) {
  // run dummy
  static const char kEncodedJpeg[] =
      "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01\x01\x01\x00\x48"
      "\x00\x48\x00\x00\xFF\xFE\x00\x13\x43\x72\x65\x61\x74\x65\x64\x20"
      "\x77\x69\x74\x68\x20\x47\x49\x4D\x50\xFF\xDB\x00\x43\x00\x03\x02"
      "\x02\x03\x02\x02\x03\x03\x03\x03\x04\x03\x03\x04\x05\x08\x05\x05"
      "\x04\x04\x05\x0A\x07\x07\x06\x08\x0C\x0A\x0C\x0C\x0B\x0A\x0B\x0B"
      "\x0D\x0E\x12\x10\x0D\x0E\x11\x0E\x0B\x0B\x10\x16\x10\x11\x13\x14"
      "\x15\x15\x15\x0C\x0F\x17\x18\x16\x14\x18\x12\x14\x15\x14\xFF\xDB"
      "\x00\x43\x01\x03\x04\x04\x05\x04\x05\x09\x05\x05\x09\x14\x0D\x0B"
      "\x0D\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14"
      "\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14"
      "\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14"
      "\x14\x14\x14\xFF\xC2\x00\x11\x08\x00\x10\x00\x10\x03\x01\x11\x00"
      "\x02\x11\x01\x03\x11\x01\xFF\xC4\x00\x17\x00\x00\x03\x01\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\x06\xFF"
      "\xC4\x00\x16\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x02\x03\x04\xFF\xDA\x00\x0C\x03\x01\x00\x02\x10"
      "\x03\x10\x00\x00\x01\xDB\x9D\x35\xEA\x16\x00\xAF\xFF\xC4\x00\x1B"
      "\x10\x00\x02\x01\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x02\x04\x03\x01\x05\x11\x12\x14\xFF\xDA\x00\x08\x01\x01\x00"
      "\x01\x05\x02\x23\xC0\x44\x0B\xF5\xC7\x15\x5E\x6D\x7D\x4A\xE3\xFF"
      "\xC4\x00\x28\x11\x00\x01\x02\x04\x03\x08\x03\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x01\x02\x03\x00\x11\x21\x31\x04\x12\x61\x13\x22"
      "\x23\x41\x51\x71\x91\xA1\xB1\xC1\xF0\xFF\xDA\x00\x08\x01\x03\x01"
      "\x01\x3F\x01\x7F\x32\x9F\xD9\xA5\xC2\xB7\x09\x33\xD7\x49\xDA\x43"
      "\xA0\xF8\x86\xB3\x37\x89\x47\x12\xE6\xB5\xF5\xFB\xB4\x36\xD2\x96"
      "\xEA\x90\x99\x84\x13\x55\x4A\xE3\x48\x67\x7D\xF2\x70\xED\xDA\x92"
      "\xBD\xE9\xE0\x72\xFB\x8F\xFF\xC4\x00\x25\x11\x00\x01\x02\x05\x03"
      "\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x11\x00\x03"
      "\x04\x21\x31\x41\x51\x61\x14\x22\x71\x91\xA2\xB1\xF1\xFF\xDA\x00"
      "\x08\x01\x02\x01\x01\x3F\x01\x4D\x24\xB0\x84\x3D\xD4\x12\x5D\xF1"
      "\x93\x9F\x6E\x40\x6C\x34\x2A\x99\x33\xE8\xC9\x42\x5C\x07\xF5\xC6"
      "\xD8\xBF\x9D\xE1\x0B\x1D\x24\xB4\x01\xDA\xCF\xCA\xAE\x7E\x23\xEC"
      "\xF9\x89\xC2\x5D\x35\x38\x94\x95\x39\xD3\x80\x2F\x7F\xDD\xF4\x11"
      "\xFF\xC4\x00\x21\x10\x00\x02\x01\x04\x02\x02\x03\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x01\x02\x03\x00\x11\x12\x22\x13\x81\x41\x51"
      "\x61\xA1\xF1\xFF\xDA\x00\x08\x01\x01\x00\x06\x3F\x02\x97\x8A\x25"
      "\x58\x5D\x90\x29\x71\xB3\x59\x7C\x7A\xFC\xA9\xE1\x61\x16\x7C\x5B"
      "\x3B\x00\x33\x93\xE0\x77\xF5\x52\x4D\x0C\x8A\x5D\x14\x62\x1C\xD8"
      "\x26\xA2\xFD\xD6\xCD\x10\x58\xC6\x6D\x20\x7D\x4B\x7B\xBF\x55\xFF"
      "\xC4\x00\x1C\x10\x01\x00\x01\x05\x01\x01\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x01\x11\x00\x21\x31\x41\x61\x51\x91\xFF\xDA\x00"
      "\x08\x01\x01\x00\x01\x3F\x21\x38\x1B\x15\x22\x1F\x10\x4D\xF7\xA5"
      "\x3D\x84\xC6\x18\x96\x2D\x46\x11\xC8\x50\x0E\xA0\xF4\x43\xB6\x4F"
      "\xC0\x76\x92\xBC\x17\x8C\x2C\xDC\x27\x4B\x98\x54\x0A\xFF\xDA\x00"
      "\x0C\x03\x01\x00\x02\x00\x03\x00\x00\x00\x10\xEF\xEF\xFF\xC4\x00"
      "\x19\x11\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x01\x11\x21\x31\x00\x41\xFF\xDA\x00\x08\x01\x03\x01\x01"
      "\x3F\x10\xBC\x28\x90\xB4\x33\x43\x1A\x11\xAF\x14\x81\x84\x5A\x85"
      "\x0B\x82\x10\x8C\x80\x44\x28\x06\x96\xC3\x05\x60\x2D\x68\xE0\xC8"
      "\x26\xCB\xB7\x72\x8B\x52\x68\x4E\x9D\x08\xBF\x42\x20\xD9\x03\xAF"
      "\x7F\xFF\xC4\x00\x19\x11\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x01\x11\x21\x31\x00\x41\xFF\xDA\x00\x08"
      "\x01\x02\x01\x01\x3F\x10\x9B\xC6\x90\x10\x46\x67\xE2\x29\x20\x4D"
      "\x67\x93\x2A\xA1\x86\x00\x12\x08\x0E\x84\xAA\xAE\xA7\x58\x02\xA8"
      "\x2D\x03\x46\x0A\xBC\xD8\x6C\x3E\x49\x1C\x43\x0A\x2D\xAA\x85\x42"
      "\x81\x58\x04\x3D\xFF\xC4\x00\x18\x10\x01\x01\x01\x01\x01\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x11\x21\x00\x31\xFF\xDA"
      "\x00\x08\x01\x01\x00\x01\x3F\x10\x4A\x4D\x7C\x09\x32\x21\x8A\xD2"
      "\x8D\xE7\x12\xFE\x4B\x22\x5D\xC8\xAC\x08\x20\x41\x2F\x51\x3D\x5C"
      "\x8D\x0E\x93\x80\xA4\x78\x88\xE3\x18\xD3\x32\x7A\x74\x6B\x60\x2D"
      "\x54\x3F\xFF\xD9";

  NVJPEG_CALL(nvjpegJpegStreamParse(handle, (unsigned char *) kEncodedJpeg,
                                    sizeof(kEncodedJpeg) - 1, false, false, state.jpeg_stream));
  unsigned int C;
  unsigned int widths[NVJPEG_MAX_COMPONENT];
  unsigned int heights[NVJPEG_MAX_COMPONENT];
  nvjpegJpegStreamGetFrameDimensions(state.jpeg_stream, widths, heights);
  nvjpegJpegStreamGetComponentsNum(state.jpeg_stream, &C);

  nvjpegJpegState_t nvjpeg_state = GetNvjpegState(state);
  NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(nvjpeg_state, state.pinned_buffer));
  nvjpegDecodeParams_t params;
  NVJPEG_CALL(nvjpegDecodeParamsCreate(handle, &params));
  NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(params, output_format));
  NVJPEG_CALL(nvjpegDecodeParamsSetAllowCMYK(params, true));
  NVJPEG_CALL(nvjpegDecodeJpegHost(handle, decoder, nvjpeg_state,
                                    params, state.jpeg_stream));
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_
