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

// TODO(spanev): Replace when it is available in the nvJPEG API
inline uint8_t GetJpegEncoding(const uint8_t* input, size_t size) {
  if (input[0] != 0xff || input[1] != 0xd8)
    return 0;
  const uint8_t* ptr = input + 2;
  const uint8_t* end = input + size;
  uint8_t marker = *ptr;
  ptr++;
  while (ptr < end) {
    do {
      // We ignore padding/custom metadata
      while (marker != 0xff && ptr != end) {
        marker = *ptr;
        ptr++;
      }
      if (ptr == end)
        return 0;
      marker = *ptr;
      ptr++;
    } while (marker == 0 || marker == 0xff);
    if (marker >= 0xc0 && marker <= 0xcf) {
      return marker;
    } else {
      // Next segment
      uint16_t segment_length = (*ptr << 8) + *(ptr+1);
      ptr += segment_length;
    }
  }
  return 0;
}

inline bool IsProgressiveJPEG(const uint8_t* raw_jpeg, size_t size) {
  const uint8_t segment_marker = GetJpegEncoding(raw_jpeg, size);
  constexpr uint8_t progressive_sof = 0xc2;
  return segment_marker == progressive_sof;
}

// Predicate to determine if the image should be decoded with the nvJPEG
// hybrid Huffman decoder instead of the nvjpeg host Huffman decoder
template <typename T>
inline bool ShouldUseHybridHuffman(EncodedImageInfo<T>& info,
                                   const uint8_t* input,
                                   size_t size,
                                   unsigned int threshold) {
  auto &roi = info.crop_window;
  unsigned int w = static_cast<unsigned int>(info.widths[0]);
  unsigned int h = static_cast<unsigned int>(roi ? (roi.y + roi.h)
                                                  : info.heights[0]);
  // TODO(spanev): replace it by nvJPEG API function when available in future release
  // We don't wanna call IsProgressiveJPEG if not needed
  return h*w > threshold && !IsProgressiveJPEG(input, size);
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

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_
