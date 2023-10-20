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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_HELPER_H_
#define DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_HELPER_H_

#include <nvjpeg.h>

#include <string>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"
#include "dali/util/crop_window.h"
#include "dali/core/backend_tags.h"
#include "dali/kernels/common/copy.h"
#include "dali/image/image_factory.h"

#if WITH_DYNAMIC_NVJPEG_ENABLED
  bool nvjpegIsSymbolAvailable(const char *name);
#else
  #define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace dali {

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

template <>
inline void cudaResultCheck<nvjpegStatus_t>(nvjpegStatus_t status) {
  switch (status) {
  case NVJPEG_STATUS_SUCCESS:
    return;
  default:
    throw dali::NvjpegError(status);
  }
}

template <>
inline void cudaResultCheck<nvjpegStatus_t>(nvjpegStatus_t status, const string &extra) {
  switch (status) {
  case NVJPEG_STATUS_SUCCESS:
    return;
  default:
    throw dali::NvjpegError(status, extra.c_str());
  }
}

// Obtain nvJPEG library version or -1 if it is not available
int nvjpegGetVersion();

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
  unsigned int h = static_cast<unsigned int>(
    roi ? (roi.anchor[0] + roi.shape[0]) : info.heights[0]);
  // TODO(spanev): replace it by nvJPEG API function when available in future release
  // We don't wanna call IsProgressiveJPEG if not needed
  return h*w > threshold && !IsProgressiveJPEG(input, size);
}

template <typename StorageType>
void HostFallback(const uint8_t *data, int size, DALIImageType image_type, uint8_t *output_buffer,
                  cudaStream_t stream, std::string file_name, CropWindow crop_window,
                  bool use_fast_idct) {
  std::unique_ptr<Image> img;
  try {
    img = ImageFactory::CreateImage(data, size, image_type);
    img->SetCropWindow(crop_window);
    img->SetUseFastIdct(use_fast_idct);
    img->Decode();
  } catch (std::exception &e) {
    DALI_FAIL(e.what() + ". File: " + file_name);
  }
  const auto decoded = img->GetImage();
  const auto shape = img->GetShape();
  kernels::copy<StorageType, StorageCPU>(output_buffer, decoded.get(), volume(shape), stream);
}

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_NVJPEG_HELPER_H_
