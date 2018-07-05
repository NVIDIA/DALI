// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/image/jpeg.h"
#ifdef DALI_USE_JPEG_TURBO
#include <turbojpeg.h>
#endif  // DALI_USE_JPEG_TURBO
#include "dali/util/ocv.h"

namespace dali {

namespace {
#define TJPG_CALL(code)                           \
  do {                                            \
    int error = code;                             \
    DALI_ASSERT(!error, tjGetErrorStr());         \
  } while (0)

#ifdef DALI_USE_JPEG_TURBO
void PrintSubsampling(int sampling) {
  switch (sampling) {
  case TJSAMP_444:
    cout << "sampling ratio: 444" << endl;
    break;
  case TJSAMP_422:
    cout << "sampling ratio: 422" << endl;
    break;
  case TJSAMP_420:
    cout << "sampling ratio: 420" << endl;
    break;
  case TJSAMP_GRAY:
    cout << "sampling ratio: gray" << endl;
    break;
  case TJSAMP_440:
    cout << "sampling ratio: 440" << endl;
    break;
  case TJSAMP_411:
    cout << "sampling ratio: 411" << endl;
    break;
  default:
    cout << "unknown sampling ratio" << endl;
  }
}
#endif  // DALI_USE_JPEG_TURBO

// Slightly modified from  https://github.com/apache/incubator-mxnet/blob/master/plugin/opencv/cv_api.cc
// http://www.64lines.com/jpeg-width-height
// Gets the JPEG size from the array of data passed to the function, file reference: http://www.obrador.com/essentialjpeg/headerinfo.htm
bool get_jpeg_size(const uint8 *data, size_t data_size, int *height, int *width) {
  // Check for valid JPEG image
  unsigned int i = 0;  // Keeps track of the position within the file
  if (data[i] == 0xFF && data[i+1] == 0xD8 && data[i+2] == 0xFF && data[i+3] == 0xE0) {
    i += 4;
    // Check for valid JPEG header (null terminated JFIF)
    if (data[i+2] == 'J' && data[i+3] == 'F' && data[i+4] == 'I'
        && data[i+5] == 'F' && data[i+6] == 0x00) {
      // Retrieve the block length of the first block since
      // the first block will not contain the size of file
      uint16_t block_length = data[i] * 256 + data[i+1];
      while (i < data_size) {
        i+=block_length;  // Increase the file index to get to the next block
        if (i >= data_size) return false;  // Check to protect against segmentation faults
        if (data[i] != 0xFF) return false;  // Check that we are truly at the start of another block
        if (data[i+1] == 0xC0) {
          // 0xFFC0 is the "Start of frame" marker which contains the file size
          // The structure of the 0xFFC0 block is quite simple
          // [0xFFC0][ushort length][uchar precision][ushort x][ushort y]
          *height = data[i+5]*256 + data[i+6];
          *width = data[i+7]*256 + data[i+8];
          return true;
        } else {
          i+=2;  // Skip the block marker
          block_length = data[i] * 256 + data[i+1];  // Go to the next block
        }
      }
      return false;  // If this point is reached then no size was found
    } else {
      return false;  // Not a valid JFIF string
    }
  } else {
    return false;  // Not a valid SOI header
  }
}

}  // namespace

bool CheckIsJPEG(const uint8 *jpeg, int) {
  if ((jpeg[0] == 255) && (jpeg[1] == 216)) {
    return true;
  }
  return false;
}

DALIError_t GetJPEGImageDims(const uint8 *jpeg, int size, int *h, int *w) {
  DALI_ENFORCE(get_jpeg_size(jpeg, size, h, w));
  return DALISuccess;
}

DALIError_t DecodeJPEGHost(const uint8 *jpeg, int size,
    DALIImageType type, Tensor<CPUBackend>* image) {
  int h, w;
  int c = (type == DALI_GRAY) ? 1 : 3;

  DALI_CALL(GetJPEGImageDims(jpeg, size, &h, &w));

#ifndef NDEBUG
  DALI_ASSERT(jpeg != nullptr);
  DALI_ASSERT(size > 0);
  DALI_ASSERT(h > 0);
  DALI_ASSERT(w > 0);
  DALI_ASSERT(image != nullptr);
  DALI_ASSERT(CheckIsJPEG(jpeg, size));
#endif

  // resize the output tensor
  image->Resize({h, w, c});
  // force allocation
  image->mutable_data<uint8_t>();

#ifdef DALI_USE_JPEG_TURBO
  // with tJPG
  tjhandle handle = tjInitDecompress();
  TJPF pixel_format;
  if (type == DALI_RGB) {
    pixel_format = TJPF_RGB;
  } else if (type == DALI_BGR) {
    pixel_format = TJPF_BGR;
  } else if (type == DALI_GRAY) {
    pixel_format = TJPF_GRAY;
  } else {
    DALI_RETURN_ERROR("Unsupported image type.");
  }

  auto error = tjDecompress2(handle, jpeg, size,
               image->mutable_data<uint8_t>(),
               w, 0, h, pixel_format, 0);

  tjDestroy(handle);

#else  // DALI_USE_JPEG_TURBO
  // without tJPG
  const int error = 1;  // since tJPG is absent

#endif  // DALI_USE_JPEG_TURBO

  // fallback to opencv if tJPG decode fails or absent
  if (error) {
    cv::Mat dst(h, w, (c == 1) ? CV_8UC1: CV_8UC3,
                image->raw_mutable_data());

    cv::Mat ret = cv::imdecode(
        CreateMatFromPtr(1, size, CV_8UC1, reinterpret_cast<const char*>(jpeg)),
        (c == 1) ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR,
        &dst);
    if (ret.empty())  // Empty Mat is returned on decoding failure
        DALI_RETURN_ERROR("OpenCV decoding fail.");

    // if RGB needed, permute from BGR
    if (type == DALI_RGB) {
      cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    }
  }

  return DALISuccess;
}

}  // namespace dali
