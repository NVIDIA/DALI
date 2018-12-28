#include <memory>

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
// Final adjustments based on https://github.com/scardine/image_size
bool get_jpeg_size(const uint8 *data, size_t data_size, int *height, int *width) {
  // Check for valid JPEG image
  unsigned int i = 0;  // Keeps track of the position within the file
  if (data[i] == 0xFF && data[i + 1] == 0xD8) {
    i += 4;
    // Retrieve the block length of the first block since
    // the first block will not contain the size of file
    uint16_t block_length = data[i] * 256 + data[i + 1];
    while (i < data_size) {
      i += block_length;  // Increase the file index to get to the next block
      if (i >= data_size) return false;  // Check to protect against segmentation faults
      if (data[i] != 0xFF) return false;  // Check that we are truly at the start of another block
      if (data[i + 1] >= 0xC0 && data[i + 1] <= 0xC3) {
        // 0xFFC0 is the "Start of frame" marker which contains the file size
        // The structure of the 0xFFC0 block is quite simple
        // [0xFFC0][ushort length][uchar precision][ushort x][ushort y]
        *height = data[i + 5] * 256 + data[i + 6];
        *width = data[i + 7] * 256 + data[i + 8];
        return true;
      } else {
        i += 2;  // Skip the block marker
        block_length = data[i] * 256 + data[i + 1];  // Go to the next block
      }
    }
    return false;  // If this point is reached then no size was found
  } else {
    return false;  // Not a valid SOI header
  }
}

}  // namespace


JpegImage::JpegImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        GenericImage(encoded_buffer, length, image_type) {
#ifdef DALI_USE_JPEG_TURBO
  tjhandle_ = tjInitDecompress();
  if (!tjhandle_) {
    std::stringstream ss;
    ss << "Failed jpeg-turbo initialization: " << tjGetErrorStr();
    DALI_FAIL(ss.str());
  }
#endif  // DALI_USE_JPEG_TURBO
}


JpegImage::~JpegImage() {
#ifdef DALI_USE_JPEG_TURBO
  tjDestroy(tjhandle_);
#endif  // DALI_USE_JPEG_TURBO
}


std::pair<std::shared_ptr<uint8_t>, Image::ImageDims>
JpegImage::DecodeImpl(DALIImageType type, const uint8 *jpeg, size_t length) const {
  const int c = (type == DALI_GRAY) ? 1 : 3;

  const auto dims = PeekDims(jpeg, length);
  const auto h = std::get<0>(dims);
  const auto w = std::get<1>(dims);

  DALI_ENFORCE(jpeg != nullptr);
  DALI_ENFORCE(length > 0);
  DALI_ENFORCE(h > 0);
  DALI_ENFORCE(w > 0);

#ifdef DALI_USE_JPEG_TURBO
  // not supported by turbojpeg
  if (type == DALI_YCbCr) {
    return GenericImage::DecodeImpl(type, jpeg, length);
  }

  // with tJPG
  TJPF pixel_format;
  if (type == DALI_RGB) {
    pixel_format = TJPF_RGB;
  } else if (type == DALI_BGR) {
    pixel_format = TJPF_BGR;
  } else if (type == DALI_GRAY) {
    pixel_format = TJPF_GRAY;
  } else {
    DALI_FAIL("Unsupported image type.");
  }

  std::shared_ptr<uint8_t> decoded_image(new uint8_t[h * w * c],
                            [=](uint8_t* data) { delete [] data;});

  auto error = tjDecompress2(tjhandle_, jpeg, length,
                             decoded_image.get(),
                             w, 0, h, pixel_format, 0);
  DALI_ENFORCE(error == 0 || error == -1, "Unexpected value");

  if (error == 0) {
    return std::make_pair(decoded_image, std::make_tuple(h, w, c));
  } else {
    // Error occurred during jpeg-turbo decompress. Falling back to Generic decode
    return GenericImage::DecodeImpl(type, jpeg, length);
  }
#else  // DALI_USE_JPEG_TURBO
  return GenericImage::DecodeImpl(type, jpeg, length);
#endif  // DALI_USE_JPEG_TURBO
}


Image::ImageDims JpegImage::PeekDims(const uint8_t *encoded_buffer, size_t length) const {
  int height, width;
  DALI_ENFORCE(get_jpeg_size(encoded_buffer, length, &height, &width));
  return std::make_tuple(height, width, 0);  // TODO(mszolucha): fill channels value
}

}  // namespace dali
