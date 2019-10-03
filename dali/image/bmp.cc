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

#include "dali/image/bmp.h"
#include "dali/core/byte_io.h"
#include "dali/core/format.h"

namespace dali {

namespace  {

enum BmpCompressionType {
  BMP_COMPRESSION_RGB       = 0,
  BMP_COMPRESSION_RLE8      = 1,
  BMP_COMPRESSION_RLE4      = 2,
  BMP_COMPRESSION_BITFIELDS = 3
};

// similar logic to what OpenCv does to determine the output number of channels
int number_of_channels(int bpp, int compression_type) {
  if (compression_type == BMP_COMPRESSION_RGB) {
    if (bpp == 1 || bpp == 4 || bpp == 8 || bpp == 24)
      return 3;
    else if (bpp == 32)
      return 4;
  } else if (compression_type == BMP_COMPRESSION_BITFIELDS) {
    if (bpp == 16)
      return 3;
    else if (bpp == 32)
      return 4;
    else
      return 1;
  } else if (compression_type == BMP_COMPRESSION_RLE8) {
    return (bpp == 4) ? 3 : 1;
  } else if (compression_type == BMP_COMPRESSION_RLE4) {
    return (bpp == 8) ? 3 : 1;
  }

  DALI_WARN(make_string(
    "configuration not supported. bpp:", bpp, "compression_type:", compression_type));
  return 0;
}

}  // namespace

// https://en.wikipedia.org/wiki/BMP_file_format#DIB_header_(bitmap_information_header)

BmpImage::BmpImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type)
  : GenericImage(encoded_buffer, length, image_type) {}

Image::Shape BmpImage::PeekShapeImpl(const uint8_t *bmp, size_t length) const {
  DALI_ENFORCE(bmp != nullptr);

  uint32_t header_size = ReadValueLE<uint32_t>(bmp + 14);
  int64_t h = 0, w = 0, c = 0;
  int bpp = 0, compression_type = BMP_COMPRESSION_RGB;
  if (length >= 22 && header_size == 12) {
    // BITMAPCOREHEADER:
    // | 32u header | 16u width | 16u height | 16u number of color planes | 16u bits per pixel
    w = ReadValueLE<uint16_t>(bmp + 18);
    h = ReadValueLE<uint16_t>(bmp + 20);
    bpp = ReadValueLE<uint16_t>(bmp + 24);
  } else if (length >= 26 && header_size >= 40) {
    // BITMAPINFOHEADER and later:
    // | 32u header | 32s width | 32s height | 16u number of color planes | 16u bits per pixel
    // | 32u compression type
    w = ReadValueLE<int32_t>(bmp + 18);
    h = abs(ReadValueLE<int32_t>(bmp + 22));
    bpp = ReadValueLE<uint16_t>(bmp + 28);
    compression_type = ReadValueLE<uint32_t>(bmp + 30);
  }
  c = number_of_channels(bpp, compression_type);
  return {h, w, c};
}


}  // namespace dali
