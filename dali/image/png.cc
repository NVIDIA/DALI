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

#include "dali/image/png.h"

namespace dali {

// Assume chunk points to a 4-byte value
int ReadIntFromPNG(const uint8 *chunk) {
  // reverse the bytes, cast
  return (unsigned int) (chunk[0] << 24 | chunk[1] << 16 | chunk[2] << 8 | chunk[3]);
}


PngImage::PngImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        GenericImage(encoded_buffer, length, image_type) {
}


Image::ImageDims PngImage::PeekDims(const uint8_t *encoded_buffer, size_t length) const {
  DALI_ENFORCE(encoded_buffer);
  DALI_ENFORCE(length >= 16);

  // IHDR needs to be the first chunk
  const uint8_t *IHDR = encoded_buffer + 8;
  const uint8_t *png_dimens = IHDR;
  if (IHDR[4] != 'I' || IHDR[5] != 'H' || IHDR[6] != 'D' || IHDR[7] != 'R') {
    // no IHDR, older PNGs format
    png_dimens = encoded_buffer;
  }

  DALI_ENFORCE(static_cast<int>(length) >= png_dimens - encoded_buffer + 16u);

  // Layout:
  // 4 bytes: chunk size (should be 13 bytes for IHDR)
  // 4 bytes: Chunk Identifier (should be "IHDR")
  // 4 bytes: Width
  // 4 bytes: Height
  // 1 byte : Bit Depth
  // 1 byte : Color Type
  // 1 byte : Compression method
  // 1 byte : Filter method
  // 1 byte : Interlace method

  const auto W = ReadIntFromPNG(png_dimens + 8);
  const auto H = ReadIntFromPNG(png_dimens + 12);
  // TODO(mszolucha): fill channels count
  const auto C = 0;
  return std::make_tuple(H, W, C);
}


}  // namespace dali
