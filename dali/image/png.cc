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
#include "dali/core/byte_io.h"

namespace dali {

namespace {

  // https://www.w3.org/TR/PNG-Chunks.html

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

constexpr int kOffsetChunkSize = 0;
constexpr int kOffsetChunkId = kOffsetChunkSize + sizeof(uint32_t);
constexpr int kOffsetWidth = kOffsetChunkId + sizeof(uint32_t);
constexpr int kOffsetHeight = kOffsetWidth + sizeof(uint32_t);
constexpr int kOffsetBitDepth = kOffsetHeight + sizeof(uint32_t);
constexpr int kOffsetColorType = kOffsetBitDepth + sizeof(uint8_t);
constexpr int kOffsetCompressionMethod = kOffsetColorType + sizeof(uint8_t);
constexpr int kOffsetFilterMethod = kOffsetCompressionMethod + sizeof(uint8_t);
constexpr int kOffsetInterlaceMethod = kOffsetFilterMethod + sizeof(uint8_t);

uint32_t ReadHeight(const uint8_t *data) {
  return ReadValueBE<uint32_t>(data + kOffsetHeight);
}

uint32_t ReadWidth(const uint8_t *data) {
  return ReadValueBE<uint32_t>(data + kOffsetWidth);
}

enum : uint8_t {
  PNG_COLOR_TYPE_GRAY       = 0,
  PNG_COLOR_TYPE_RGB        = 2,
  PNG_COLOR_TYPE_PALETTE    = 3,
  PNG_COLOR_TYPE_GRAY_ALPHA = 4,
  PNG_COLOR_TYPE_RGBA       = 6
};

uint8_t ReadColorType(const uint8_t *data) {
  return ReadValueBE<uint8_t>(data + kOffsetColorType);
}

int ReadNumberOfChannels(const uint8_t *data) {
  int color_type = ReadColorType(data);
  switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:
    case PNG_COLOR_TYPE_GRAY_ALPHA:
      return 1;
    case PNG_COLOR_TYPE_RGB:
    case PNG_COLOR_TYPE_PALETTE:  // 1 byte but it's converted to 3-channel BGR by OpenCV
      return 3;
    case PNG_COLOR_TYPE_RGBA:
      return 4;
    default:
      DALI_FAIL("color type not supported: " + std::to_string(color_type));
  }
  return 0;
}

}  // namespace


PngImage::PngImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        GenericImage(encoded_buffer, length, image_type) {
}


Image::Shape PngImage::PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const {
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

  const int64_t W = ReadWidth(png_dimens);
  const int64_t H = ReadHeight(png_dimens);
  const int64_t C = ReadNumberOfChannels(png_dimens);
  return {H, W, C};
}


}  // namespace dali
