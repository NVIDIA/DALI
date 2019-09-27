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

constexpr int kSizeChunkSize = 4;
constexpr int kSizeChunkId = 4;
constexpr int kSizeWidth = 4;
constexpr int kSizeHeight = 4;
constexpr int kSizeBitDepth = 1;
constexpr int kSizeColorType = 1;
constexpr int kSizeCompressionMethod = 1;
constexpr int kSizeFilterMethod = 1;
constexpr int kSizeInterlaceMethod = 1;

constexpr int kOffsetChunkSize = 0;
constexpr int kOffsetChunkId = kOffsetChunkSize + kSizeChunkSize;
constexpr int kOffsetWidth = kOffsetChunkId + kSizeChunkId;
constexpr int kOffsetHeight = kOffsetWidth + kSizeWidth;
constexpr int kOffsetBitDepth = kOffsetHeight + kSizeHeight;
constexpr int kOffsetColorType = kOffsetBitDepth + kSizeBitDepth;
constexpr int kOffsetCompressionMethod = kOffsetColorType + kSizeColorType;
constexpr int kOffsetFilterMethod = kOffsetCompressionMethod + kSizeCompressionMethod;
constexpr int kOffsetInterlaceMethod = kOffsetFilterMethod + kSizeFilterMethod;

template <typename T, int offset, int nbytes>
T ReadValue(const uint8_t* data) {
  static_assert(std::is_unsigned<T>::value, "T must be an unsigned type");
  static_assert(sizeof(T) >= nbytes, "T can't hold the requested number of bytes");
  T value = 0;
  for (int i = 0; i < nbytes; i++) {
    value = (value << 8) + data[offset + i];
  }
  return value;
}

uint32_t ReadHeight(const uint8_t *data) {
  return ReadValue<uint32_t, kOffsetHeight, kSizeHeight>(data);
}

uint32_t ReadWidth(const uint8_t *data) {
  return ReadValue<uint32_t, kOffsetWidth, kSizeWidth>(data);
}

enum : uint8_t {
  PNG_COLOR_TYPE_GRAY       = 0,
  PNG_COLOR_TYPE_RGB        = 2,
  PNG_COLOR_TYPE_PALETTE    = 3,
  PNG_COLOR_TYPE_GRAY_ALPHA = 4,
  PNG_COLOR_TYPE_RGBA       = 6
};

uint8_t ReadColorType(const uint8_t *data) {
  return ReadValue<uint8_t, kOffsetColorType, kSizeColorType>(data);
}

int ReadNumberOfChannels(const uint8_t *data) {
  int color_type = ReadColorType(data);
  switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:
    case PNG_COLOR_TYPE_GRAY_ALPHA:
      return 1;
    case PNG_COLOR_TYPE_RGB:
    case PNG_COLOR_TYPE_PALETTE:  // 1 byte but it's converted to 3-channel BGR by OpenCV
    case PNG_COLOR_TYPE_RGBA:     // RGBA is converted to 3-channel BGR by OpenCV
      return 3;
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
