// Copyright (c) 2017-2018, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <typename T>
T ConsumeValue(const uint8_t*& ptr) {
  auto value = ReadValueLE<T>(ptr);
  ptr += sizeof(T);
  return value;
}

bool is_color_palette(const uint8_t* palette_start, size_t ncolors, size_t palette_entry_size) {
  const uint8_t* palette_end = palette_start + ncolors * palette_entry_size;
  for (auto p = palette_start; p < palette_end; p += palette_entry_size) {
    const auto b = p[0], g = p[1], r = p[2];  // a = p[3];
    if (b != g || b != r)
      return true;
  }
  return false;
}

int number_of_channels(int bpp, int compression_type,
                       const uint8_t* palette_start = nullptr, size_t ncolors = 0,
                       size_t palette_entry_size = 0) {
  if (compression_type == BMP_COMPRESSION_RGB || compression_type == BMP_COMPRESSION_RLE8) {
    if (bpp <= 8 && ncolors <= (1_uz << bpp)) {
      return is_color_palette(palette_start, ncolors, palette_entry_size) ? 3 : 1;
    } else if (bpp == 24) {
      return 3;
    } else if (bpp == 32) {
      return 4;
    }
  } else if (compression_type == BMP_COMPRESSION_BITFIELDS) {
    if (bpp == 16) {
      return 3;
    } else if (bpp == 32) {
      return 4;
    }
  }

  DALI_WARN(make_string(
    "configuration not supported. bpp: ", bpp, " compression_type:", compression_type,
    "ncolors:", ncolors));
  return 0;
}

}  // namespace

// https://en.wikipedia.org/wiki/BMP_file_format#DIB_header_(bitmap_information_header)

BmpImage::BmpImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type)
  : GenericImage(encoded_buffer, length, image_type) {}

Image::Shape BmpImage::PeekShapeImpl(const uint8_t *bmp, size_t length) const {
  DALI_ENFORCE(bmp != nullptr);
  DALI_ENFORCE(length >= 18);
  auto ptr = bmp + 14;
  uint32_t header_size = ConsumeValue<uint32_t>(ptr);
  int64_t h = 0, w = 0, c = 0;
  int bpp = 0, compression_type = BMP_COMPRESSION_RGB;
  const uint8_t* palette_start = nullptr;
  size_t ncolors = 0;
  size_t palette_entry_size = 0;
  if (length >= 26 && header_size == 12) {
    // BITMAPCOREHEADER:
    // | 32u header | 16u width | 16u height | 16u number of color planes | 16u bits per pixel
    w = ConsumeValue<uint16_t>(ptr);
    h = ConsumeValue<uint16_t>(ptr);
    ptr += 2;  // skip
    bpp = ConsumeValue<uint16_t>(ptr);
    if (bpp <= 8) {
      palette_start = ptr;
      palette_entry_size = 3;
      ncolors = (1_uz << bpp);
    }
  } else if (length >= 50 && header_size >= 40) {
    // BITMAPINFOHEADER and later:
    // | 32u header | 32s width | 32s height | 16u number of color planes | 16u bits per pixel
    // | 32u compression type
    w = ConsumeValue<int32_t>(ptr);
    h = abs(ConsumeValue<int32_t>(ptr));
    ptr += 2;  // skip
    bpp = ConsumeValue<uint16_t>(ptr);
    compression_type = ConsumeValue<uint32_t>(ptr);
    ptr += 12;  // skip
    ncolors = ConsumeValue<uint32_t>(ptr);
    ptr += header_size - 36;  // skip
    if (bpp <= 8) {
      palette_start = ptr;
      palette_entry_size = 4;
      ncolors = ncolors == 0 ? 1_uz << bpp : ncolors;
    }
    // sanity check
    if (palette_start != nullptr) {
      assert(palette_start + (ncolors * palette_entry_size) <= bmp + length);
    }

    c = number_of_channels(bpp, compression_type, palette_start, ncolors, palette_entry_size);
  }
  return {h, w, c};
}


}  // namespace dali
