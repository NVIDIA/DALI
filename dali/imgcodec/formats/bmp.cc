// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/imgcodec/formats/bmp.h"
#include "dali/core/byte_io.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace imgcodec {

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

bool is_color_palette(InputStream *stream, size_t ncolors, size_t palette_entry_size) {
  SmallVector<uint8_t, 8> entry;
  entry.resize(palette_entry_size);
  for (size_t i = 0; i < ncolors; i++) {
    stream->ReadAll(entry.data(), palette_entry_size);

    const auto b = entry[0], g = entry[1], r = entry[2];  // a = p[3];
    if (b != g || b != r)
      return true;
  }
  return false;
}

int number_of_channels(InputStream *stream,
                       int bpp, int compression_type, size_t ncolors = 0,
                       size_t palette_entry_size = 0) {
  if (compression_type == BMP_COMPRESSION_RGB || compression_type == BMP_COMPRESSION_RLE8) {
    if (bpp <= 8 && ncolors <= (1_uz << bpp)) {
      return is_color_palette(stream, ncolors, palette_entry_size) ? 3 : 1;
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

  throw std::runtime_error(make_string(
    "configuration not supported. bpp: ", bpp, " compression_type:", compression_type,
    "ncolors:", ncolors));
}

}  // namespace

struct BitmapCoreHeader {
  uint32_t header_size;
  uint16_t width, heigth, planes, bpp;
};
static_assert(sizeof(BitmapCoreHeader) == 12);

struct BitmapInfoHeader {
  int32_t header_size;
  int32_t width, heigth;
  uint16_t planes, bpp;
  uint32_t compression, image_size;
  int32_t x_pixels_per_meter, y_pixels_per_meter;
  uint32_t colors_used, colors_important;
};
static_assert(sizeof(BitmapinfoHeader) == 40);

ImageInfo BmpParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ssize_t length = stream->Size();

  DALI_ENFORCE(length >= 18);

  static constexpr int kHeaderStart = 14;
  stream->SeekRead(kHeaderStart);
  uint32_t header_size = stream->ReadOne<uint32_t>();
  stream->Skip<uint32_t>(-1);  // we'll read it again - it's part of the header struct
  int64_t h = 0, w = 0, c = 0;
  int bpp = 0, compression_type = BMP_COMPRESSION_RGB;
  size_t ncolors = 0;
  size_t palette_entry_size = 0;
  ptrdiff_t palette_start = 0;

  if (length >= 26 && header_size == 12) {
    BitmapCoreHeader header = {};
    stream->ReadAll(&header, 1);
    w = header.width;
    h = header.heigth;
    bpp = header.bpp;
    if (bpp <= 8) {
      palette_start = stream->TellRead();
      palette_entry_size = 3;
      ncolors = (1_uz << bpp);
    }
  } else if (length >= 50 && header_size >= 40) {
    BitmapInfoHeader header = {};
    stream->ReadAll(&header, 1);
    w = abs(header.width);
    h = abs(header.heigth);
    bpp = header.bpp;
    compression_type = header.compression;
    ncolors = header.colors_used;
    if (bpp <= 8) {
      palette_start = stream->TellRead();
      palette_entry_size = 4;
      ncolors = ncolors == 0 ? 1_uz << bpp : ncolors;
    }
  }

  // sanity check
  if (palette_start != 0) {
    assert(palette_start + (ncolors * palette_entry_size) <= length);
  }

  c = number_of_channels(stream, bpp, compression_type, ncolors, palette_entry_size);

  ImageInfo info;
  info.shape = {h, w, c};
  return info;
}

bool BmpParser::CanParse(ImageSource *encoded) const {
  int length = encoded->Size();
  if (length < 18)
    return 0;
  if (encoded->Kind() == InputKind::HostMemory) {
    char *header = encoded->RawData<char>();
    return header[0] == 'B' && header[1] == 'M';
  } else {
    char header[2];
    encoded->Open()->ReadAll(header, 2);
    return header[0] == 'B' && header[1] == 'M';
  }
}

}  // namespace imgcodec
}  // namespace dali
