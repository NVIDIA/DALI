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

#include <stdexcept>
#include "dali/imgcodec/parsers/bmp.h"
#include "dali/core/byte_io.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/endian_util.h"

namespace dali {
namespace imgcodec {

namespace  {

enum BmpCompressionType {
  BMP_COMPRESSION_RGB       = 0,
  BMP_COMPRESSION_RLE8      = 1,
  BMP_COMPRESSION_RLE4      = 2,
  BMP_COMPRESSION_BITFIELDS = 3
};

bool is_color_palette(InputStream *stream, int ncolors, int palette_entry_size) {
  SmallVector<uint8_t, 8> entry;
  entry.resize(palette_entry_size);
  for (int i = 0; i < ncolors; i++) {
    stream->ReadAll(entry.data(), palette_entry_size);

    const auto b = entry[0], g = entry[1], r = entry[2];  // a = p[3];
    if (b != g || b != r)
      return true;
  }
  return false;
}

int number_of_channels(InputStream *stream,
                       int bpp, int compression_type, int ncolors = 0,
                       int palette_entry_size = 0) {
  if (compression_type == BMP_COMPRESSION_RGB || compression_type == BMP_COMPRESSION_RLE8) {
    if (bpp <= 8 && ncolors <= (1_i64 << bpp)) {
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
SWAP_ENDIAN_FIELDS(BitmapCoreHeader, header_size, width, heigth, planes, bpp);

struct BitmapInfoHeader {
  int32_t header_size;
  int32_t width, heigth;
  uint16_t planes, bpp;
  uint32_t compression, image_size;
  int32_t x_pixels_per_meter, y_pixels_per_meter;
  uint32_t colors_used, colors_important;
};
static_assert(sizeof(BitmapInfoHeader) == 40);
SWAP_ENDIAN_FIELDS(BitmapInfoHeader,
  header_size,
  width, heigth,
  planes, bpp,
  compression, image_size,
  x_pixels_per_meter, y_pixels_per_meter,
  colors_used, colors_important);

ImageInfo BmpParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ssize_t length = stream->Size();

  // https://en.wikipedia.org/wiki/BMP_file_format#DIB_header_(bitmap_information_header)
  DALI_ENFORCE(length >= 18);

  static constexpr int kHeaderStart = 14;
  stream->SeekRead(kHeaderStart);
  uint32_t header_size = ReadValueLE<uint32_t>(*stream);
  stream->Skip<uint32_t>(-1);  // we'll read it again - it's part of the header struct
  int64_t h = 0, w = 0, c = 0;
  int bpp = 0, compression_type = BMP_COMPRESSION_RGB;
  int ncolors = 0;
  int palette_entry_size = 0;
  ptrdiff_t palette_start = 0;

  if (length >= 26 && header_size == 12) {
    BitmapCoreHeader header = {};
    stream->ReadAll(&header, 1);
    from_little_endian(header);
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
    stream->Skip(header_size - sizeof(header));  // Skip the ignored part of header
    from_little_endian(header);
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
  } else {
    const char *file_info = encoded->SourceInfo()
                          ? encoded->SourceInfo()
                          : "a file";

    DALI_FAIL(make_string("Unexpected length of a BMP header ", header_size,
                          " in ", file_info, " which is ", length, " bytes long."));
  }

  // sanity check
  if (palette_start != 0) {  // this silences a warning about unused variable
    assert(palette_start + (ncolors * palette_entry_size) <= length);
  }

  c = number_of_channels(stream.get(), bpp, compression_type, ncolors, palette_entry_size);

  ImageInfo info;
  info.shape = {h, w, c};
  info.orientation = {};
  return info;
}

bool BmpParser::CanParse(ImageSource *encoded) const {
  if (encoded->Kind() == InputKind::HostMemory) {
    int length = encoded->Size();
    if (length < 18)
      return false;
    const char *header = encoded->RawData<char>();
    return header[0] == 'B' && header[1] == 'M';
  } else {
    char header[2];
    auto stream = encoded->Open();
    size_t length = stream->Size();
    if (length < 18u)
      return false;
    stream->ReadAll(header, 2);
    return header[0] == 'B' && header[1] == 'M';
  }
}

}  // namespace imgcodec
}  // namespace dali
