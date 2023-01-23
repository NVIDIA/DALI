// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "third_party/opencv/exif/exif.h"
#include "dali/core/error_handling.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/core/byte_io.h"

namespace dali {
namespace imgcodec {

using jpeg_marker_t = std::array<uint8_t, 2>;
using jpeg_exif_header_t = std::array<uint8_t, 6>;

constexpr jpeg_marker_t sos_marker = {0xff, 0xda};
constexpr jpeg_marker_t soi_marker = {0xff, 0xd8};
constexpr jpeg_marker_t eoi_marker = {0xff, 0xd9};
constexpr jpeg_marker_t app1_marker = {0xff, 0xe1};

constexpr jpeg_exif_header_t exif_header = {'E', 'x', 'i', 'f', 0, 0};

bool IsValidMarker(const jpeg_marker_t &marker) {
  return marker[0] == 0xff && marker[1] != 0x00;
}

bool IsSofMarker(const jpeg_marker_t &marker) {
  // According to https://www.w3.org/Graphics/JPEG/itu-t81.pdf table B.1 Marker code assignments
  // SOF markers are from range 0xFFC0-0xFFCF, excluding 0xFFC4, 0xFFC8 and 0xFFCC.
  if (!IsValidMarker(marker) || marker[1] < 0xc0 || marker[1] > 0xcf) return false;
  return marker[1] != 0xc4 && marker[1] != 0xc8 && marker[1] != 0xcc;
}

ImageInfo JpegParser::GetInfo(ImageSource *encoded) const {
  return GetExtendedInfo(encoded).img_info;
}

JpegParser::ExtendedImageInfo JpegParser::GetExtendedInfo(ImageSource *encoded) const {
  JpegParser::ExtendedImageInfo info{};
  auto stream = encoded->Open();

  jpeg_marker_t first_marker = stream->ReadOne<jpeg_marker_t>();
  DALI_ENFORCE(first_marker == soi_marker);

  bool read_shape = false, read_orientation = false;
  while (!read_shape || !read_orientation) {
    jpeg_marker_t marker;
    marker[0] = stream->ReadOne<uint8_t>();
    // https://www.w3.org/Graphics/JPEG/itu-t81.pdf section B.1.1.2 Markers
    // Any marker may optionally be preceded by any number of fill bytes,
    // which are bytes assigned code '\xFF'
    do {
      marker[1] = stream->ReadOne<uint8_t>();
    } while (marker[1] == 0xff);
    DALI_ENFORCE(IsValidMarker(marker),
                 make_string("Invalid marker found in JPEG image: ", encoded->SourceInfo()));
    if (marker == sos_marker)
      break;

    uint16_t size = ReadValueBE<uint16_t>(*stream);
    ptrdiff_t next_marker_offset = stream->TellRead() - 2 + size;
    if (IsSofMarker(marker)) {
      info.sof_marker = marker;
      stream->Skip(1);  // Skip the precision field
      auto height = ReadValueBE<uint16_t>(*stream);
      auto width = ReadValueBE<uint16_t>(*stream);
      auto nchannels = stream->ReadOne<uint8_t>();
      info.img_info.shape = {height, width, nchannels};
      read_shape = true;
    } else if (marker == app1_marker && stream->ReadOne<jpeg_exif_header_t>() == exif_header) {
      std::vector<uint8_t> exif_block(size - 8);
      stream->Read(exif_block.data(), size - 8);
      cv::ExifReader reader;
      if (!reader.parseExif(exif_block.data(), exif_block.size()))
        continue;
      auto entry = reader.getTag(cv::ORIENTATION);
      if (entry.tag != cv::INVALID_TAG) {
        info.img_info.orientation =
            FromExifOrientation(static_cast<ExifOrientation>(entry.field_u16));
        read_orientation = true;
      }
    }
    stream->SeekRead(next_marker_offset, SEEK_SET);
  }
  if (!read_shape)
    DALI_FAIL(make_string("Couldn't read the dimensions of JPEG image: ", encoded->SourceInfo()));
  return info;
}

bool JpegParser::CanParse(ImageSource *encoded) const {
  jpeg_marker_t first_marker;
  return (ReadHeader(first_marker.data(), encoded, first_marker.size()) == first_marker.size() &&
          first_marker == soi_marker);
}

}  // namespace imgcodec
}  // namespace dali
