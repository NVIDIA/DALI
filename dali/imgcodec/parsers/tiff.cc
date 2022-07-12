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

#include "dali/imgcodec/parsers/tiff.h"
#include "dali/core/byte_io.h"
#include "dali/imgcodec/image_orientation.h"

namespace dali {
namespace imgcodec {

constexpr int ENTRY_SIZE = 12;

enum TiffTag : uint16_t {
  WIDTH_TAG = 256,
  HEIGHT_TAG = 257,
  ORIENTATION_TAG = 274,
  SAMPLESPERPIXEL_TAG = 277
};

enum TiffDataType : uint16_t {
  TYPE_WORD = 3,
  TYPE_DWORD = 4
};

using tiff_magic_t = std::array<uint8_t, 4>;
constexpr tiff_magic_t le_header = {'I', 'I', 42, 0}, be_header = {'M', 'M', 0, 42};

template<typename T, bool is_little_endian>
T TiffRead(InputStream& stream) {
  if constexpr (is_little_endian) {
    return ReadValueLE<T>(stream);
  } else {
    return ReadValueBE<T>(stream);
  }
}

template <bool is_little_endian>
ImageInfo GetInfoImpl(ImageSource *encoded) {
  ImageInfo info;
  info.orientation = {0, false, false};

  auto stream = encoded->Open();
  stream->SeekRead(4, SEEK_SET);
  const auto ifd_offset = TiffRead<uint32_t, is_little_endian>(*stream);
  stream->SeekRead(ifd_offset, SEEK_SET);
  const auto entry_count = TiffRead<uint16_t, is_little_endian>(*stream);

  bool width_read = false, height_read = false, nchannels_read = false;
  int64_t width = 0, height = 0, nchannels = 0;

  for (int entry_idx = 0;
       entry_idx < entry_count && !(width_read && height_read && nchannels_read);
       entry_idx++) {
    const auto entry_offset = ifd_offset + sizeof(uint16_t) + entry_idx * ENTRY_SIZE;
    stream->SeekRead(entry_offset, SEEK_SET);
    const auto tag_id = TiffRead<uint16_t, is_little_endian>(*stream);
    if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG || tag_id == SAMPLESPERPIXEL_TAG
        || tag_id == ORIENTATION_TAG) {
      const auto value_type = TiffRead<uint16_t, is_little_endian>(*stream);
      const auto value_count = TiffRead<uint32_t, is_little_endian>(*stream);
      DALI_ENFORCE(value_count == 1);

      int64_t value;
      if (value_type == TYPE_WORD) {
        value = TiffRead<uint16_t, is_little_endian>(*stream);
      } else if (value_type == TYPE_DWORD) {
        value = TiffRead<uint32_t, is_little_endian>(*stream);
      } else {
        DALI_FAIL("Couldn't read TIFF image dims.");
      }

      if (tag_id == WIDTH_TAG) {
        width = value;
        width_read = true;
      } else if (tag_id == HEIGHT_TAG) {
        height = value;
        height_read = true;
      } else if (tag_id == SAMPLESPERPIXEL_TAG) {
        nchannels = value;
        nchannels_read = true;
      } else if (tag_id == ORIENTATION_TAG) {
        info.orientation = FromExifOrientation(static_cast<ExifOrientation>(value));
      }
    }
  }

  DALI_ENFORCE(width_read && height_read && nchannels_read,
    "TIFF image dims haven't been read properly");

  info.shape = {height, width, nchannels};
  return info;
}

ImageInfo TiffParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  DALI_ENFORCE(stream->Size() >= 8);

  tiff_magic_t header = stream->ReadOne<tiff_magic_t>();
  if (header == le_header) {
    return GetInfoImpl<true>(encoded);
  } else {
    return GetInfoImpl<false>(encoded);
  }
}

bool TiffParser::CanParse(ImageSource *encoded) const {
  tiff_magic_t header;
  ReadHeader(header.data(), encoded, sizeof(header));
  return (header == le_header || header == be_header);
}

}  // namespace imgcodec
}  // namespace dali
