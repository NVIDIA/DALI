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

#include <algorithm>
#include <string>
#include <vector>
#include "dali/imgcodec/parsers/webp.h"
#include "dali/core/byte_io.h"
#include "third_party/opencv/exif/exif.h"

namespace dali {
namespace imgcodec {

namespace {

// Header layout:
// RiffHeader | ChunkHeader | <WebpLossyHeader/WebpLosslessHeader>
//
// The WebpLossyHeader and WebpLosslessHeader are selected based on a identifier,
// respectively "VP8 " and "VP8L" in vp8 ChunkHeader. Value "VP8X" signifies
// WebP extended file format, which is not supported.

// Struct must be packed for reading them to work correctly
#pragma pack(push, 1)

struct RiffHeader {
  std::array<uint8_t, 4> riff_text;
  uint32_t file_size;
  std::array<uint8_t, 4> webp_text;
};
static_assert(sizeof(RiffHeader) == 12);

using chunk_identifier_t = std::array<uint8_t, 4>;

// The structures must be packed for reading them to work
#pragma pack(push, 1) 

// Chunk specification:
// https://developers.google.com/speed/webp/docs/riff_container#riff_file_format
struct ChunkHeader {
  chunk_identifier_t identifier;
  uint32_t chunk_size;
};
static_assert(sizeof(ChunkHeader) == 8);

// Simple file format (lossy)
// https://datatracker.ietf.org/doc/html/rfc6386#section-9.1
struct WebpLossyHeader {
  std::array<uint8_t, 3> frame_tag;
  std::array<uint8_t, 3> sync_code;
  uint16_t width;
  uint16_t height;
};
static_assert(sizeof(WebpLossyHeader) == 10);

// Simple file format (lossless)
// https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#2_riff_header
struct WebpLosslessHeader {
  uint8_t signature_byte;
  uint32_t features;
};
static_assert(sizeof(WebpLosslessHeader) == 5);

// Extended file format
// https://developers.google.com/speed/webp/docs/riff_container#extended_file_format
struct WebpExtendedHeader {
  uint8_t layout_mask;
  std::array<uint8_t, 3> reserved;
  std::array<uint8_t, 3> width;
  std::array<uint8_t, 3> height;
};
static_assert(sizeof(WebpExtendedHeader) == 10);

#pragma pack(pop) // end of packing scope

// Specific bits in WebpExtendedHeader::layout_mask
const uint8_t EXTENDED_LAYOUT_RESERVED = 1 << 0;
const uint8_t EXTENDED_LAYOUT_ANIMATION = 1 << 1;
const uint8_t EXTENDED_LAYOUT_XMP_METADATA = 1 << 2;
const uint8_t EXTENDED_LAYOUT_EXIF_METADATA = 1 << 3;
const uint8_t EXTENDED_LAYOUT_ALPHA = 1 << 4;
const uint8_t EXTENDED_LAYOUT_ICC_PROFILE = 1 << 5;

#pragma pack(pop)  // end of packing scope

template<size_t N>
constexpr std::array<uint8_t, N - 1> tag(const char (&c)[N]) {
  std::array<uint8_t, N - 1> a{};
  std::copy(&c[0], &c[N - 1], a.begin());
  return a;
}

bool is_valid_riff_header(const RiffHeader &header) {
  return header.riff_text == tag("RIFF") && header.webp_text == tag("WEBP");
}

bool is_simple_lossy_format(const ChunkHeader &vp8_header) {
  return vp8_header.identifier == tag("VP8 ");
}

bool is_simple_lossless_format(const ChunkHeader &vp8_header) {
  return vp8_header.identifier == tag("VP8L");
}

bool is_extended_format(const ChunkHeader &vp8_header) {
  return vp8_header.identifier == tag("VP8X");
}

bool is_sync_code_valid(const WebpLossyHeader &header) {
  return header.sync_code == std::array<uint8_t, 3>{0x9D, 0x01, 0x2A};
}

bool is_sync_code_valid(const WebpLosslessHeader &header) {
  return header.signature_byte == 0x2F;
}

template<size_t N>
std::string sequence_of_integers(const std::array<uint8_t, N> &data) {
  std::string result;
  for (size_t i = 0; i < N; i++)
    result += (i == 0 ? "" : " ") + std::to_string(data[i]);
  return result;
}

// TODO: unify fetching the EXIF tags from accross the imgcodec parsers
void fetch_info_from_exif_data(std::vector<uint8_t> &data, ImageInfo &info) {
  cv::ExifReader reader;
  reader.parseExif(data.data(), data.size());
  const auto orientation_value = reader.getTag(cv::ORIENTATION).field_u16;
  const auto exif_orientation = static_cast<ExifOrientation>(orientation_value);
  info.orientation = FromExifOrientation(exif_orientation);
}

// Iterate over the chunks seeking the EXIF chunk
// InputStream must be at the start of a chunk
void seek_exif_data(InputStream &stream, ImageInfo &info) {
  while (true) {
    const auto header = stream.ReadOne<ChunkHeader>();
    if (header.identifier == tag("EXIF")) {
      // Parse the chunk data into the orientation
      std::vector<uint8_t> buffer(header.chunk_size);
      stream.ReadBytes(buffer.data(), buffer.size());
      fetch_info_from_exif_data(buffer, info);
      break;
    } else {
      // Skip the rest of the chunk
      stream.Skip(header.chunk_size);
    }
  }
}

}  // namespace

ImageInfo WebpParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ImageInfo info;

  stream->Skip<RiffHeader>();
  const auto vp8_header = stream->ReadOne<ChunkHeader>();
  if (is_simple_lossy_format(vp8_header)) {
    const auto lossy_header = stream->ReadOne<WebpLossyHeader>();
    if (!is_sync_code_valid(lossy_header)) {
      DALI_FAIL("Sync code 157 1 42 not found at expected position. Found " +
                sequence_of_integers(lossy_header.sync_code) + " instead");
    }

    // only the last 14 bits of the fields code the dimensions
    const int w = lossy_header.width & 0x3FFF;
    const int h = lossy_header.height & 0x3FFF;

    // VP8 always uses RGB
    info.shape = {h, w, 3};
  } else if (is_simple_lossless_format(vp8_header)) {
    const auto lossless_header = stream->ReadOne<WebpLosslessHeader>();
    if (!is_sync_code_valid(lossless_header)) {
        DALI_FAIL("Sync code 47 not found at expected position. Found " +
                  std::to_string(lossless_header.signature_byte) + " instead.");
    }

    // VP8L shape information are packed inside the features field
    const int w = (lossless_header.features & 0x00003FFF) + 1;
    const int h = ((lossless_header.features & 0x0FFFC000) >> 14) + 1;
    const int alpha = (lossless_header.features & 0x10000000) >> 28;

    // VP8L always uses RGBA
    info.shape = {h, w, 3 + alpha};
  } else if (is_extended_format(vp8_header)) {
    const auto extended_header = stream->ReadOne<WebpExtendedHeader>();
    const int w = ReadValueLE<uint32_t, 3>(extended_header.width.data()) + 1;
    const int h = ReadValueLE<uint32_t, 3>(extended_header.height.data()) + 1;
    const bool alpha = extended_header.layout_mask & EXTENDED_LAYOUT_ALPHA;

    info.shape = {h, w, 3 + alpha};

    if (extended_header.layout_mask & EXTENDED_LAYOUT_EXIF_METADATA) {
      // Skip the rest of the chunk, so the stream is at the start of a next chunk
      stream->Skip(vp8_header.chunk_size - sizeof(extended_header));
      seek_exif_data(*stream, info);
    }
  } else {
    DALI_FAIL("Unrecognized WebP header: " + sequence_of_integers(vp8_header.identifier));
  }

  std::cerr << "Orientation: (" << info.orientation.flip_x 
                        << ", " << info.orientation.flip_y
                        << ", " << info.orientation.rotate << ")\n";

  return info;
}

bool WebpParser::CanParse(ImageSource *encoded) const {
  static_assert(sizeof(WebpLossyHeader) > sizeof(WebpLosslessHeader));
  uint8_t data[sizeof(RiffHeader) + sizeof(ChunkHeader) + sizeof(WebpLossyHeader)];
  if (!ReadHeader(data, encoded, sizeof(data)))
    return false;
  MemInputStream stream(data, sizeof(data));

  if (!is_valid_riff_header(stream.ReadOne<RiffHeader>()))
    return false;

  const auto vp8_header = stream.ReadOne<ChunkHeader>();
  if (is_simple_lossy_format(vp8_header)) {
    return is_sync_code_valid(stream.ReadOne<WebpLossyHeader>());
  } else if (is_simple_lossless_format(vp8_header)) {
    return is_sync_code_valid(stream.ReadOne<WebpLosslessHeader>());
  } else if (is_extended_format(vp8_header)) {
    return true;  // no sync code here
  } else {
    return false;  // other formats are not supported
  }
}

}  // namespace imgcodec
}  // namespace dali
