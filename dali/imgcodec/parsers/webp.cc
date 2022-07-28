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
#include "dali/core/endian_util.h"
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

// Chunk specification:
// https://developers.google.com/speed/webp/docs/riff_container#riff_file_format
struct ChunkHeader {
  std::array<uint8_t, 4> identifier;
  uint32_t chunk_size;
};
static_assert(sizeof(ChunkHeader) == 8);

// Simple file format (lossy)
// https://datatracker.ietf.org/doc/html/rfc6386#section-9.1
struct WebpLossyHeader {
  std::array<uint8_t, 3> frame_tag;
  std::array<uint8_t, 3> sync_code;
  uint16_t width_field;
  uint16_t height_field;
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
  std::array<uint8_t, 3> width_field;
  std::array<uint8_t, 3> height_field;
};
static_assert(sizeof(WebpExtendedHeader) == 10);

#pragma pack(pop)  // end of packing scope

// Specify fields that are coded in LE to ensure portability
SWAP_ENDIAN_FIELDS(RiffHeader, file_size);
SWAP_ENDIAN_FIELDS(ChunkHeader, chunk_size);
SWAP_ENDIAN_FIELDS(WebpLossyHeader, width_field, height_field);
SWAP_ENDIAN_FIELDS(WebpLosslessHeader, features);

// Specific bits in WebpExtendedHeader::layout_mask
const uint8_t EXTENDED_LAYOUT_RESERVED = 1 << 0;
const uint8_t EXTENDED_LAYOUT_ANIMATION = 1 << 1;
const uint8_t EXTENDED_LAYOUT_XMP_METADATA = 1 << 2;
const uint8_t EXTENDED_LAYOUT_EXIF_METADATA = 1 << 3;
const uint8_t EXTENDED_LAYOUT_ALPHA = 1 << 4;
const uint8_t EXTENDED_LAYOUT_ICC_PROFILE = 1 << 5;

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

// If chunk_size is odd, there is a one byte padding that must be skipped
void skip_rest_of_chunk(InputStream &stream,
                        const ChunkHeader &header,
                        size_t bytes_of_data_read = 0) {
  stream.Skip(header.chunk_size + header.chunk_size % 2 - bytes_of_data_read);
}

void fetch_info_from_exif_data(std::vector<uint8_t> &data, ImageInfo &info) {
  cv::ExifReader reader;
  reader.parseExif(data.data(), data.size());
  const auto entry = reader.getTag(cv::ORIENTATION);
  if (entry.tag != cv::INVALID_TAG) {
    const auto exif_orientation = static_cast<ExifOrientation>(entry.field_u16);
    info.orientation = FromExifOrientation(exif_orientation);
  }
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
      skip_rest_of_chunk(stream, header);
    }
  }
}

}  // namespace

ImageInfo WebpParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ImageInfo info;

  stream->Skip<RiffHeader>();
  auto vp8_header = stream->ReadOne<ChunkHeader>();
  to_little_endian(vp8_header);

  if (is_simple_lossy_format(vp8_header)) {
    auto lossy_header = stream->ReadOne<WebpLossyHeader>();
    to_little_endian(lossy_header);

    if (!is_sync_code_valid(lossy_header)) {
      DALI_FAIL("Sync code 157 1 42 not found at expected position. Found " +
                sequence_of_integers(lossy_header.sync_code) + " instead");
    }

    // only the last 14 bits of the fields code the dimensions
    const int mask = (1 << 14) - 1;
    const int w = lossy_header.width_field & mask;
    const int h = lossy_header.height_field & mask;

    // VP8 always uses RGB
    info.shape = {h, w, 3};
  } else if (is_simple_lossless_format(vp8_header)) {
    auto lossless_header = stream->ReadOne<WebpLosslessHeader>();
    to_little_endian(lossless_header);

    if (!is_sync_code_valid(lossless_header)) {
        DALI_FAIL("Sync code 47 not found at expected position. Found " +
                  std::to_string(lossless_header.signature_byte) + " instead.");
    }

    // VP8L shape information are packed inside the features field
    const int bit_length = 14;
    const int mask = (1 << bit_length) - 1;
    const int w = (lossless_header.features & mask) + 1;
    const int h = ((lossless_header.features >> bit_length) & mask) + 1;
    const bool alpha = lossless_header.features & (1 << (2 * bit_length));

    // VP8L always uses RGBA
    info.shape = {h, w, 3 + alpha};
  } else if (is_extended_format(vp8_header)) {
    const auto extended_header = stream->ReadOne<WebpExtendedHeader>();

    // Both dimensions are encoded with 24 bits, as (width - 1) i (height - 1) respectively
    const int w = ReadValueLE<uint32_t, 3>(extended_header.width_field.data()) + 1;
    const int h = ReadValueLE<uint32_t, 3>(extended_header.height_field.data()) + 1;
    const bool alpha = extended_header.layout_mask & EXTENDED_LAYOUT_ALPHA;

    // VP8X is encoded with VP8 and VP8L bitstreams, so it uses RGBA
    info.shape = {h, w, 3 + alpha};

    if (extended_header.layout_mask & EXTENDED_LAYOUT_EXIF_METADATA) {
      skip_rest_of_chunk(*stream, vp8_header, sizeof(extended_header));
      seek_exif_data(*stream, info);
    }
  } else {
    DALI_FAIL("Unrecognized WebP header: " + sequence_of_integers(vp8_header.identifier));
  }

  return info;
}

bool WebpParser::CanParse(ImageSource *encoded) const {
  static_assert(sizeof(WebpLossyHeader) > sizeof(WebpLosslessHeader));
  uint8_t data[sizeof(RiffHeader) + sizeof(ChunkHeader) + sizeof(WebpLossyHeader)];
  if (!ReadHeader(data, encoded, sizeof(data)))
    return false;
  MemInputStream stream(data, sizeof(data));

  auto riff_header = stream.ReadOne<RiffHeader>();
  to_little_endian(riff_header);
  if (!is_valid_riff_header(riff_header))
    return false;

  auto vp8_header = stream.ReadOne<ChunkHeader>();
  to_little_endian(vp8_header);
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
