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
#include "dali/imgcodec/parsers/webp.h"
#include "dali/core/byte_io.h"

namespace dali {
namespace imgcodec {

namespace {

// Chunk specification:
// https://developers.google.com/speed/webp/docs/riff_container#riff_file_format
//
// Header layout:
// RiffHeader | vp8_header_t | <WebpLossyHeader/WebpLosslessHeader>
//
// The WebpLossyHeader and WebpLosslessHeader are selected based on a identifier,
// respectively "VP8 " and "VP8L" in vp8_header_t. "VP8X" in vp8_header_t signifies
// WebP extended file format, which is not supported.

// Struct must be packed for reading them to work correctly
#pragma pack(push, 1)

struct RiffHeader {
  std::array<uint8_t, 4> riff_text;
  uint32_t file_size;
  std::array<uint8_t, 4> webp_text;
};
static_assert(sizeof(RiffHeader) == 12);

using vp8_header_t = std::array<uint8_t, 4>;

// Simple file format (lossy)
// https://datatracker.ietf.org/doc/html/rfc6386#section-9.1
struct WebpLossyHeader {
  uint32_t chunk_size;
  std::array<uint8_t, 3> frame_tag;
  std::array<uint8_t, 3> sync_code;
};
static_assert(sizeof(WebpLossyHeader) == 10);

// Simple file format (lossless)
// https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#2_riff_header
struct WebpLosslessHeader {
  uint32_t chunk_size;
  uint8_t signature_byte;
};
static_assert(sizeof(WebpLosslessHeader) == 5);

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

bool is_simple_lossy_format(const vp8_header_t &header) {
  return header == tag("VP8 ");
}

bool is_simple_lossless_format(const vp8_header_t &header) {
  return header == tag("VP8L");
}

bool is_extended_format(const vp8_header_t &header) {
  return header == tag("VP8X");
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

}  // namespace

ImageInfo WebpParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ImageInfo info;

  stream->Skip<RiffHeader>();
  const auto vp8_header = stream->ReadOne<vp8_header_t>();
  if (is_simple_lossy_format(vp8_header)) {
    const auto lossy_header = stream->ReadOne<WebpLossyHeader>();
    if (!is_sync_code_valid(lossy_header)) {
      DALI_FAIL("Sync code 157 1 42 not found at expected position. Found " +
                sequence_of_integers(lossy_header.sync_code) + " instead");
    }

    const int w = ReadValueLE<uint16_t>(*stream) & 0x3FFF;
    const int h = ReadValueLE<uint16_t>(*stream) & 0x3FFF;

    // VP8 always uses RGB
    info.shape = {h, w, 3};
  } else if (is_simple_lossless_format(vp8_header)) {
    const auto lossless_header = stream->ReadOne<WebpLosslessHeader>();
    if (!is_sync_code_valid(lossless_header)) {
        DALI_FAIL("Sync code 47 not found at expected position. Found " +
                  std::to_string(lossless_header.signature_byte) + " instead.");
    }

    // VP8L shape information starts after the sync code
    const auto features = ReadValueLE<uint32_t>(*stream);
    const int w = (features & 0x00003FFF) + 1;
    const int h = ((features & 0x0FFFC000) >> 14) + 1;
    const int alpha = (features & 0x10000000) >> 28;

    // VP8L always uses RGBA
    info.shape = {h, w, 3 + alpha};
  } else if (is_extended_format(vp8_header)) {
    DALI_FAIL("WebP extended file format is not supported.");
  } else {
    DALI_FAIL("Unrecognized WebP header: " + sequence_of_integers(vp8_header));
  }

  return info;
}

bool WebpParser::CanParse(ImageSource *encoded) const {
  static_assert(sizeof(WebpLossyHeader) > sizeof(WebpLosslessHeader));
  uint8_t data[sizeof(RiffHeader) + sizeof(vp8_header_t) + sizeof(WebpLossyHeader)];
  if (!ReadHeader(data, encoded, sizeof(data)))
    return false;
  MemInputStream stream(data, sizeof(data));

  if (!is_valid_riff_header(stream.ReadOne<RiffHeader>()))
    return false;

  const auto vp8_header = stream.ReadOne<vp8_header_t>();
  if (is_simple_lossy_format(vp8_header)) {
    return is_sync_code_valid(stream.ReadOne<WebpLossyHeader>());
  } else if (is_simple_lossless_format(vp8_header)) {
    return is_sync_code_valid(stream.ReadOne<WebpLosslessHeader>());
  } else {
    return false;  // other formats are not supported
  }
}

}  // namespace imgcodec
}  // namespace dali
