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

#include <string>
#include "dali/imgcodec/parsers/webp.h"
#include "dali/core/byte_io.h"

namespace dali {
namespace imgcodec {

namespace {
  bool is_pattern_matching(const uint8_t *data, const std::string &pattern) {
    for (size_t i = 0; i < pattern.size(); i++)
      if (data[i] != pattern[i])
        return false;
    return true;
  }

  bool is_valid_sync_code_lossy(const uint8_t *data) {
    return data[0] == 0x9D && data[1] == 0x01 && data[2] == 0x2A;
  }

  const uint8_t SIGNATURE_BYTE_LOSTLESS = 0x2F;
}  // namespace

ImageInfo WebpParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ssize_t length = stream->Size();

  // Skipping 12 bytes of the RIFF header
  // "RIFF" (4 bytes)
  // File Size (4 bytes)
  // "WEBP" (4 bytes)
  stream->SeekRead(12, SEEK_CUR);

  ImageInfo info;
  const auto header = stream->ReadOne<std::array<uint8_t, 4>>();
  if (is_pattern_matching(header.data(), "VP8 ")) {
    // Simple file format (lossy)
    // https://datatracker.ietf.org/doc/html/rfc6386#section-9.1
    // Skipping ...? (4 bytes) and frame tag (3 bytes)
    stream->SeekRead(7, SEEK_CUR);

    // Verify sync code
    const auto sync_code = stream->ReadOne<std::array<uint8_t, 3>>();
    if (!is_valid_sync_code_lossy(sync_code.data())) {
      DALI_FAIL("Sync code 157 1 42 not found at expected position. Found " +
                std::to_string(sync_code[0]) + " " +
                std::to_string(sync_code[1]) + " " +
                std::to_string(sync_code[2]) + " instead.");
    }

    const int w = ReadValueLE<uint16_t>(*stream) & 0x3FFF;
    const int h = ReadValueLE<uint16_t>(*stream) & 0x3FFF;
    info.shape = {h, w, 3};
  } else if (is_pattern_matching(header.data(), "VP8L")) {
    // Simple file format (lossless)
    // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#2_riff_header
    // Skipping number of bytes in the lossless tream (4 bytes)
    stream->SeekRead(4, SEEK_CUR);

    // Verify the signature byte
    const auto sync_code = stream->ReadOne<uint8_t>();
    if (sync_code != SIGNATURE_BYTE_LOSTLESS) {
        DALI_FAIL("Sync code 47 not found at expected position. Found " +
                  std::to_string(sync_code) + " instead.");
    }

    // VP8L shape information starts after the sync code
    const auto features = ReadValueLE<uint32_t>(*stream);
    const int w = (features & 0x00003FFF) + 1;
    const int h = ((features & 0x0FFFC000) >> 14) + 1;
    const int alpha = (features & 0x10000000) >> 28;

    // VP8L always uses RGBA
    info.shape = {h, w, 3 + alpha};
  } else if (is_pattern_matching(header.data(), "VP8X")) {
    // Extended file format
    DALI_FAIL("WebP extended file format is not supported.");
  } else {
    DALI_FAIL("Unrecognized WebP header: " +
              std::to_string(header[0]) + " " +
              std::to_string(header[1]) + " " +
              std::to_string(header[2]) + " " +
              std::to_string(header[3]));
  }

  return info;
}

bool WebpParser::CanParse(ImageSource *encoded) const {
  // Sync code in lossy version ends at byte 25
  uint8_t header[26];
  if (!ReadHeader(header, encoded, sizeof(header)))
    return false;

  // Check the RIFF header
  if (!is_pattern_matching(header, "RIFF"))
    return false;
  // 4 bytes of file size skipped
  if (!is_pattern_matching(header + 8, "WEBP"))
    return false;

  if (is_pattern_matching(header + 12, "VP8 ")) {
    // Simple file format (lossy)
    return is_valid_sync_code_lossy(header + 23);
  } else if (is_pattern_matching(header + 12, "VP8L")) {
    // Simple file format (lossless)
    return header[20] == SIGNATURE_BYTE_LOSTLESS;
  } else if (is_pattern_matching(header + 12, "VP8X")) {
    // Extended file format -- not supported
    return false;
  } else {
    return false;
  }
}

}  // namespace imgcodec
}  // namespace dali
