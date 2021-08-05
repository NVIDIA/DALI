// Copyright (c) 2021, Lennart Behme. All rights reserved.
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

#include "dali/image/webp.h"
#include "dali/core/byte_io.h"

namespace dali {

WebpImage::WebpImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
         GenericImage(encoded_buffer, length, image_type) {
}

Image::Shape WebpImage::PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const {
  DALI_ENFORCE(encoded_buffer);
  DALI_ENFORCE(length >= 16);

  // RIFF header has 12 bytes
  const uint8_t *vp8_data = encoded_buffer + 12;
  if (vp8_data[0] == 'V' && vp8_data[1] == 'P' && vp8_data[2] == '8' && vp8_data[3] == ' ') {
    // Simple file format (lossy)
    // https://datatracker.ietf.org/doc/html/rfc6386#section-9.1
    DALI_ENFORCE(length >= 30);

    // Verify the sync code
    // According to my understanding of the standard, the sync code should start at vp8_data[7]
    // but looking into the binary data, I always found it at vp8_data[11].
    if (vp8_data[11] != 0x9D || vp8_data[12] != 0x01 || vp8_data[13] != 0x2A) {
      DALI_FAIL("Sync code 157 1 42 not found at expected position. Found " +
                std::to_string(vp8_data[11]) + " " + std::to_string(vp8_data[12]) +
                " " + std::to_string(vp8_data[13]) + " instead.")
    }

    // VP8 shape information starts after the sync code
    const int64_t W = ReadValueLE<uint16_t>(vp8_data + 14) & 0x3FFF;
    const int64_t H = ReadValueLE<uint16_t>(vp8_data + 16) & 0x3FFF;

    // VP8 always uses RGB
    return {H, W, 3};
  } else if (vp8_data[0] == 'V' && vp8_data[1] == 'P' && vp8_data[2] == '8' && vp8_data[3] == 'L') {
    // Simple file format (lossless)
    // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#2_riff_header
    DALI_ENFORCE(length >= 25);

    // Verify the signature byte
    if (vp8_data[8] != 0x2F) {
      DALI_FAIL("Sync code 47 not found at expected position. Found " +
                std::to_string(vp8_data[8]) + " instead.")
    }

    // VP8L shape information starts after the sync code
    const uint32_t features = ReadValueLE<uint32_t>(vp8_data + 9);
    const int64_t W = (features & 0x00003FFF) + 1;
    const int64_t H = ((features & 0x0FFFC000) >> 14) + 1;
    const int8_t alpha = (features & 0x10000000) >> 28;

    // VP8L always uses RGBA
    return {H, W, 3 + alpha};
  } else if (vp8_data[0] == 'V' && vp8_data[1] == 'P' && vp8_data[2] == '8' && vp8_data[3] == 'X') {
    // Extended file format
    DALI_FAIL("WebP extended file format is not supported.");
  } else {
    DALI_FAIL("Unrecognized WebP header: " + std::to_string(vp8_data[0]) +
              std::to_string(vp8_data[1]) + std::to_string(vp8_data[2]) +
              std::to_string(vp8_data[3]));
  }
}

}  // namespace dali
