// Copyright (c) 2017-2018, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/image/tiff.h"

namespace dali {

namespace {

constexpr int COUNT_SIZE = 2;
constexpr int ENTRY_SIZE = 12;
constexpr int WIDTH_TAG = 256;
constexpr int HEIGHT_TAG = 257;
constexpr int PHOTOMETRIC_INTERPRETATION_TAG = 262;
constexpr int SAMPLESPERPIXEL_TAG = 277;
constexpr int TYPE_WORD = 3;
constexpr int TYPE_DWORD = 4;
constexpr int PHOTOMETRIC_PALETTE = 3;

constexpr std::array<int, 4> le_header = {77, 77, 0, 42};

bool is_little_endian(const unsigned char *tiff) {
  DALI_ENFORCE(tiff);
  for (unsigned int i = 0; i < le_header.size(); i++) {
    if (tiff[i] != le_header[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

TiffImage::TiffImage(const uint8_t *encoded_buffer, size_t length, dali::DALIImageType image_type)
    : GenericImage(encoded_buffer, length, image_type) {}

Image::Shape TiffImage::PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const {
  DALI_ENFORCE(encoded_buffer != nullptr);
  TiffBuffer buffer(
    std::string(reinterpret_cast<const char *>(encoded_buffer),
    static_cast<size_t>(length)),
    is_little_endian(encoded_buffer));

  const auto ifd_offset = buffer.Read<uint32_t>(4);
  const auto entry_count = buffer.Read<uint16_t>(ifd_offset);
  bool width_read = false, height_read = false, samples_per_px_read = false, palette_read = false;
  int64_t width = 0, height = 0, nchannels = 0;

  for (int entry_idx = 0; entry_idx < entry_count; entry_idx++) {
    const auto entry_offset = ifd_offset + COUNT_SIZE + entry_idx * ENTRY_SIZE;
    const auto tag_id = buffer.Read<uint16_t>(entry_offset);
    if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG || tag_id == SAMPLESPERPIXEL_TAG
        || tag_id == PHOTOMETRIC_INTERPRETATION_TAG) {
      const auto value_type = buffer.Read<uint16_t>(entry_offset + 2);
      const auto value_count = buffer.Read<uint32_t>(entry_offset + 4);
      DALI_ENFORCE(value_count == 1);

      int64_t value;
      if (value_type == TYPE_WORD) {
        value = buffer.Read<uint16_t>(entry_offset + 8);
      } else if (value_type == TYPE_DWORD) {
        value = buffer.Read<uint32_t>(entry_offset + 8);
      } else {
        DALI_FAIL("Couldn't read TIFF image dims.");
      }

      if (tag_id == WIDTH_TAG) {
        width = value;
        width_read = true;
      } else if (tag_id == HEIGHT_TAG) {
        height = value;
        height_read = true;
      } else if (tag_id == SAMPLESPERPIXEL_TAG && !palette_read) {
        // If the palette is present, the SAMPLESPERPIXEL tag is always set to 1, so it does not
        // indicate the actual number of channels. That's why we ignore it for palette images.
        nchannels = value;
        samples_per_px_read = true;
      } else if (tag_id == PHOTOMETRIC_INTERPRETATION_TAG && value == PHOTOMETRIC_PALETTE) {
        nchannels = 3;
        palette_read = true;
      }
    }
    if (width_read && height_read && palette_read)
      break;
  }

  DALI_ENFORCE(width_read && height_read && (samples_per_px_read || palette_read),
    "TIFF image dimensions haven't been peeked properly");

  return {height, width, nchannels};
}

}  // namespace dali
