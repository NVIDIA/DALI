// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/image/jpeg.h"
#include <cmath>
#include <memory>
#include "dali/image/jpeg_mem.h"
#include "dali/util/ocv.h"

namespace dali {

JpegImage::JpegImage(const uint8_t *encoded_buffer,
                     size_t length,
                     DALIImageType image_type)
  : GenericImage(encoded_buffer, length, image_type) {
}
#if !defined(DALI_USE_JPEG_TURBO)
bool get_jpeg_size(const uint8 *data, size_t data_size, int *height, int *width) {
  // Check for valid JPEG image
  unsigned int i = 0;  // Keeps track of the position within the file
  if (data[i] == 0xFF && data[i + 1] == 0xD8) {
    i += 4;
    // Retrieve the block length of the first block since
    // the first block will not contain the size of file
    uint16_t block_length = data[i] * 256 + data[i + 1];
    while (i < data_size) {
      i += block_length;  // Increase the file index to get to the next block
      if (i >= data_size) return false;  // Check to protect against segmentation faults
      if (data[i] != 0xFF) return false;  // Check that we are truly at the start of another block
      if (data[i + 1] >= 0xC0 && data[i + 1] <= 0xC3) {
        // 0xFFC0 is the "Start of frame" marker which contains the file size
        // The structure of the 0xFFC0 block is quite simple
        // [0xFFC0][ushort length][uchar precision][ushort x][ushort y]
        *height = data[i + 5] * 256 + data[i + 6];
        *width = data[i + 7] * 256 + data[i + 8];
        return true;
      } else {
        i += 2;  // Skip the block marker
        block_length = data[i] * 256 + data[i + 1];  // Go to the next block
      }
    }
    return false;  // If this point is reached then no size was found
  } else {
    return false;  // Not a valid SOI header
  }
}
#endif

std::pair<std::shared_ptr<uint8_t>, Image::ImageDims>
JpegImage::DecodeImpl(DALIImageType type, const uint8 *jpeg, size_t length) const {
  const int c = IsColor(type) ? 3 : 1;
  const auto dims = PeekDims(jpeg, length);
  const auto h = std::get<0>(dims);
  const auto w = std::get<1>(dims);

  DALI_ENFORCE(jpeg != nullptr);
  DALI_ENFORCE(length > 0);
  DALI_ENFORCE(h > 0);
  DALI_ENFORCE(w > 0);

#ifdef DALI_USE_JPEG_TURBO
  // not supported by libjpeg-turbo
  if (type == DALI_YCbCr) {
    return GenericImage::DecodeImpl(type, jpeg, length);
  }

  jpeg::UncompressFlags flags;
  flags.components = c;

  flags.crop = false;
  auto crop_window_generator = GetCropWindowGenerator();
  if (crop_window_generator) {
    flags.crop = true;
    auto crop = crop_window_generator(h, w);
    DALI_ENFORCE(crop.IsInRange(h, w));
    flags.crop_x = crop.x;
    flags.crop_y = crop.y;
    flags.crop_width = crop.w;
    flags.crop_height = crop.h;
  }

  DALI_ENFORCE(type == DALI_RGB || type == DALI_BGR || type == DALI_GRAY,
               "Color space not supported by libjpeg-turbo");
  flags.color_space = type;

  std::shared_ptr<uint8_t> decoded_image;
  int cropped_h = 0;
  int cropped_w = 0;
  uint8_t* result = jpeg::Uncompress(
    jpeg, length, flags, nullptr /* nwarn */,
    [&decoded_image, &cropped_h, &cropped_w](int width, int height, int channels) -> uint8* {
      decoded_image.reset(
        new uint8_t[height * width * channels],
        [](uint8_t* data){ delete [] data; } );
      cropped_h = height;
      cropped_w = width;
      return decoded_image.get();
    });

  if (result == nullptr) {
    // Failed to decode, fallback
    return GenericImage::DecodeImpl(type, jpeg, length);
  }

  return std::make_pair(decoded_image, std::make_tuple(cropped_h, cropped_w, c));
#else  // DALI_USE_JPEG_TURBO
  return GenericImage::DecodeImpl(type, jpeg, length);
#endif  // DALI_USE_JPEG_TURBO
}

Image::ImageDims JpegImage::PeekDims(const uint8_t *encoded_buffer,
                                     size_t length) const {
  int height = 0, width = 0, components = 0;
#ifdef DALI_USE_JPEG_TURBO
  DALI_ENFORCE(
    jpeg::GetImageInfo(encoded_buffer, length, &width, &height, &components) == true);
#else
  DALI_ENFORCE(get_jpeg_size(encoded_buffer, length, &height, &width));
#endif
  return std::make_tuple(height, width, components);
}

}  // namespace dali
