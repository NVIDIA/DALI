// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/byte_io.h"

namespace dali {

JpegImage::JpegImage(const uint8_t *encoded_buffer,
                     size_t length,
                     DALIImageType image_type)
  : GenericImage(encoded_buffer, length, image_type) {
}

#ifndef DALI_USE_JPEG_TURBO
bool get_jpeg_size(const uint8 *data, size_t data_size, int *height, int *width, int *nchannels) {
  unsigned int i = 0;
  if (!(data[i] == 0xFF && data[i + 1] == 0xD8))
    return false;  // Not a valid SOI header

  i += 4;
  // Retrieve the block length of the first block since
  // the first block will not contain the size of file
  uint16_t block_length = ReadValueBE<uint16_t>(data + i);
  while (i < data_size) {
    i += block_length;  // Increase the file index to get to the next block
    if (i >= data_size) return false;  // Check to protect against segmentation faults
    if (data[i] != 0xFF) return false;  // Check that we are truly at the start of another block
    if (data[i + 1] >= 0xC0 && data[i + 1] <= 0xC3) {
      // 0xFFC0 is the "Start of frame" marker which contains the file size
      // The structure of the 0xFFC0 block is quite simple
      // [0xFFC0][ushort length][uchar precision][ushort x][ushort y][uchar number_of_components]
      *height = ReadValueBE<uint16_t>(data + i + 5);
      *width  = ReadValueBE<uint16_t>(data + i + 7);
      *nchannels = ReadValueBE<uint8_t>(data + i + 9);
      return true;
    } else {
      i += 2;  // Skip the block marker
      block_length = ReadValueBE<uint16_t>(data + i);  // Go to the next block
    }
  }
  return false;  // If this point is reached then no size was found
}
#endif

std::pair<std::shared_ptr<uint8_t>, Image::Shape>
JpegImage::DecodeImpl(DALIImageType type, const uint8 *jpeg, size_t length) const {
  const auto shape = PeekShapeImpl(jpeg, length);
  const auto h = shape[0];
  const auto w = shape[1];
  assert(shape[2] <= 3);  // peek shape should clamp to 3 channels
  if (type == DALI_ANY_DATA) {
    type = shape[2] == 3 ? DALI_RGB : DALI_GRAY;
  }
  const auto c = NumberOfChannels(type);

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
  if (UseFastIdct()) {
    flags.dct_method = JDCT_FASTEST;
  }
  flags.components = c;

  flags.crop = false;
  auto crop_window_generator = GetCropWindowGenerator();
  if (crop_window_generator) {
    flags.crop = true;
    TensorShape<> shape{static_cast<int>(h), static_cast<int>(w)};
    auto crop = crop_window_generator(shape, "HW");
    crop.EnforceInRange(shape);
    flags.crop_y = crop.anchor[0];
    flags.crop_x = crop.anchor[1];
    flags.crop_height = crop.shape[0];
    flags.crop_width = crop.shape[1];
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

  return {decoded_image, {cropped_h, cropped_w, c}};
#else  // DALI_USE_JPEG_TURBO
  return GenericImage::DecodeImpl(type, jpeg, length);
#endif  // DALI_USE_JPEG_TURBO
}

Image::Shape JpegImage::PeekShapeImpl(const uint8_t *encoded_buffer,
                                      size_t length) const {
  int height = 0, width = 0, components = 0;
#ifdef DALI_USE_JPEG_TURBO
  DALI_ENFORCE(
    jpeg::GetImageInfo(encoded_buffer, length, &width, &height, &components) == true);
#else
  DALI_ENFORCE(get_jpeg_size(encoded_buffer, length, &height, &width, &components));
#endif

  if (components > 3)  // We support only 1 or 3 channels
    components = 3;
  return {height, width, components};
}

}  // namespace dali
