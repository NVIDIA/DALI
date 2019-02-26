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


std::pair<std::shared_ptr<uint8_t>, Image::ImageDims>
JpegImage::DecodeImpl(DALIImageType type, const uint8 *jpeg, size_t length) const {
  const int c = NumberOfChannels(type);
  DALI_ENFORCE(c == 1 || c == 3 || c == 4,
    "Number of channels not supported " + std::to_string(c));
  const auto dims = PeekDims(jpeg, length);
  const auto h = std::get<0>(dims);
  const auto w = std::get<1>(dims);

  DALI_ENFORCE(jpeg != nullptr);
  DALI_ENFORCE(length > 0);
  DALI_ENFORCE(h > 0);
  DALI_ENFORCE(w > 0);

#ifdef DALI_USE_JPEG_TURBO
  // not supported by libjpeg-turbo
  if (type == DALI_YCbCr
   || type == DALI_RGBA || type == DALI_BGRA
   || type == DALI_ARGB || type == DALI_ABGR) {
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
  DALI_ENFORCE(
    jpeg::GetImageInfo(encoded_buffer, length, &width, &height, &components) == true);
  return std::make_tuple(height, width, components);
}

}  // namespace dali
