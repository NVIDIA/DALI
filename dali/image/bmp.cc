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

#include "dali/image/bmp.h"
#include "dali/core/parse_utils.h"

namespace dali {

// https://en.wikipedia.org/wiki/BMP_file_format#DIB_header_(bitmap_information_header)

namespace {
  constexpr int kHeaderSizeOffset = 14;
}  // namespace


BmpImage::BmpImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type)
  : GenericImage(encoded_buffer, length, image_type) {}

Image::Shape BmpImage::PeekShapeImpl(const uint8_t *bmp, size_t length) const {
  DALI_ENFORCE(bmp != nullptr);

  uint32_t header_size =
      ReadValueLE<uint32_t>(bmp + kHeaderSizeOffset);
  int64_t h = 0, w = 0;
  // BITMAPCOREHEADER: | 32u header | 16u width | 16u height | ...
  if (length >= 22 && header_size == 12) {
    w = ReadValueLE<uint16_t>(bmp + 18);
    h = ReadValueLE<uint16_t>(bmp + 20);
    // BITMAPINFOHEADER and later: | 32u header | 32s width | 32s height | ...
  } else if (length >= 26 && header_size >= 40) {
    w = ReadValueLE<int32_t>(bmp + 18);
    h = abs(ReadValueLE<int32_t>(bmp + 22));
  }
  std::cout << h << " x " << w << " x " << 0 << std::endl;
  return {h, w, 0};  // TODO(mszolucha): fill channels
}


}  // namespace dali
