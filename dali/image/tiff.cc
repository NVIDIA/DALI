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

#include "dali/image/tiff.h"

namespace dali {

namespace legacy_impl {

constexpr int COUNT_SIZE = 2;
constexpr int ENTRY_SIZE = 12;
constexpr int WIDTH_TAG = 256;
constexpr int HEIGHT_TAG = 257;
constexpr int TYPE_WORD = 3;
constexpr int TYPE_DWORD = 4;

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

kernels::TensorShape<3> PeekDimsImpl(const uint8_t *encoded_buffer, size_t length) {
  TiffBuffer buffer(
    std::string(reinterpret_cast<const char *>(encoded_buffer),
    static_cast<size_t>(length)),
    is_little_endian(encoded_buffer));

  const auto ifd_offset = buffer.Read<uint32_t>(4);
  const auto entry_count = buffer.Read<uint16_t>(ifd_offset);
  bool width_read = false, height_read = false;
  size_t width = 0, height = 0;

  for (int entry_idx = 0;
       entry_idx < entry_count && !(width_read && height_read);
       entry_idx++) {
    const auto entry_offset = ifd_offset + COUNT_SIZE + entry_idx * ENTRY_SIZE;
    const auto tag_id = buffer.Read<uint16_t>(entry_offset);
    if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG) {
      const auto value_type = buffer.Read<uint16_t>(entry_offset + 2);
      const auto value_count = buffer.Read<uint32_t>(entry_offset + 4);
      DALI_ENFORCE(value_count == 1);

      size_t value;
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
      } else {
        height = value;
        height_read = true;
      }
    }
  }
  if (!(width_read && height_read)) {
    DALI_FAIL("TIFF image dims haven't been peeked properly");
  }

  // TODO(mszolucha): fill channels count
  return {static_cast<int64_t>(height),
          static_cast<int64_t>(width),
          0};
}

}  // namespace

TiffImage::TiffImage(const uint8_t *encoded_buffer, size_t length, dali::DALIImageType image_type) :
        GenericImage(encoded_buffer, length, image_type) {
#ifdef DALI_USE_LIBTIFF
  libtiff_decoder_.reset(new LibtiffImpl(make_span(encoded_buffer, length)));
#endif
}

Image::ImageDims TiffImage::PeekDims(const uint8_t *encoded_buffer, size_t length) const {
  DALI_ENFORCE(encoded_buffer != nullptr);

  kernels::TensorShape<3> shape;
#ifdef DALI_USE_LIBTIFF
  shape = libtiff_decoder_->Dims();
#else
  shape = legacy_impl::PeekDimsImpl(encoded_buffer, length);
#endif

  return std::make_tuple(
    static_cast<size_t>(shape[0]),
    static_cast<size_t>(shape[1]),
    static_cast<size_t>(shape[2]));
}

std::pair<std::shared_ptr<uint8_t>, Image::ImageDims>
TiffImage::DecodeImpl(DALIImageType type, const uint8 *encoded_buffer, size_t length) const {
#ifdef DALI_USE_LIBTIFF
  if (!libtiff_decoder_->CanDecode()) {
    DALI_WARN("Falling back to GenericImage");
    return GenericImage::DecodeImpl(type, encoded_buffer, length);
  }
  auto roi_generator = GetCropWindowGenerator();
  std::shared_ptr<uint8_t> decoded_data;
  kernels::TensorShape<3> decoded_shape;
  std::tie(decoded_data, decoded_shape) = libtiff_decoder_->Decode(roi_generator);
  return {
    decoded_data,
    std::make_tuple(static_cast<size_t>(decoded_shape[0]),
                    static_cast<size_t>(decoded_shape[1]),
                    static_cast<size_t>(decoded_shape[2]))};
#else
  return GenericImage::DecodeImpl(type, encoded_buffer, length);
#endif
}

}  // namespace dali
