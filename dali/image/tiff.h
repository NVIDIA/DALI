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

#ifndef DALI_IMAGE_TIFF_H_
#define DALI_IMAGE_TIFF_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "dali/image/generic_image.h"

namespace dali {

/**
 * Class, that handles byte buffer for tiff image
 */
class TiffBuffer {
 public:
  explicit TiffBuffer(const std::string &buffer, bool little_endian = false) :
          stream_(buffer),
          little_endian_(little_endian) {
    buffer_size_ = buffer.length();
  }


  /**
   * Reads value of given type (byte length), at given offset in the buffer.
   * This function guarantees, that such value can be read.
   * @tparam ValueType Type of return value
   * @param offset Offset (in bytes) from beginning of the buffer
   * @return value
   */
  template<typename ValueType>
  ValueType Read(unsigned int offset = 0) {
    DALI_ENFORCE(stream_.good());
    DALI_ENFORCE(offset + sizeof(ValueType) < buffer_size_);
    static_assert(std::is_integral<ValueType>::value, "Only integral values supported");

    stream_.seekg(offset);
    ValueType ret;
    stream_.read(reinterpret_cast<char *>(&ret), sizeof(ValueType));
    if (little_endian_) {
      to_little_endian(&ret);
    }
    DALI_ENFORCE(stream_.good());
    return ret;
  }

 private:
  /**
   * Converts given value in big-endian representation to little-endian
   * @tparam T
   * @param value pointer to given value
   */
  template<typename T>
  void to_little_endian(T *value) {
    static_assert(std::is_integral<T>::value, "Converting floating point value unsupported");

    char *value_bytes = reinterpret_cast<char *>(value);
    std::vector<char> value_copy(value_bytes, value_bytes + sizeof(T));

    for (auto it = value_copy.rbegin(); it != value_copy.rend(); ++it) {
      value_bytes[std::distance(value_copy.rbegin(), it)] = *it;
    }
  }

  std::istringstream stream_;
  size_t buffer_size_;
  bool little_endian_;
};

/**
 * Tiff image decoding is performed using OpenCV, thus it's the same as Generic decoding
 */
class TiffImage : public GenericImage {
 public:
  TiffImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

 protected:
  Image::Shape PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const override;
};

}  // namespace dali

#endif  // DALI_IMAGE_TIFF_H_
