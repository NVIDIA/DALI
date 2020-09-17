// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/image/jpeg2k.h"
#include "dali/core/span.h"
#include "dali/core/byte_io.h"

namespace dali {

using block_type_t = std::array<uint8_t, 4>;

namespace {

constexpr block_type_t jp2_sig_type = {{'j', 'P', ' ', ' '}};

constexpr block_type_t jp2_format_type = {{'f', 't', 'y', 'p'}};

constexpr block_type_t jp2_header_type = {{'j', 'p', '2', 'h'}};

constexpr block_type_t jp2_im_header_type = {{'i', 'h', 'd', 'r'}};

constexpr uint32_t kBlockHdrSize = 8;

inline uint32_t read_block_size(const uint8_t *data) {
  // Size is the first 4-byte chunk of a block.
  return ReadValueBE<uint32_t>(data);
}

bool validate_block_type(const uint8_t *block_ptr, const block_type_t &type) {
  // Block type is its second 4-byte chunk.
  return span<const uint8_t, 4>(block_ptr + 4) == make_cspan(type);
}

uint32_t advance_one_block(span<const uint8_t> data, uint32_t index, const block_type_t &type) {
  DALI_ENFORCE(index + kBlockHdrSize < static_cast<size_t>(data.size()) &&
               validate_block_type(&data[index], type));
  index += read_block_size(&data[index]);
  DALI_ENFORCE(index < data.size());
  return index;
}

}  // namespace

bool CheckIsJPEG2k(const uint8_t *jpeg2k, int size) {
  if (size < static_cast<int>(kBlockHdrSize))
    return false;
  assert(jpeg2k != nullptr && "null pointer passed with non-zero size");
  return validate_block_type(jpeg2k, jp2_sig_type);
}

Image::Shape Jpeg2kImage::PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const {
  assert(encoded_buffer);
  auto data = span<const uint8_t>(encoded_buffer, length);
  uint32_t index = 0;
  index = advance_one_block(data, index, jp2_sig_type);
  index = advance_one_block(data, index, jp2_format_type);
  DALI_ENFORCE(validate_block_type(&data[index], jp2_header_type));
  index += kBlockHdrSize;
  DALI_ENFORCE(validate_block_type(&data[index], jp2_im_header_type));
  DALI_ENFORCE(index + kBlockHdrSize + 2*sizeof(uint32_t) + sizeof(uint16_t)
                 < static_cast<size_t>(data.size()));
  index += kBlockHdrSize;
  auto height = ReadValueBE<uint32_t>(&data[index]);
  index += sizeof(uint32_t);
  auto width = ReadValueBE<uint32_t>(&data[index]);
  index += sizeof(uint32_t);
  auto channels = ReadValueBE<uint16_t>(&data[index]);
  return {height, width, channels};
}

}  // namespace dali
