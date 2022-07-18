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

#include "dali/imgcodec/parsers/jpeg2000.h"
#include "dali/core/byte_io.h"

namespace dali {
namespace imgcodec {

namespace {

using block_type_t = std::array<uint8_t, 4>;
const block_type_t jp2_sig_type = {'j', 'P', ' ', ' '};
const block_type_t jp2_format_type = {'f', 't', 'y', 'p'};
const block_type_t jp2_header_type = {'j', 'p', '2', 'h'};
const block_type_t jp2_im_header_type = {'i', 'h', 'd', 'r'};

void advance_one_block(InputStream &stream, const block_type_t &expected_type) {
  const auto block_size = ReadValueBE<uint32_t>(stream);
  const auto block_type = stream.ReadOne<block_type_t>();
  DALI_ENFORCE(block_type == expected_type, "Unexpected block type");

  // Skipping the rest of the block
  stream.Skip(block_size - sizeof(block_size) - sizeof(block_type));
}

}  // namespace

ImageInfo Jpeg2000Parser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();

  advance_one_block(*stream, jp2_sig_type);
  advance_one_block(*stream, jp2_format_type);

  // jp2 header starts
  stream->Skip<uint32_t>();  // block size skipped
  DALI_ENFORCE(stream->ReadOne<block_type_t>() == jp2_header_type,
               "Expected type 'jp2h'");
  stream->Skip<uint32_t>();  // imabe block size skipped
  DALI_ENFORCE(stream->ReadOne<block_type_t>() == jp2_im_header_type,
               "Expected type 'ihdr'");

  const int h = ReadValueBE<uint32_t>(*stream);
  const int w = ReadValueBE<uint32_t>(*stream);
  const int c = ReadValueBE<uint16_t>(*stream);

  ImageInfo info;
  info.shape = {h, w, c};
  return info;
}

bool Jpeg2000Parser::CanParse(ImageSource *encoded) const {
  uint8_t header[8];
  if (!ReadHeader(header, encoded, sizeof(header)))
    return false;

  MemInputStream stream(header, sizeof(header));
  stream.Skip<uint32_t>();  // block size
  return stream.ReadOne<block_type_t>() == jp2_sig_type;
}

}  // namespace imgcodec
}  // namespace dali
