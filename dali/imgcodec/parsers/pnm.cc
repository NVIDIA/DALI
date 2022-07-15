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

#include "dali/imgcodec/parsers/pnm.h"

namespace dali {
namespace imgcodec {

namespace {
  // comments can appear in the middle of tokens, and the newline at the
  // end is part of the comment, not counted as whitespace
  // http://netpbm.sourceforge.net/doc/pbm.html
  void skip_comment(InputStream &stream) {
    char c;
    do {
      c = stream.ReadOne<char>();
    } while (c != '\n');
  }

  void skip_spaces(InputStream &stream) {
    while (true) {
      char c = stream.ReadOne<char>();
      if (c == '#')
        skip_comment(stream);
      else if (!isspace(c))
        break;
    }
    stream.Skip(-1);  // return the nonspace byte to the stream
  }

  int parse_int(InputStream &stream) {
    int result = 0;
    while (true) {
      char c = stream.ReadOne<char>();
      if (isdigit(c))
        result = result * 10 + (c - '0');
      else if (c == '#')
        skip_comment(stream);
      else
        break;
    }
    stream.Skip(-1);  // return the nondigit byte to the stream
    return result;
  }
}  // namespace

ImageInfo PnmParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ssize_t length = stream->Size();

  // http://netpbm.sourceforge.net/doc/ppm.html
  stream->Skip();
  int channels = 1;
  auto magic = stream->ReadOne<char>();
  if ((magic == '3') || (magic == '6')) {
    channels = 3;             // formats "P3" and "P6" are RGB color, all
                              // other formats are bitmaps or greymaps
  }

  skip_spaces(*stream);
  int width = parse_int(*stream);
  skip_spaces(*stream);
  int height = parse_int(*stream);

  ImageInfo info;
  info.shape = {height, width, channels};
  return info;
}

bool PnmParser::CanParse(ImageSource *encoded) const {
  uint8_t header[3];
  return ReadHeader(header, encoded, sizeof(header)) == sizeof(header)
      && header[0] == 'P'
      && '1' <= header[1] && header[1] <= '6'
      && isspace(header[2]);
}

}  // namespace imgcodec
}  // namespace dali
