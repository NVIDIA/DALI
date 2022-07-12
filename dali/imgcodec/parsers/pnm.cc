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

ImageInfo PnmParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  ssize_t length = stream->Size();

  // http://netpbm.sourceforge.net/doc/ppm.html
  stream->Skip();
  int channels = 1;
  auto magic = stream->ReadOne<char>();
  if ((magic == '3') || (magic == '6')) {
    channels = 3;             // formats "p3" and "p6" are RGB color, all
                              // other formats are bitmaps or greymaps
  }

  int state = 0;
  enum {
    STATE_START = 0, STATE_WIDTH = 1, STATE_HEIGHT = 2, STATE_DONE = 3
  };
  std::array<int, 2> dim = {0, 0};
  auto cur = stream->ReadOne<char>();
  do {
    // comments can appear in the middle of tokens, and the newline at the
    // end is part of the comment, not counted as whitespace
    // http://netpbm.sourceforge.net/doc/pbm.html
    if (cur == '#') {
      do {
        cur = stream->ReadOne<char>();
      } while (cur != '\n');
      cur = stream->ReadOne<char>();
    } else if (isspace(cur)) {
      if (++state < STATE_DONE) {
        do {
          cur = stream->ReadOne<char>();
        } while (isspace(cur));
      }
    } else {
      DALI_ENFORCE(isdigit(cur));
      DALI_ENFORCE(state);
      dim[state - 1] = dim[state - 1] * 10 + (cur - '0');
      cur = stream->ReadOne<char>();
    }
  } while (state < STATE_DONE);

  int h = dim[STATE_HEIGHT - 1];
  int w = dim[STATE_WIDTH - 1];
  ImageInfo info;
  info.shape = {h, w, channels};
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
