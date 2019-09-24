// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/image/pnm.h"
#include <cctype>               // for isspace() and isdigit()

namespace dali {

PnmImage::PnmImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        GenericImage(encoded_buffer, length, image_type) {
}

Image::Shape PnmImage::PeekShapeImpl(const uint8_t *pnm, size_t length) const {
  DALI_ENFORCE(pnm);

  // http://netpbm.sourceforge.net/doc/ppm.html
  auto end_ptr = pnm + length;
  auto at_ptr  = pnm + 1;
  DALI_ENFORCE(at_ptr < end_ptr);
  int channels = 1;
  if ((*at_ptr == '3') || (*at_ptr == '6')) {
      channels = 3;             // formats "p3" and "p6" are RGB color, all
                                // other formats are bitmaps or greymaps
  }
  ++at_ptr;

  int state = 0;
  enum {
    STATE_START = 0, STATE_WIDTH = 1, STATE_HEIGHT = 2, STATE_DONE = 3
  };
  int dim[2] = {0, 0};
  do {
      DALI_ENFORCE(at_ptr < end_ptr);
      // comments can appear in the middle of tokens, and the newline  at the
      // end is part of the comment, not counted as whitespace
      // http://netpbm.sourceforge.net/doc/pbm.html
      if (*at_ptr == '#') {
          do {
              ++at_ptr;
              DALI_ENFORCE(at_ptr < end_ptr);
          } while (!isspace(*at_ptr));
      } else if (isspace(*at_ptr)) {
          if (++state < STATE_DONE) {
              do {
                  ++at_ptr;
                  DALI_ENFORCE(at_ptr < end_ptr);
              } while (isspace(*at_ptr));
          }
      } else {
          DALI_ENFORCE(isdigit(*at_ptr));
          dim[state-1] = dim[state-1]*10 + (*at_ptr - '0');
          ++at_ptr;
      }
  } while (state < STATE_DONE);
  int h = dim[STATE_HEIGHT-1];
  int w = dim[STATE_WIDTH-1];
  return {h, w, channels};
}


}  // namespace dali
