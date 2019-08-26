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

#ifndef DALI_UTIL_CROP_WINDOW_H_
#define DALI_UTIL_CROP_WINDOW_H_

#include <functional>

namespace dali {

struct CropWindow {
    int x, y, w, h;

    CropWindow(int _x, int _y, int _w, int _h)
      : x(_x), y(_y), w(_w), h(_h)
    {}

    CropWindow()
      : x(0), y(0), w(0), h(0)
    {}

    operator bool() const {
      return w > 0 && h > 0;
    }

    inline bool operator==(const CropWindow& oth) const {
      return x == oth.x
          && y == oth.y
          && h == oth.h
          && w == oth.w;
    }

    inline bool operator!=(const CropWindow& oth) const {
      return !operator==(oth);
    }

    inline bool IsInRange(int H, int W) const {
      return x >= 0
          && x < W
          && y >= 0
          && y < H
          && x+w >= 0
          && x+w <= W
          && y+h >= 0
          && y+h <= H;
    }
};

using CropWindowGenerator = std::function<CropWindow(int /*H*/, int /*W*/)>;

}  // namespace dali

#endif  // DALI_UTIL_CROP_WINDOW_H_
