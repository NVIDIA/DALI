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

#include <gtest/gtest.h>
#include "dali/kernels/imgproc/sampler_test.h"

namespace dali {
namespace kernels {

using Pixel = std::array<uint8_t, 3>;

TEST(SamplerCPU, NN) {
  SamplerTestData sd;
  auto surf = sd.GetSurface(false);
  ASSERT_EQ(surf.channels, 3);
  Sampler<DALI_INTERP_NN, uint8_t> sampler(surf);

  Pixel border = { 50, 100, 200 };

  for (float y = -1; y <= surf.height+1; y += 0.1f) {
    int iy = floorf(y);
    for (float x = -1; x <= surf.width+1; x += 0.1f) {
      int ix = floorf(x);

      Pixel ref;
      if (ix < 0 || iy < 0 || ix >= surf.width || iy >= surf.height) {
        ref = border;
      } else {
        for (int c = 0; c < surf.channels; c++)
          ref[c] = surf(ix, iy, c);
      }
      Pixel pixel;

      sampler(pixel.data(), x, y, border.data());
      EXPECT_EQ(pixel, ref) << " mismatch at (" << x << ",  " << y << ")";
      for (int c = 0; c < surf.channels; c++) {
        EXPECT_EQ(sampler.at(ix, iy, c, border.data()), ref[c]);
        EXPECT_EQ(sampler.at(x,  y,  c, border.data()), ref[c]);
      }
    }
  }
}

TEST(SamplerCPU, Linear) {
  SamplerTestData sd;
  auto surf = sd.GetSurface(false);
  Sampler<DALI_INTERP_LINEAR, uint8_t> sampler(surf);

  Pixel border = { 10, 10, 10 };
  ASSERT_EQ(sampler.surface.channels, 3);

  auto fetch = [&surf, &border](int ix, int iy)->Pixel {
    if (ix < 0 || iy < 0 || ix >= surf.width || iy >= surf.height) {
      return border;
    } else {
      Pixel px;
      for (int c = 0; c < surf.channels; c++)
        px[c] = surf(ix, iy, c);
      return px;
    }
  };

  for (int iy = -1; iy <= surf.height; iy++) {
    for (int ix = -1; ix <= surf.width; ix++) {
      Pixel ref;
      if (ix < 0 || iy < 0 || ix >= surf.width || iy >= surf.height) {
        ref = border;
      } else {
        for (int c = 0; c < surf.channels; c++)
          ref[c] = surf(ix, iy, c);
      }

      float x = ix + 0.5f;
      float y = iy + 0.5f;

      Pixel pixel = { 0, 0, 0 };
      sampler(pixel.data(), x, y, border.data());
      EXPECT_EQ(pixel, ref) << " mismatch at (" << x << ",  " << y << ")";
    }
  }

  for (float y = -1; y <= surf.height+2; y += 0.125f) {
    float fy = y - 0.5f;
    int iy0 = floorf(fy);
    int iy1 = iy0 + 1;
    float qy = fy - iy0;
    float py = 1 - qy;

    for (float x = -1; x <= surf.width+2; x += 0.125f) {
      float fx = x - 0.5f;
      int ix0 = floorf(fx);
      int ix1 = ix0 + 1;

      Pixel src[4];
      src[0] = fetch(ix0, iy0);
      src[1] = fetch(ix1, iy0);
      src[2] = fetch(ix0, iy1);
      src[3] = fetch(ix1, iy1);
      Pixel ref;

      float qx = fx - ix0;
      float px = 1 - qx;
      for (int c = 0; c < surf.channels; c++) {
        float s0 = src[0][c] * px + src[1][c] * qx;
        float s1 = src[2][c] * px + src[3][c] * qx;
        ref[c] = s0 * py + s1 * qy;
      }

      Pixel pixel = { 0, 0, 0 };
      sampler(pixel.data(), x, y, border.data());
      EXPECT_EQ(pixel, ref) << " mismatch at (" << x << ",  " << y << ")";
      for (int c = 0; c < surf.channels; c++) {
        EXPECT_EQ(sampler.at<uint8_t>(x, y, c, border.data()), ref[c])
         << " mismatch at (" << x << ",  " << y << ")[" << c << "]";
      }
    }
  }
}

}  // namespace kernels
}  // namespace dali
