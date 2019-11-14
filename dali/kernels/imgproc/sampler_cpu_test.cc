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
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {

using Pixel = std::array<uint8_t, 3>;
using PixelF = std::array<float, 3>;

TEST(Sampler2D_CPU, NN) {
  SamplerTestData<uint8_t> sd;
  auto surf = sd.GetSurface2D(false);
  ASSERT_EQ(surf.channels, 3);
  Sampler2D<DALI_INTERP_NN, uint8_t> sampler(surf);

  Pixel border = { 50, 100, 200 };

  for (float y = -1; y <= surf.size.y+1; y += 0.1f) {
    int iy = floorf(y);
    for (float x = -1; x <= surf.size.x+1; x += 0.1f) {
      int ix = floorf(x);

      Pixel ref;
      if (ix < 0 || iy < 0 || ix >= surf.size.x || iy >= surf.size.y) {
        ref = border;
      } else {
        for (int c = 0; c < surf.channels; c++)
          ref[c] = surf(ix, iy, c);
      }
      Pixel pixel;

      vec2 pos = { x, y };
      ivec2 ipos = { ix, iy };
      sampler(pixel.data(), pos, border.data());
      EXPECT_EQ(pixel, ref) << " mismatch at " << pos;
      for (int c = 0; c < surf.channels; c++) {
        EXPECT_EQ(sampler.at(ipos, c, border.data()), ref[c]);
        EXPECT_EQ(sampler.at(pos,  c, border.data()), ref[c]);
      }
    }
  }
}

TEST(Sampler3D_CPU, NN) {
  SamplerTestData<uint8_t> sd;
  auto surf = sd.GetSurface3D(false);
  ASSERT_EQ(surf.channels, 3);
  Sampler3D<DALI_INTERP_NN, uint8_t> sampler(surf);

  Pixel border = { 50, 100, 200 };

  for (float z = -1; z <= surf.size.z+1; z += 0.1f) {
    int iz = floorf(z);
    for (float y = -1; y <= surf.size.y+1; y += 0.1f) {
      int iy = floorf(y);
      for (float x = -1; x <= surf.size.x+1; x += 0.1f) {
        int ix = floorf(x);

        Pixel ref;
        if (ix < 0 || iy < 0 || iz < 0 ||
            ix >= surf.size.x || iy >= surf.size.y || iz >= surf.size.z) {
          ref = border;
        } else {
          for (int c = 0; c < surf.channels; c++)
            ref[c] = surf(ix, iy, iz, c);
        }
        Pixel pixel;

        vec3 pos = { x, y, z };
        ivec3 ipos = { ix, iy, iz };

        sampler(pixel.data(), pos, border.data());
        EXPECT_EQ(pixel, ref) << " mismatch at " << pos;
        for (int c = 0; c < surf.channels; c++) {
          EXPECT_EQ(sampler.at(ipos, c, border.data()), ref[c]);
          EXPECT_EQ(sampler.at(pos,  c, border.data()), ref[c]);
        }
      }
    }
  }
}


TEST(Sampler2D_CPU, Linear) {
  SamplerTestData<uint8_t> sd;
  auto surf = sd.GetSurface2D(false);
  Sampler2D<DALI_INTERP_LINEAR, uint8_t> sampler(surf);

  Pixel border = { 10, 10, 10 };
  ASSERT_EQ(sampler.surface.channels, 3);

  auto fetch = [&surf, &border](int ix, int iy)->Pixel {
    if (ix < 0 || iy < 0 || ix >= surf.size.x || iy >= surf.size.y) {
      return border;
    } else {
      Pixel px;
      for (int c = 0; c < surf.channels; c++)
        px[c] = surf(ix, iy, c);
      return px;
    }
  };

  for (int iy = -1; iy <= surf.size.y; iy++) {
    for (int ix = -1; ix <= surf.size.x; ix++) {
      Pixel ref = fetch(ix, iy);
      float x = ix + 0.5f;
      float y = iy + 0.5f;

      Pixel pixel = { 0, 0, 0 };
      vec2 pos = { x, y };
      sampler(pixel.data(), pos, border.data());
      EXPECT_EQ(pixel, ref) << " mismatch at " << pos;
    }
  }

  const float epsF = 255.00006 - 255;  // 4 ULP in IEEE 32-bit float for 0-255 range
  const float eps = 0.50000025f;  // 0.5 + 4 ULP
  for (float y = -1; y <= surf.size.y+2; y += 0.125f) {
    float fy = y - 0.5f;
    int iy0 = floorf(fy);
    int iy1 = iy0 + 1;
    float qy = fy - iy0;
    float py = 1 - qy;

    for (float x = -1; x <= surf.size.x+2; x += 0.125f) {
      float fx = x - 0.5f;
      int ix0 = floorf(fx);
      int ix1 = ix0 + 1;

      Pixel src[4];
      src[0] = fetch(ix0, iy0);
      src[1] = fetch(ix1, iy0);
      src[2] = fetch(ix0, iy1);
      src[3] = fetch(ix1, iy1);
      PixelF ref;

      float qx = fx - ix0;
      float px = 1 - qx;
      for (int c = 0; c < surf.channels; c++) {
        float s0 = src[0][c] * px + src[1][c] * qx;
        float s1 = src[2][c] * px + src[3][c] * qx;
        ref[c] = s0 * py + s1 * qy;
      }

      vec2 pos = { x, y };
      Pixel pixel = { 0, 0, 0 };
      PixelF pixelF = { 0, 0, 0 };
      sampler(pixel.data(), pos, border.data());
      for (int c = 0; c < surf.channels; c++) {
        EXPECT_NEAR(pixel[c], ref[c], eps)
          << " mismatch at " << pos << "[" << c << "] when sampling all channels";
      }

      for (int c = 0; c < surf.channels; c++) {
        EXPECT_NEAR(sampler.at<uint8_t>(pos, c, border.data()), ref[c], eps)
         << " mismatch at " << pos  << "[" << c << "] when sampling single channel";
      }

      sampler(pixelF.data(), pos, border.data());
      for (int c = 0; c < surf.channels; c++) {
        EXPECT_NEAR(pixelF[c], ref[c], epsF)
          << " mismatch at " << pos << "[" << c << "] when sampling all channels";
      }

      for (int c = 0; c < surf.channels; c++) {
        EXPECT_NEAR(sampler.at<float>(pos, c, border.data()), ref[c], epsF)
         << " mismatch at " << pos << "[" << c << "] when sampling single channel";
      }
    }
  }
}

TEST(Sampler3D_CPU, Linear) {
  SamplerTestData<uint8_t> sd;
  auto surf = sd.GetSurface3D(false);
  Sampler3D<DALI_INTERP_LINEAR, uint8_t> sampler(surf);

  Pixel border = { 10, 10, 10 };
  ASSERT_EQ(sampler.surface.channels, 3);

  auto fetch = [&surf, &border](int ix, int iy, int iz)->Pixel {
    if (ix < 0 || iy < 0 || iz < 0 ||
        ix >= surf.size.x || iy >= surf.size.y || iz >= surf.size.z) {
      return border;
    } else {
      Pixel px;
      for (int c = 0; c < surf.channels; c++)
        px[c] = surf(ix, iy, iz, c);
      return px;
    }
  };

  for (int iz = -1; iz <= surf.size.z; iz++) {
    for (int iy = -1; iy <= surf.size.y; iy++) {
      for (int ix = -1; ix <= surf.size.x; ix++) {
        Pixel ref = fetch(ix, iy, iz);
        ivec3 ipos = { ix, iy, iz };
        vec3 pos = ipos + 0.5f;

        Pixel pixel = { 0, 0, 0 };
        sampler(pixel.data(), pos, border.data());
        EXPECT_EQ(pixel, ref) << " mismatch at " << pos;
        sampler(pixel.data(), ipos, border.data());
        EXPECT_EQ(pixel, ref) << " mismatch at integer position " << pos;
      }
    }
  }

  const float epsF = 255.00006 - 255;  // 4 ULP in IEEE 32-bit float for 0-255 range
  const float eps = 0.50000025f;  // 0.5 + 4 ULP
  for (float z = -1; z <= surf.size.z+2; z += 0.125f) {
    float fz = z - 0.5f;
    int iz0 = floorf(fz);
    int iz1 = iz0 + 1;
    float qz = fz - iz0;
    float pz = 1 - qz;

    for (float y = -1; y <= surf.size.y+2; y += 0.125f) {
      float fy = y - 0.5f;
      int iy0 = floorf(fy);
      int iy1 = iy0 + 1;
      float qy = fy - iy0;
      float py = 1 - qy;

      for (float x = -1; x <= surf.size.x+2; x += 0.125f) {
        float fx = x - 0.5f;
        int ix0 = floorf(fx);
        int ix1 = ix0 + 1;

        Pixel src[2][2][2];
        src[0][0][0] = fetch(ix0, iy0, iz0);
        src[0][0][1] = fetch(ix1, iy0, iz0);
        src[0][1][0] = fetch(ix0, iy1, iz0);
        src[0][1][1] = fetch(ix1, iy1, iz0);
        src[1][0][0] = fetch(ix0, iy0, iz1);
        src[1][0][1] = fetch(ix1, iy0, iz1);
        src[1][1][0] = fetch(ix0, iy1, iz1);
        src[1][1][1] = fetch(ix1, iy1, iz1);
        PixelF ref;

        float qx = fx - ix0;
        float px = 1 - qx;
        for (int c = 0; c < surf.channels; c++) {
          // interpolate in x
          float sy00 = src[0][0][0][c] * px + src[0][0][1][c] * qx;
          float sy01 = src[0][1][0][c] * px + src[0][1][1][c] * qx;
          float sy10 = src[1][0][0][c] * px + src[1][0][1][c] * qx;
          float sy11 = src[1][1][0][c] * px + src[1][1][1][c] * qx;
          // interpolate in y
          float sz0 = sy00 * py + sy01 * qy;
          float sz1 = sy10 * py + sy11 * qy;
          // interpolate in z
          ref[c] = sz0 * pz + sz1 * qz;
        }

        Pixel pixel = { 0, 0, 0 };
        PixelF pixelF = { 0, 0, 0 };
        vec3 pos = { x, y, z };
        sampler(pixel.data(), pos, border.data());
        for (int c = 0; c < surf.channels; c++) {
          EXPECT_NEAR(pixel[c], ref[c], eps)
            << " mismatch " << pos << "[" << c << "] when sampling all channels";
        }

        for (int c = 0; c < surf.channels; c++) {
          EXPECT_NEAR(sampler.at<uint8_t>(pos, c, border.data()), ref[c], eps)
          << " mismatch " << pos << ")[" << c << "] when sampling single channel";
        }

        sampler(pixelF.data(), pos, border.data());
        for (int c = 0; c < surf.channels; c++) {
          EXPECT_NEAR(pixelF[c], ref[c], epsF)
            << " mismatch at " << pos << "[" << c << "] when sampling all channels";
        }

        for (int c = 0; c < surf.channels; c++) {
          EXPECT_NEAR(sampler.at<float>(pos, c, border.data()), ref[c], epsF)
          << " mismatch at " << pos << "[" << c << "] when sampling single channel";
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace dali
