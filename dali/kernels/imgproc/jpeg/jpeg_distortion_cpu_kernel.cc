// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/imgproc/jpeg/jpeg_distortion_cpu_kernel.h"

#include <algorithm>
#include <cmath>
#include "dali/core/convert.h"
#include "dali/core/geom/mat.h"
#include "dali/core/geom/vec.h"
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"
#include "dali/kernels/imgproc/jpeg/dct_8x8.h"
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_kernel.h"

namespace dali {
namespace kernels {
namespace jpeg {

namespace {

inline vec<3, uint8_t> sample_rgb_clamped(const uint8_t *in, int H, int W, int x, int y) {
  x = std::clamp(x, 0, W - 1);
  y = std::clamp(y, 0, H - 1);
  const uint8_t *p = in + (static_cast<int64_t>(y) * W + x) * 3;
  return {p[0], p[1], p[2]};
}

inline void fwd_dct_8x8(float blk[8][8]) {
  for (int r = 0; r < 8; r++)
    dct_fwd_8x8_1d<1>(&blk[r][0]);
  for (int c = 0; c < 8; c++)
    dct_fwd_8x8_1d<8>(&blk[0][c]);
}

inline void inv_dct_8x8(float blk[8][8]) {
  for (int c = 0; c < 8; c++)
    dct_inv_8x8_1d<8>(&blk[0][c]);
  for (int r = 0; r < 8; r++)
    dct_inv_8x8_1d<1>(&blk[r][0]);
}

inline void quantize_8x8(float blk[8][8], const mat<8, 8, uint8_t> &Q) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      float q = static_cast<float>(Q(i, j));
      blk[i][j] = q * std::round(blk[i][j] / q);
    }
  }
}

template <bool HorzSub, bool VertSub>
void RunSampleImpl(uint8_t *out, const uint8_t *in, int H, int W,
                   const mat<8, 8, uint8_t> &lumaQ,
                   const mat<8, 8, uint8_t> &chromaQ) {
  constexpr int LX = 1 + static_cast<int>(HorzSub);
  constexpr int LY = 1 + static_cast<int>(VertSub);
  constexpr int macro_w = 8 * LX;
  constexpr int macro_h = 8 * LY;

  const int aligned_W = align_up(W, macro_w);
  const int aligned_H = align_up(H, macro_h);

  for (int by = 0; by < aligned_H; by += macro_h) {
    for (int bx = 0; bx < aligned_W; bx += macro_w) {
      float luma[LY][LX][8][8];
      float cb[8][8];
      float cr[8][8];

      // Forward path: sample + RGB->YCbCr (uint8_t domain) + shift -128
      for (int cy = 0; cy < 8; cy++) {
        for (int cx = 0; cx < 8; cx++) {
          int y0 = by + cy * LY;
          int x0 = bx + cx * LX;

          vec<3, uint8_t> rgb[LY][LX];
          for (int i = 0; i < LY; i++) {
            for (int j = 0; j < LX; j++) {
              rgb[i][j] = sample_rgb_clamped(in, H, W, x0 + j, y0 + i);
            }
          }

          // Compute average rgb for chroma (in uint8_t to match GPU kernel behavior)
          vec<3, uint8_t> avg_rgb;
          if (HorzSub && VertSub) {
            avg_rgb = {
              ConvertSat<uint8_t>((rgb[0][0][0] + rgb[0][1][0] + rgb[1][0][0] + rgb[1][1][0])
                                  * 0.25f),
              ConvertSat<uint8_t>((rgb[0][0][1] + rgb[0][1][1] + rgb[1][0][1] + rgb[1][1][1])
                                  * 0.25f),
              ConvertSat<uint8_t>((rgb[0][0][2] + rgb[0][1][2] + rgb[1][0][2] + rgb[1][1][2])
                                  * 0.25f)
            };
          } else if (HorzSub) {
            avg_rgb = {
              ConvertSat<uint8_t>((rgb[0][0][0] + rgb[0][1][0]) * 0.5f),
              ConvertSat<uint8_t>((rgb[0][0][1] + rgb[0][1][1]) * 0.5f),
              ConvertSat<uint8_t>((rgb[0][0][2] + rgb[0][1][2]) * 0.5f)
            };
          } else if (VertSub) {
            avg_rgb = {
              ConvertSat<uint8_t>((rgb[0][0][0] + rgb[1][0][0]) * 0.5f),
              ConvertSat<uint8_t>((rgb[0][0][1] + rgb[1][0][1]) * 0.5f),
              ConvertSat<uint8_t>((rgb[0][0][2] + rgb[1][0][2]) * 0.5f)
            };
          } else {
            avg_rgb = rgb[0][0];
          }

          // Convert to YCbCr in uint8_t domain (matches GPU kernel), then shift by -128
          cb[cy][cx] = static_cast<float>(color::jpeg::rgb_to_cb<uint8_t>(avg_rgb)) - 128.0f;
          cr[cy][cx] = static_cast<float>(color::jpeg::rgb_to_cr<uint8_t>(avg_rgb)) - 128.0f;

          for (int i = 0; i < LY; i++) {
            for (int j = 0; j < LX; j++) {
              int local_y = cy * LY + i;
              int local_x = cx * LX + j;
              int blk_y = local_y / 8;
              int blk_x = local_x / 8;
              int in_blk_y = local_y & 7;
              int in_blk_x = local_x & 7;
              luma[blk_y][blk_x][in_blk_y][in_blk_x] =
                  static_cast<float>(color::jpeg::rgb_to_y<uint8_t>(rgb[i][j])) - 128.0f;
            }
          }
        }
      }

      // DCT, quantize, inverse DCT
      fwd_dct_8x8(cb);
      fwd_dct_8x8(cr);
      for (int i = 0; i < LY; i++)
        for (int j = 0; j < LX; j++)
          fwd_dct_8x8(luma[i][j]);

      quantize_8x8(cb, chromaQ);
      quantize_8x8(cr, chromaQ);
      for (int i = 0; i < LY; i++)
        for (int j = 0; j < LX; j++)
          quantize_8x8(luma[i][j], lumaQ);

      inv_dct_8x8(cb);
      inv_dct_8x8(cr);
      for (int i = 0; i < LY; i++)
        for (int j = 0; j < LX; j++)
          inv_dct_8x8(luma[i][j]);

      // Backward path: shift +128, YCbCr->RGB, write
      for (int cy = 0; cy < 8; cy++) {
        for (int cx = 0; cx < 8; cx++) {
          uint8_t Cb_u = ConvertSat<uint8_t>(cb[cy][cx] + 128.0f);
          uint8_t Cr_u = ConvertSat<uint8_t>(cr[cy][cx] + 128.0f);

          for (int i = 0; i < LY; i++) {
            for (int j = 0; j < LX; j++) {
              int y = by + cy * LY + i;
              int x = bx + cx * LX + j;
              if (x >= W || y >= H)
                continue;

              int local_y = cy * LY + i;
              int local_x = cx * LX + j;
              int blk_y = local_y / 8;
              int blk_x = local_x / 8;
              int in_blk_y = local_y & 7;
              int in_blk_x = local_x & 7;
              uint8_t Y_u = ConvertSat<uint8_t>(
                  luma[blk_y][blk_x][in_blk_y][in_blk_x] + 128.0f);

              auto rgb = color::jpeg::ycbcr_to_rgb<uint8_t>(
                  vec<3, uint8_t>{Y_u, Cb_u, Cr_u});

              uint8_t *p = out + (static_cast<int64_t>(y) * W + x) * 3;
              p[0] = rgb[0];
              p[1] = rgb[1];
              p[2] = rgb[2];
            }
          }
        }
      }
    }
  }
}

}  // namespace

void JpegCompressionDistortionCPU::RunSample(
    const TensorView<StorageCPU, uint8_t, 3> &out,
    const TensorView<StorageCPU, const uint8_t, 3> &in,
    int quality,
    bool horz_subsample,
    bool vert_subsample) {
  const auto &sh = in.shape;
  assert(out.shape == sh);
  const int H = sh[0];
  const int W = sh[1];
  assert(sh[2] == 3);
  if (H == 0 || W == 0) return;

  const auto lumaQ = GetLumaQuantizationTable(quality);
  const auto chromaQ = GetChromaQuantizationTable(quality);

  BOOL_SWITCH(horz_subsample, HorzSub, (
    BOOL_SWITCH(vert_subsample, VertSub, (
      RunSampleImpl<HorzSub, VertSub>(out.data, in.data, H, W, lumaQ, chromaQ);
    ));  // NOLINT
  ));  // NOLINT
}

}  // namespace jpeg
}  // namespace kernels
}  // namespace dali
