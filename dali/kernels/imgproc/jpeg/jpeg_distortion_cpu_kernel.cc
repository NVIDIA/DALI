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

// Quantization table with precomputed reciprocals so the hot-path quantize step
// uses multiplications instead of divisions (mirrors the GPU's __frcp_rn use).
struct QuantTable {
  float q[8][8];
  float inv_q[8][8];
};

inline QuantTable PrepareQuantTable(const mat<8, 8, uint8_t> &Q) {
  QuantTable t;
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      float qv = static_cast<float>(Q(i, j));
      t.q[i][j] = qv;
      t.inv_q[i][j] = 1.0f / qv;
    }
  }
  return t;
}

// 8x8 DCT/quantize helpers templated on the row stride of the surrounding
// buffer. RowStride=8 for standalone 8x8 blocks (chroma); RowStride=macro_w
// for luma blocks tiled inside a flat macro_h x macro_w scratch buffer.
template <int RowStride>
inline void fwd_dct_8x8(float *blk) {
  for (int r = 0; r < 8; r++)
    dct_fwd_8x8_1d<1>(blk + r * RowStride);
  for (int c = 0; c < 8; c++)
    dct_fwd_8x8_1d<RowStride>(blk + c);
}

template <int RowStride>
inline void inv_dct_8x8(float *blk) {
  for (int c = 0; c < 8; c++)
    dct_inv_8x8_1d<RowStride>(blk + c);
  for (int r = 0; r < 8; r++)
    dct_inv_8x8_1d<1>(blk + r * RowStride);
}

template <int RowStride>
inline void quantize_8x8(float *blk, const QuantTable &Q) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      float &v = blk[i * RowStride + j];
      v = Q.q[i][j] * std::round(v * Q.inv_q[i][j]);
    }
  }
}

// Process a single macroblock. BoundsCheck=true is only needed for the partial
// rightmost/bottommost macroblock; interior macroblocks compile to a faster path
// without input clamps or output bounds checks.
template <bool HorzSub, bool VertSub, bool BoundsCheck>
inline void ProcessMacroblock(uint8_t *out, const uint8_t *in, int H, int W,
                              int by, int bx,
                              const QuantTable &lumaQ,
                              const QuantTable &chromaQ) {
  constexpr int LX = 1 + static_cast<int>(HorzSub);
  constexpr int LY = 1 + static_cast<int>(VertSub);
  constexpr int macro_w = 8 * LX;
  constexpr int macro_h = 8 * LY;

  auto sample_rgb = [&](int x, int y) -> vec<3, uint8_t> {
    if constexpr (BoundsCheck) {
      x = std::clamp(x, 0, W - 1);
      y = std::clamp(y, 0, H - 1);
    }
    const uint8_t *p = in + (static_cast<int64_t>(y) * W + x) * 3;
    return {p[0], p[1], p[2]};
  };

  // Flat row-major luma buffer covering the whole macroblock. The DCT/quantize
  // calls walk it as a tiling of LY*LX 8x8 blocks via the row-stride template.
  float luma[macro_h][macro_w];
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
          rgb[i][j] = sample_rgb(x0 + j, y0 + i);
        }
      }

      // Compute average rgb for chroma (in uint8_t to match GPU kernel behavior)
      vec<3, uint8_t> avg_rgb;
      if constexpr (HorzSub && VertSub) {
        avg_rgb = {
          ConvertSat<uint8_t>((rgb[0][0][0] + rgb[0][1][0] + rgb[1][0][0] + rgb[1][1][0])
                              * 0.25f),
          ConvertSat<uint8_t>((rgb[0][0][1] + rgb[0][1][1] + rgb[1][0][1] + rgb[1][1][1])
                              * 0.25f),
          ConvertSat<uint8_t>((rgb[0][0][2] + rgb[0][1][2] + rgb[1][0][2] + rgb[1][1][2])
                              * 0.25f)
        };
      } else if constexpr (HorzSub) {
        avg_rgb = {
          ConvertSat<uint8_t>((rgb[0][0][0] + rgb[0][1][0]) * 0.5f),
          ConvertSat<uint8_t>((rgb[0][0][1] + rgb[0][1][1]) * 0.5f),
          ConvertSat<uint8_t>((rgb[0][0][2] + rgb[0][1][2]) * 0.5f)
        };
      } else if constexpr (VertSub) {
        avg_rgb = {
          ConvertSat<uint8_t>((rgb[0][0][0] + rgb[1][0][0]) * 0.5f),
          ConvertSat<uint8_t>((rgb[0][0][1] + rgb[1][0][1]) * 0.5f),
          ConvertSat<uint8_t>((rgb[0][0][2] + rgb[1][0][2]) * 0.5f)
        };
      } else {
        avg_rgb = rgb[0][0];
      }

      cb[cy][cx] = static_cast<float>(color::jpeg::rgb_to_cb<uint8_t>(avg_rgb)) - 128.0f;
      cr[cy][cx] = static_cast<float>(color::jpeg::rgb_to_cr<uint8_t>(avg_rgb)) - 128.0f;

      for (int i = 0; i < LY; i++) {
        for (int j = 0; j < LX; j++) {
          luma[cy * LY + i][cx * LX + j] =
              static_cast<float>(color::jpeg::rgb_to_y<uint8_t>(rgb[i][j])) - 128.0f;
        }
      }
    }
  }

  // DCT, quantize, inverse DCT
  fwd_dct_8x8<8>(&cb[0][0]);
  fwd_dct_8x8<8>(&cr[0][0]);
  for (int by_ = 0; by_ < LY; by_++)
    for (int bx_ = 0; bx_ < LX; bx_++)
      fwd_dct_8x8<macro_w>(&luma[by_ * 8][bx_ * 8]);

  quantize_8x8<8>(&cb[0][0], chromaQ);
  quantize_8x8<8>(&cr[0][0], chromaQ);
  for (int by_ = 0; by_ < LY; by_++)
    for (int bx_ = 0; bx_ < LX; bx_++)
      quantize_8x8<macro_w>(&luma[by_ * 8][bx_ * 8], lumaQ);

  inv_dct_8x8<8>(&cb[0][0]);
  inv_dct_8x8<8>(&cr[0][0]);
  for (int by_ = 0; by_ < LY; by_++)
    for (int bx_ = 0; bx_ < LX; bx_++)
      inv_dct_8x8<macro_w>(&luma[by_ * 8][bx_ * 8]);

  // Backward path: shift +128, YCbCr->RGB, write
  for (int cy = 0; cy < 8; cy++) {
    for (int cx = 0; cx < 8; cx++) {
      uint8_t Cb_u = ConvertSat<uint8_t>(cb[cy][cx] + 128.0f);
      uint8_t Cr_u = ConvertSat<uint8_t>(cr[cy][cx] + 128.0f);

      for (int i = 0; i < LY; i++) {
        for (int j = 0; j < LX; j++) {
          int y = by + cy * LY + i;
          int x = bx + cx * LX + j;
          if constexpr (BoundsCheck) {
            if (x >= W || y >= H)
              continue;
          }

          uint8_t Y_u = ConvertSat<uint8_t>(luma[cy * LY + i][cx * LX + j] + 128.0f);

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

template <bool HorzSub, bool VertSub>
void RunSampleImpl(uint8_t *out, const uint8_t *in, int H, int W,
                   const QuantTable &lumaQ,
                   const QuantTable &chromaQ) {
  constexpr int LX = 1 + static_cast<int>(HorzSub);
  constexpr int LY = 1 + static_cast<int>(VertSub);
  constexpr int macro_w = 8 * LX;
  constexpr int macro_h = 8 * LY;

  const int aligned_W = align_up(W, macro_w);
  const int aligned_H = align_up(H, macro_h);

  for (int by = 0; by < aligned_H; by += macro_h) {
    bool edge_y = by + macro_h > H;
    for (int bx = 0; bx < aligned_W; bx += macro_w) {
      bool edge_x = bx + macro_w > W;
      if (edge_x || edge_y) {
        ProcessMacroblock<HorzSub, VertSub, true>(out, in, H, W, by, bx, lumaQ, chromaQ);
      } else {
        ProcessMacroblock<HorzSub, VertSub, false>(out, in, H, W, by, bx, lumaQ, chromaQ);
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

  const auto lumaQ = PrepareQuantTable(GetLumaQuantizationTable(quality));
  const auto chromaQ = PrepareQuantTable(GetChromaQuantizationTable(quality));

  BOOL_SWITCH(horz_subsample, HorzSub, (
    BOOL_SWITCH(vert_subsample, VertSub, (
      RunSampleImpl<HorzSub, VertSub>(out.data, in.data, H, W, lumaQ, chromaQ);
    ));  // NOLINT
  ));  // NOLINT
}

}  // namespace jpeg
}  // namespace kernels
}  // namespace dali
