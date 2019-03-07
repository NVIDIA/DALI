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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CPU_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CPU_H_

#include <cassert>
#include <type_traits>
#include "dali/kernels/static_switch.h"
#include "dali/kernels/common/convert.h"
#include "dali/kernels/imgproc/surface.h"

namespace dali {
namespace kernels {

struct FilterWindow;
struct ResamplingFilter;

void InitializeResamplingFilter(int32_t *out_indices, float *out_coeffs, int out_size,
                                float srcx0, float scale, const ResamplingFilter &filter);

template <int static_channels, bool clamp_left, bool clamp_right, typename Out, typename In>
void ResampleCol(Out *out, const In *in, int x, int w, const int32_t *in_columns,
                 const float *coeffs, int support, int dynamic_channels) {
  const float bias = std::is_integral<Out>::value ? 0.5f : 0;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  int x0 = in_columns[x];
  int k0 = x * support;

  if (static_channels < 0) {
    // we don't know how many channels we have at compile time - inner loop over filter
    for (int c = 0; c < channels; c++) {
      float sum = bias;
      for (int k = 0; k < support; k++) {
        int srcx = x0 + k;
        if (clamp_left) if (srcx < 0) srcx = 0;
        if (clamp_right) if (srcx > w-1) srcx = w-1;
        sum += coeffs[k0 + k] * in[srcx * channels + c];
      }
      out[channels * x + c] = clamp<Out>(sum);
    }
  } else {
    // we know how many channels we have at compile time - inner loop over channels
    float tmp[static_channels > 0 ? static_channels : 1];  // NOLINT
    for (int c = 0; c < channels; c++)
      tmp[c] = bias;

    for (int k = 0; k < support; k++) {
      int srcx = x0 + k;
      if (clamp_left) if (srcx < 0) srcx = 0;
      if (clamp_right) if (srcx > w-1) srcx = w-1;
      for (int c = 0; c < channels; c++) {
        tmp[c] += coeffs[k0 + k] * in[srcx * channels + c];
      }
    }

    for (int c = 0; c < channels; c++)
      out[channels * x + c] = clamp<Out>(tmp[c]);
  }
}

template <int static_channels = -1, typename Out, typename In>
void ResampleHorz_Channels(
    Surface2D<Out> out, Surface2D<In> in, const int *in_columns,
    const float *coeffs, int support) {
  const int channels = static_channels < 0 ? out.channels : static_channels;

  int first_regular_col = 0;
  int last_regular_col = out.width - 1;
  while (first_regular_col < out.width && in_columns[first_regular_col] < 0)
    first_regular_col++;
  while (last_regular_col >= 0 && in_columns[last_regular_col] + support > in.width)
    last_regular_col--;

  for (int y = 0; y < out.height; y++) {
    Out *out_row = &out(0, y);
    const In *in_row = &in(0, y);

    int x = 0;

    for (; x < first_regular_col && x <= last_regular_col; x++) {
      ResampleCol<static_channels, true, false>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }
    for (; x < first_regular_col; x++) {
      ResampleCol<static_channels, true, true>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }
    for (; x <= last_regular_col; x++) {
      ResampleCol<static_channels, false, false>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }
    for (; x < out.width; x++) {
      ResampleCol<static_channels, false, true>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }
  }
}

template <typename Out, typename In>
void ResampleVert(
    Surface2D<Out> out, Surface2D<In> in, const int32_t *in_rows,
    const float *row_coeffs, int support) {
  constexpr float bias = std::is_integral<Out>::value ? 0.5f : 0;
  constexpr int tile = 64;
  float tmp[tile];  // NOLINT

  int flat_w = out.width * out.channels;

  assert(support > 0);
  const In **in_row_ptrs = static_cast<const In **>(alloca(support * sizeof(const In *)));

  for (int y = 0; y < out.height; y++) {
    Out *out_row = &out(0, y, 0);

    for (int k = 0; k < support; k++) {
      int sy = in_rows[y] + k;
      if (sy < 0) sy = 0;
      else if (sy > in.height-1) sy = in.height-1;
      in_row_ptrs[k] = &in(0, sy);
    }

    for (int x0 = 0; x0 < flat_w; x0 += tile) {
      int tile_w = x0 + tile <= flat_w ? tile : flat_w - x0;
      assert(tile_w <= tile);
      for (int j = 0; j < tile_w; j++)
        tmp[j] = bias;

      for (int k = 0; k < support; k++) {
        float flt = row_coeffs[y * support + k];
        const In *in_row = in_row_ptrs[k];
        for (int j = 0; j < tile_w; j++) {
          tmp[j] += flt * in_row[x0 + j];
        }
      }

      for (int j = 0; j < tile_w; j++)
        out_row[x0 + j] = clamp<Out>(tmp[j]);
    }
  }
}

template <typename Out, typename In>
inline void ResampleHorz(Surface2D<Out> out, Surface2D<In> in,
                         const int *in_columns, const float *col_coeffs, int support) {
  VALUE_SWITCH(out.channels, static_channels, (1, 2, 3, 4), (
    ResampleHorz_Channels<static_channels>(out, in, in_columns, col_coeffs, support);
  ), (  // NOLINT
    ResampleHorz_Channels<-1>(out, in, in_columns, col_coeffs, support);
  ));   // NOLINT
}

template <typename Out, typename In>
inline void ResampleAxis(Surface2D<Out> out, Surface2D<In> in,
                         const int *in_indices, const float *coeffs, int support, int axis) {
  if (axis == 0)
    ResampleVert(out, in, in_indices, coeffs, support);
  else if (axis == 1)
    ResampleHorz(out, in, in_indices, coeffs, support);
  else
    assert(!"Invalid axis index");
}

/// @brief Resamples `in` using Nearest Neighbor interpolation and stores result in `out`
/// @param out - output surface
/// @param in - input surface
/// @param src_x0 - starting X coordinate of input
/// @param src_y0 - starting Y coordinate of input
/// @param scale_x - step of X input coordinate taken for each output pixel
/// @param scale_y - step of Y input coordinate taken for each output row
/// @remarks The function clamps input coordinates to fit in range defined by `in` dimensions.
///          Scales can be negative to achieve flipping.
template <typename Out, typename In>
void ResampleNN(Surface2D<Out> out, Surface2D<const In> in,
                float src_x0, float src_y0, float scale_x, float scale_y) {
  assert(out.channels == in.channels);
  assert((in.channel_stride == 1 && out.channel_stride == 1) ||
         (in.channels == 1 && out.channels == 1));
  // assume HWC layout with contiguous pixels (not necessarily rows)
  assert(out.pixel_stride == out.channels);

  if (scale_x == 1) {
    // FAST PATH - not scaling along X axis - just copy with repeated boundaries
    int sx0 = std::floor(src_x0 + 0.5f);
    int dx1 = sx0 + in.width;
    if (dx1 > out.width)
      dx1 = out.width;
    int dx0 = 0;
    if (sx0 < 0) {
      dx0 = std::min(-sx0, out.width);
    }

    float sy = src_y0 + 0.5f * scale_y;
    for (int y = 0; y < out.height; y++, sy += scale_y) {
      int srcy = std::floor(sy);

      if (srcy < 0) srcy = 0;
      else if (srcy > in.height-1) srcy = in.height-1;

      Out *out_ch = &out(0, y, 0);

      int x;
      const In *first_px = &in(0, srcy, 0);
      for (x = 0; x < dx0; x++) {
        for (int c = 0; c < out.channels; c++)
          *out_ch++ = first_px[c];
      }

      const In *in_row = &in(sx0 + dx0, srcy, 0);
      for (int j = x * out.channels; j < dx1 * out.channels; j++)
          *out_ch++ = clamp<Out>(*in_row++);

      x = dx1;
      const In *last_px = &in(in.width-1, srcy, 0);
      for (; x < out.width; x++) {
        for (int c = 0; c < out.channels; c++)
          *out_ch++ = last_px[c];
      }
    }
    return;
  }

  constexpr int max_span_width = 256;
  int col_offsets[max_span_width];  // NOLINT (kOnstant)

  for (int x0 = 0; x0 < out.width; x0 += max_span_width) {
    float sy = src_y0 + 0.5f * scale_y;

    int span_width = x0 + max_span_width <= out.width ? max_span_width : out.width - x0;

    for (int j = 0; j < span_width; j++) {
      int x = x0 + j;
      float sx = src_x0 + (x + 0.5f) * scale_x;
      int srcx = std::floor(sx);
      if (srcx < 0) srcx = 0;
      else if (srcx > in.width-1) srcx += in.width - 1;
      col_offsets[j] = srcx * in.pixel_stride;
    }

    for (int y = 0; y < out.height; y++, sy += scale_y) {
      int srcy = std::floor(sy);

      if (srcy < 0) srcy = 0;
      else if (srcy > in.height-1) srcy = in.height-1;

      const In *in_row = &in(0, srcy);
      Out *out_ch = &out(x0, y, 0);

      for (int j = 0; j < span_width; j++) {
        const In *in_ch = &in_row[col_offsets[j]];
        for (int c = 0; c < out.channels; c++) {
          *out_ch++ = clamp<Out>(*in_ch++);
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CPU_H_
