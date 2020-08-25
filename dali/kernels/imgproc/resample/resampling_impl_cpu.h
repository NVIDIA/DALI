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
#include "dali/core/static_switch.h"
#include "dali/core/convert.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {

struct FilterWindow;
struct ResamplingFilter;

DLL_PUBLIC
void InitializeResamplingFilter(int32_t *out_indices, float *out_coeffs, int out_size,
                                float srcx0, float scale, const ResamplingFilter &filter);

template <int static_channels, bool clamp_left, bool clamp_right, typename Out, typename In>
void ResampleCol(Out *out, const In *in, int x, int w, const int32_t *in_columns,
                 const float *coeffs, int support, int dynamic_channels) {
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  int x0 = in_columns[x];
  int k0 = x * support;

  if (static_channels < 0) {
    // we don't know how many channels we have at compile time - inner loop over filter
    for (int c = 0; c < channels; c++) {
      float sum = 0;
      for (int k = 0; k < support; k++) {
        int srcx = x0 + k;
        if (clamp_left) if (srcx < 0) srcx = 0;
        if (clamp_right) if (srcx > w-1) srcx = w-1;
        sum += coeffs[k0 + k] * in[srcx * channels + c];
      }
      out[channels * x + c] = ConvertSat<Out>(sum);
    }
  } else {
    // we know how many channels we have at compile time - inner loop over channels
    float tmp[static_channels > 0 ? static_channels : 1] = { 0 };  // NOLINT

    for (int k = 0; k < support; k++) {
      int srcx = x0 + k;
      if (clamp_left) if (srcx < 0) srcx = 0;
      if (clamp_right) if (srcx > w-1) srcx = w-1;
      for (int c = 0; c < channels; c++) {
        tmp[c] += coeffs[k0 + k] * in[srcx * channels + c];
      }
    }

    for (int c = 0; c < channels; c++)
      out[channels * x + c] = ConvertSat<Out>(tmp[c]);
  }
}

inline bool GetFirstAndLastRegularCol(int &first_regular_col,
                                      int &last_regular_col,
                                      int out_width, int in_width, const int *in_col_idxs,
                                      int support) {
  bool flipped = in_col_idxs[out_width-1] < in_col_idxs[0];
  first_regular_col = 0;
  last_regular_col = out_width - 1;
  if (flipped) {
    while (first_regular_col < out_width && in_col_idxs[first_regular_col] + support > in_width)
      first_regular_col++;
    while (last_regular_col >= 0 && in_col_idxs[last_regular_col] < 0)
      last_regular_col--;
  } else {
    while (first_regular_col < out_width && in_col_idxs[first_regular_col] < 0)
      first_regular_col++;
    while (last_regular_col >= 0 && in_col_idxs[last_regular_col] + support > in_width)
      last_regular_col--;
  }

  return flipped;
}

template <int static_channels = -1, typename Out, typename In>
void ResamplHorzRow(Out *out_row, int out_width, const In *in_row, int in_width, int channels,
                    const int *in_columns, const float *coeffs, int support,
                    int first_regular_col, int last_regular_col, bool flipped) {
  int x = 0;
  if (flipped) {
    for (; x < first_regular_col && x <= last_regular_col; x++) {
      ResampleCol<static_channels, false, true>(
        out_row, in_row, x, in_width, in_columns, coeffs, support, channels);
    }
  } else {
    for (; x < first_regular_col && x <= last_regular_col; x++) {
      ResampleCol<static_channels, false, false>(
        out_row, in_row, x, in_width, in_columns, coeffs, support, channels);
    }
  }

  for (; x < first_regular_col; x++) {
    ResampleCol<static_channels, true, true>(
      out_row, in_row, x, in_width, in_columns, coeffs, support, channels);
  }
  for (; x <= last_regular_col; x++) {
    ResampleCol<static_channels, false, false>(
      out_row, in_row, x, in_width, in_columns, coeffs, support, channels);
  }

  if (flipped) {
    for (; x < out_width; x++) {
      ResampleCol<static_channels, true, false>(
        out_row, in_row, x, in_width, in_columns, coeffs, support, channels);
    }
  } else {
    for (; x < out_width; x++) {
      ResampleCol<static_channels, false, true>(
        out_row, in_row, x, in_width, in_columns, coeffs, support, channels);
    }
  }
}

template <int static_channels = -1, typename Out, typename In>
void ResampleHorz_Channels(
    Surface2D<Out> out, Surface2D<In> in, const int *in_columns,
    const float *coeffs, int support) {
  const int channels = static_channels < 0 ? out.channels : static_channels;

  int first_regular_col, last_regular_col;
  bool flipped = GetFirstAndLastRegularCol(first_regular_col, last_regular_col,
                                           out.size.x, in.size.x, in_columns, support);

  for (int y = 0; y < out.size.y; y++) {
    Out *out_row = &out(0, y);
    const In *in_row = &in(0, y);

    ResamplHorzRow<static_channels>(out_row, out.size.x, in_row, in.size.x, channels,
                                    in_columns, coeffs, support,
                                    first_regular_col, last_regular_col, flipped);
  }
}


template <int static_channels = -1, typename Out, typename In>
void ResampleHorz_Channels(
    Surface3D<Out> out, Surface3D<In> in, const int *in_columns,
    const float *coeffs, int support) {
  const int channels = static_channels < 0 ? out.channels : static_channels;

  int first_regular_col, last_regular_col;
  bool flipped = GetFirstAndLastRegularCol(first_regular_col, last_regular_col,
                                           out.size.x, in.size.x, in_columns, support);

  for (int z = 0; z < out.size.z; z++) {
    for (int y = 0; y < out.size.y; y++) {
      Out *out_row = &out(0, y, z);
      const In *in_row = &in(0, y, z);

      ResamplHorzRow<static_channels>(out_row, out.size.x, in_row, in.size.x, channels,
                                      in_columns, coeffs, support,
                                      first_regular_col, last_regular_col, flipped);
    }
  }
}

template <typename Out, typename In>
void ResampleVert(
    Surface2D<Out> out, Surface2D<In> in, const int32_t *in_rows,
    const float *row_coeffs, int support) {
  constexpr int tile = 64;
  float tmp[tile];  // NOLINT

  int flat_w = out.size.x * out.channels;

  assert(support > 0);
  const In **in_row_ptrs = static_cast<const In **>(alloca(support * sizeof(const In *)));

  for (int y = 0; y < out.size.y; y++) {
    Out *out_row = &out(0, y, 0);

    for (int k = 0; k < support; k++) {
      int sy = in_rows[y] + k;
      if (sy < 0) sy = 0;
      else if (sy > in.size.y-1) sy = in.size.y-1;
      in_row_ptrs[k] = &in(0, sy);
    }

    for (int x0 = 0; x0 < flat_w; x0 += tile) {
      int tile_w = x0 + tile <= flat_w ? tile : flat_w - x0;
      assert(tile_w <= tile);
      for (int j = 0; j < tile_w; j++)
        tmp[j] = 0;

      for (int k = 0; k < support; k++) {
        float flt = row_coeffs[y * support + k];
        const In *in_row = in_row_ptrs[k];
        for (int j = 0; j < tile_w; j++) {
          tmp[j] += flt * in_row[x0 + j];
        }
      }

      for (int j = 0; j < tile_w; j++)
        out_row[x0 + j] = ConvertSat<Out>(tmp[j]);
    }
  }
}


template <typename Out, typename In>
void ResampleVert(
    Surface3D<Out> out, Surface3D<In> in, const int32_t *in_rows,
    const float *row_coeffs, int support) {
  for (int z = 0; z < out.size.z; z++) {
    ResampleVert(out.slice(z), in.slice(z), in_rows, row_coeffs, support);
  }
}

template <typename Out, typename In>
inline void ResampleDepth(Surface2D<Out> out, Surface2D<In> in,
                         const int *in_columns, const float *col_coeffs, int support) {
  assert(!"Unreachable code");
}

template <typename T>
inline Surface2D<T> FuseXY(const Surface3D<T> &surface) {
  return { surface.data,
           surface.size.x * surface.size.y, surface.size.z, surface.channels,
           surface.strides.x, surface.strides.z, surface.channel_stride };
}

template <typename T>
inline Surface2D<T> SliceY(const Surface3D<T> &surface, int y) {
  return { surface.data + surface.strides.y * y,
           surface.size.x, surface.size.z, surface.channels,
           surface.strides.x, surface.strides.z, surface.channel_stride };
}


template <typename Out, typename In>
inline void ResampleDepth(Surface3D<Out> out, Surface3D<In> in,
                         const int *in_slices, const float *slice_coeffs, int support) {
  if (in.strides.y == in.size.x * in.strides.x &&
      out.strides.y == out.size.x * out.strides.x) {
    // We're processing entire width of the image, so we can safely fuse width and height into
    // long rows and treat depth as height and reuse the code for 2D vertical pass.
    ResampleVert(FuseXY(out), FuseXY(in), in_slices, slice_coeffs, support);
  } else {
    // Cannot fuse - process row by row
    Surface2D<Out> out_xz = SliceY(out, 0);
    Surface2D<In> in_xz = SliceY(in, 0);
    for (int y = 0; y < out.size.y; y++) {
      ResampleVert(out_xz, in_xz, in_slices, slice_coeffs, support);
      out_xz.data += out.strides.y;
      if (y < in.size.y)
        in_xz.data += in.strides.y;
    }
  }
}

template <int spatial_ndim, typename Out, typename In>
inline void ResampleHorz(Surface<spatial_ndim, Out> out, Surface<spatial_ndim, In> in,
                         const int *in_columns, const float *col_coeffs, int support) {
  VALUE_SWITCH(out.channels, static_channels, (1, 2, 3, 4), (
    ResampleHorz_Channels<static_channels>(out, in, in_columns, col_coeffs, support);
  ), (  // NOLINT
    ResampleHorz_Channels<-1>(out, in, in_columns, col_coeffs, support);
  ));   // NOLINT
}

template <int spatial_ndim, typename Out, typename In>
inline void ResampleAxis(Surface<spatial_ndim, Out> out, Surface<spatial_ndim, In> in,
                         const int *in_indices, const float *coeffs, int support, int axis) {
  if (axis == 2)
    ResampleDepth(out, in, in_indices, coeffs, support);
  else if (axis == 1)
    ResampleVert(out, in, in_indices, coeffs, support);
  else if (axis == 0)
    ResampleHorz(out, in, in_indices, coeffs, support);
  else
    assert(!"Invalid axis index");
}

/**
 * @brief Resamples `in` using Nearest Neighbor interpolation and stores result in `out`
 * @param out - output surface
 * @param in - input surface
 * @param origin - input coordinates corresponding to output's origin
 * @param scale - step of input coordinates taken for each output pixel
 * @remarks The function clamps input coordinates to fit in range defined by `in` dimensions.
 *          Scales can be negative to achieve flipping.
 */
template <typename Out, typename In>
void ResampleNN(Surface2D<Out> out, Surface2D<const In> in,
                vec2 origin, vec2 scale) {
  assert(out.channels == in.channels);
  assert((in.channel_stride == 1 && out.channel_stride == 1) ||
         (in.channels == 1 && out.channels == 1));
  // assume HWC layout with contiguous pixels (not necessarily rows)
  assert(out.strides.x == out.channels);

  if (scale.x == 1) {
    // FAST PATH - not scaling along X axis - just copy with repeated boundaries
    int sx0 = std::floor(origin.x + 0.5f);
    int dx1 = sx0 + in.size.x;
    if (dx1 > out.size.x)
      dx1 = out.size.x;
    int dx0 = 0;
    if (sx0 < 0) {
      dx0 = std::min(-sx0, out.size.x);
    }

    float sy = origin.y + 0.5f * scale.y;
    for (int y = 0; y < out.size.y; y++, sy += scale.y) {
      int srcy = std::floor(sy);

      if (srcy < 0) srcy = 0;
      else if (srcy > in.size.y-1) srcy = in.size.y-1;

      Out *out_ch = &out(0, y, 0);

      int x;
      const In *first_px = &in(0, srcy, 0);
      for (x = 0; x < dx0; x++) {
        for (int c = 0; c < out.channels; c++)
          *out_ch++ = first_px[c];
      }

      const In *in_row = &in(sx0 + dx0, srcy, 0);
      for (int j = x * out.channels; j < dx1 * out.channels; j++)
          *out_ch++ = ConvertSat<Out>(*in_row++);

      x = dx1;
      const In *last_px = &in(in.size.x-1, srcy, 0);
      for (; x < out.size.x; x++) {
        for (int c = 0; c < out.channels; c++)
          *out_ch++ = last_px[c];
      }
    }
    return;
  }

  constexpr int max_span_width = 256;
  int col_offsets[max_span_width];  // NOLINT (kOnstant)

  for (int x0 = 0; x0 < out.size.x; x0 += max_span_width) {
    float sy = origin.y + 0.5f * scale.y;

    int span_width = x0 + max_span_width <= out.size.x ? max_span_width : out.size.x - x0;

    for (int j = 0; j < span_width; j++) {
      int x = x0 + j;
      float sx = origin.x + (x + 0.5f) * scale.x;
      int srcx = std::floor(sx);
      if (srcx < 0) srcx = 0;
      else if (srcx > in.size.x-1) srcx += in.size.x - 1;
      col_offsets[j] = srcx * in.strides.x;
    }

    for (int y = 0; y < out.size.y; y++, sy += scale.y) {
      int srcy = std::floor(sy);

      if (srcy < 0) srcy = 0;
      else if (srcy > in.size.y-1) srcy = in.size.y-1;

      const In *in_row = &in(0, srcy);
      Out *out_ch = &out(x0, y, 0);

      for (int j = 0; j < span_width; j++) {
        const In *in_ch = &in_row[col_offsets[j]];
        for (int c = 0; c < out.channels; c++) {
          *out_ch++ = ConvertSat<Out>(*in_ch++);
        }
      }
    }
  }
}


/**
 * @brief Resamples `in` using Nearest Neighbor interpolation and stores result in `out`
 * @param out - output surface
 * @param in - input surface
 * @param origin - input coordinates corresponding to output's origin
 * @param scale - step of input coordinates taken for each output pixel
 * @remarks The function clamps input coordinates to fit in range defined by `in` dimensions.
 *          Scales can be negative to achieve flipping.
 */
template <typename Out, typename In, int n>
void ResampleNN(Surface<n, Out> out, Surface<n, const In> in,
                vec<n> origin, vec<n> scale) {
  static_assert(n > 2, "This function only works with surfaces of dimensionality > 2");
  const float step = scale[n-1];
  float src = origin[n-1] + 0.5f * step;
  for (int i = 0; i < out.size[n-1]; i++, src += step) {
    int isrc = clamp<int>(std::floor(src), 0, in.size[n-1]-1);
    ResampleNN(out.slice(i), in.slice(isrc), sub<n-1>(origin), sub<n-1>(scale));
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CPU_H_
