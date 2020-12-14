// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/common/simd.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {

struct FilterWindow;
struct ResamplingFilter;

DLL_PUBLIC
void InitializeResamplingFilter(int32_t *out_indices, float *out_coeffs, int out_size,
                                float srcx0, float scale, const ResamplingFilter &filter);

/**
 * @brief Calculates a single pixel for horizontal resampling
 * @param out        - output row
 * @param in         - input row
 * @param x          - output column index
 * @param w          - input width
 * @param in_columns - precomputed leftmost indices of kernel footprints in input
 *                     for each output column
 * @param coeffs     - per-column resampling kernels - each kernel starts at index
 *                     x * support
 * @param support    - size of a resampling kernel
 * @param dynamic_channels - number of channels, if not known at compile time
 * @tparam static_channels - number of channels, if known at compile time, or < 0 if not known
 */
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

template <typename Out, typename In>
struct SIMD_vert_resample_impl {
#ifdef __SSE2__
  static constexpr int kVecSize = 16;
  static constexpr int load_lanes = kVecSize / sizeof(In);
  static constexpr int store_lanes = kVecSize / sizeof(Out);
  static constexpr int kNumLanes = load_lanes > store_lanes ? load_lanes : store_lanes;
  static constexpr int kNumVecs = kNumLanes * sizeof(float) / kVecSize;

  using vec_pack = simd::multivec<kNumVecs>;
#endif

  static void run(Out *out, const In **rows, const float *kernel, int support,
                   int begin_col, int end_col) {
    int i = begin_col;
#ifdef __SSE2__
    for (; i + kNumLanes <= end_col; i += kNumLanes) {
      vec_pack vtmp = vec_pack::zero();

      for (int k = 0; k < support; k++) {
        vec_pack vin = vec_pack::load(rows[k] + i);
        __m128 coeff = _mm_set1_ps(kernel[k]);
        for (int v = 0; v < kNumVecs; v++)
          vtmp.v[v] = _mm_add_ps(vtmp.v[v], _mm_mul_ps(coeff, vin.v[v]));
      }
      store(out + i, vtmp);
    }
#endif
    for (; i < end_col; i++) {
      float tmp = 0;
      for (int k = 0; k < support; k++)
        tmp += rows[k][i] * kernel[k];
      out[i] = ConvertSat<Out>(tmp);
    }
  }
};


template <typename Out, typename In>
struct SIMD_horz_resample_impl {
#ifdef __SSE2__
  static constexpr int kVecSize = 16;
  static constexpr int kNumLanes = kVecSize / sizeof(Out);
  static constexpr int kNumVecs = kNumLanes * sizeof(float) / kVecSize;

  using vec_pack = simd::multivec<kNumVecs>;
#endif

  template <int static_channels, bool clamp_left, bool clamp_right>
  inline int run(Out *out, const In *in, int ox0, int ox1, int w,
                 const int32_t *in_columns,
                 const float *coeffs, int support,
                 int dynamic_channels) {
    const int channels = static_channels < 0 ? dynamic_channels : static_channels;

    int x = ox0;
#ifdef __SSE2__
    float tmpin[kNumLanes];
    for (; x + kNumLanes <= ox1; x += kNumLanes) {
      Out tmp_out[kNumLanes];

      if (static_channels < 0) {
        // we don't know how many channels we have at compile time - inner loop over filter
        for (int c = 0; c < channels; c++) {
          vec_pack vout;
          vout = vec_pack::zero();

          for (int k = 0; k < support; k++) {
            float tmp_coeffs[kNumLanes];
            for (int l = 0; l < kNumLanes; l++)
              tmp_coeffs[l] = coeffs[(x + l) * support + k];  // interleave per-column coefficients
            vec_pack vcoeffs = vec_pack::load(tmp_coeffs);

            for (int l = 0; l < kNumLanes; l++) {
              int srcx = in_columns[x + l] + k;
              if (clamp_left) if (srcx < 0) srcx = 0;
              if (clamp_right) if (srcx > w-1) srcx = w-1;
              tmpin[l] = in[srcx * channels + c];
            }
            vec_pack vin = vec_pack::load(tmpin);

            for (int v = 0; v < kNumVecs; v++)
              vout.v[v] = _mm_add_ps(vout.v[v], _mm_mul_ps(vcoeffs.v[v], vin.v[v]));
          }
          store(tmp_out, vout);
          for (int l = 0; l < kNumLanes; l++)
            out[channels * (x + l) + c] = tmp_out[l];  // interleave
        }
      } else {
        // we know how many channels we have at compile time - inner loop over channels
        static constexpr int kNCh = static_channels > 0 ? static_channels : 1;
        vec_pack vout[kNCh];
        Out tmp_out[kNCh][kNumLanes];
        float tmp_in[kNCh][kNumLanes];

        for (int c = 0; c < channels; c++)
          vout[c] = vec_pack::zero();

        for (int k = 0; k < support; k++) {
          float tmp_coeffs[kNumLanes];
          for (int l = 0; l < kNumLanes; l++)
            tmp_coeffs[l] = coeffs[(x + l) * support + k];  // interleave per-column coefficients
          vec_pack vcoeffs = vec_pack::load(tmp_coeffs);


          for (int l = 0; l < kNumLanes; l++) {
            int srcx = in_columns[x + l] + k;
            if (clamp_left) if (srcx < 0) srcx = 0;
            if (clamp_right) if (srcx > w-1) srcx = w-1;
            for (int c = 0; c < channels; c++) {
              tmp_in[c][l] = in[srcx * channels + c];
            }
          }

          for (int c = 0; c < channels; c++) {
            vec_pack vin = vec_pack::load(tmp_in[c]);
            for (int v = 0; v < kNumVecs; v++)
              vout[c].v[v] = _mm_add_ps(vout[c].v[v], _mm_mul_ps(vcoeffs.v[v], vin.v[v]));
          }
        }

        for (int c = 0; c < channels; c++)
          store(tmp_out[c], vout[c]);

        for (int l = 0; l < kNumLanes; l++)
          for (int c = 0; c < channels; c++)
            out[channels * (x + l) + c] = tmp_out[c][l];  // interleave channels
      }
    }
#endif

    for (; x < ox1; x++) {
      ResampleCol<static_channels, clamp_left, clamp_right, Out, In>(
          out, in, x, w, in_columns, coeffs, support, dynamic_channels);
    }

    return x;
  }
};



/**
 * @brief Calcualtes the indices of first and last _output_ columns that does not need
 *        _input_ coordinate clamping
 *
 * @param first_regular_col  [out] leftmost _output_ column which can be calculated without
 *                                 clamping _input_ coordinates
 * @param last_regular_col   [out] rightmost _output_ column which can be calculated without
 *                                 clamping _input_ coordinates
 * @param out_width              - width of the output surface
 * @param in_width               - width of the input surface
 * @param in_col_idxs            - precomputed leftmost indices of kernel footprints in input
 *                                 for each output column
 * @param support                - size of the resampling kernel
 *
 * @return true, if the resampling is flipped (right to left) or false otherwise
 */
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

/**
 * @brief Calcualtes the indices of first and last _output_ columns that does not need
 *        _input_ coordinate clamping
 *
 * @param out               - output row
 * @param out_width         - width of the output surface
 * @param in                - input row
 * @param in_width          - width of the input surface
 * @param in_columns        - precomputed leftmost indices of kernel footprints in input
 *                            for each output column
 * @param coeffs            - per-column resampling kernels - each kernel starts at index
 *                            x * support
 * @param first_regular_col - index of the first _output_ column which can be calculated without
 *                            applying boundary conditions in _input_
 * @param first_regular_col - index of the last _output_ column which can be calculated without
 *                            applying boundary conditions in _input_
 * @param flipped           - true, if values in in_columns decrease
 */
template <int static_channels = -1, typename Out, typename In>
void ResamplHorzRow(Out *out_row, int out_width, const In *in_row, int in_width, int channels,
                    const int *in_columns, const float *coeffs, int support,
                    int first_regular_col, int last_regular_col, bool flipped) {
  int x = 0;
  // if last_regular_col < first_regular_col, then we can only use one-sided clamp
  // up to last_regular_col-1
  int max_one_sided_clamp = std::min(first_regular_col, last_regular_col+1);

  SIMD_horz_resample_impl<Out, In> impl;
  if (flipped) {
    x = impl.template run<static_channels, false, true>(
        out_row, in_row, x, max_one_sided_clamp, in_width, in_columns, coeffs, support, channels);
  } else {
    x = impl.template run<static_channels, true, false>(
        out_row, in_row, x, max_one_sided_clamp, in_width, in_columns, coeffs, support, channels);
  }

  x = impl.template run<static_channels, true, true>(
        out_row, in_row, x, first_regular_col, in_width, in_columns, coeffs, support, channels);
  x = impl.template run<static_channels, false, false>(
        out_row, in_row, x, last_regular_col+1, in_width, in_columns, coeffs, support, channels);

  if (flipped) {
    impl.template run<static_channels, true, false>(
        out_row, in_row, x, out_width, in_width, in_columns, coeffs, support, channels);
  } else {
    impl.template run<static_channels, false, true>(
        out_row, in_row, x, out_width, in_width, in_columns, coeffs, support, channels);
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
  constexpr int tile = 256;

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
      SIMD_vert_resample_impl<Out, In> res;
      res.run(out_row, in_row_ptrs, &row_coeffs[y * support], support, x0, x0 + tile_w);
    }
  }
}

/**
 * @brief Resamples a surface depthwise
 *
 * @param out          - output surface
 * @param in           - input surface
 * @param in_rows      - precomputed topmost indices of kernel footprints in input
 *                       for each output row
 * @param row_coeffs   - per-row resampling kernels - each kernel starts at index
 *                       y * support
 * @param support      - size of the resampling kernel
 */
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


/**
 * @brief Resamples a surface depthwise
 *
 * @param out          - output surface
 * @param in           - input surface
 * @param in_slices    - precomputed starting z indices of kernel footprints in input
 *                       for each output slice
 * @param slice_coeffs - per-slice resampling kernels - each kernel starts at index
 *                       z * support
 * @param support      - size of the resampling kernel
 */
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


/**
 * @brief Resamples a surface horizontally
 *
 * @param out         - output surface
 * @param in          - input surface
 * @param in_columns  - precomputed leftmost indices of kernel footprints in input
 *                      for each output column
 * @param coeffs      - per-column resampling kernels - each kernel starts at index
 *                      x * support
 * @param support     - size of the resampling kernel
 */
template <int spatial_ndim, typename Out, typename In>
inline void ResampleHorz(Surface<spatial_ndim, Out> out, Surface<spatial_ndim, In> in,
                         const int *in_columns, const float *col_coeffs, int support) {
  VALUE_SWITCH(out.channels, static_channels, (1, 2, 3, 4), (
    ResampleHorz_Channels<static_channels>(out, in, in_columns, col_coeffs, support);
  ), (  // NOLINT
    ResampleHorz_Channels<-1>(out, in, in_columns, col_coeffs, support);
  ));   // NOLINT
}

/**
 * @brief Resamples an axis
 *
 * @param out         - output surface
 * @param in          - input surface
 * @param in_indices  - precomputed starting indices of kernel footprints in input
 *                      for each output slice/row/column (depending on axis)
 * @param coeffs      - per-index resampling kernels - each kernel starts at index
 *                      idx * support, wherei idx is output index in given axis
 * @param support     - size of the resampling kernel
 * @param axis        - selects resampled axis in vec order
 *                      0 - horizontal (X), 1 - vertical (Y), 2 - depthwise (Z)
 */
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
