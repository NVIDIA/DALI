// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_STRUCTURE_LABEL_BBOX_H_
#define DALI_KERNELS_IMGPROC_STRUCTURE_LABEL_BBOX_H_

#include <cmath>
#include <limits>
#include <type_traits>
#include "dali/core/geom/box.h"
#include "dali/core/span.h"
#include "dali/kernels/imgproc/structure/connected_components.h"

namespace dali {
namespace kernels {
namespace label_bbox {
namespace detail {

using connected_components::detail::TensorSlice;
using connected_components::detail::FullTensorSlice;

struct Range {
  int64_t lo = std::numeric_limits<int64_t>::max(), hi = -1;
};

template <typename T>
enable_if_t<std::is_integral<T>::value, T> next(T x) {
  return x + 1;
}

template <typename T>
enable_if_t<std::is_floating_point<T>::value, T> next(T x) {
  return std::nextafter(x, std::numeric_limits<T>::max());
}

/**
 * @brief Sets idx-th bit in hits and returns the previous value.
 */
inline bool hit(span<uint32_t> hits, unsigned idx) {
  uint32_t flag = (1u << (idx&31));
  uint32_t &h = hits[idx >> 5];
  bool ret = h & flag;
  h |= flag;
  return ret;
}

/**
 * @brief Calculates a bounding box for each label in `in`
 *
 * @param boxes       output span of bounding boxes
 * @param ranges      information about first and last occurrence of each label in current row
 * @param hits        information about which labels have been encountered in this row
 * @param in          current tensor slice
 * @param dim_mapping mapping from simplified to original (and bounding box) dimensions
 * @param background  label value interpreted as background (background has no bounding box)
 * @param pos         origin of current slice
 */
template <typename Coord, typename Label, int simplified_ndim, int ndim>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           span<Range> ranges,
                           span<uint32_t> hits,
                           TensorSlice<const Label, 1> in,
                           ivec<simplified_ndim> dim_mapping,
                           Label background,
                           i64vec<ndim> pos = {}) {
  for (auto &mask : hits)
    mask = 0u;  // mark all labels as not found in this row

  const unsigned nboxes = ranges.size();
  for (int64_t i = 0; i < in.size[0]; i++) {
    if (in.data[i] != background) {
      // We make a "hole" in the label indices for the background.
      int skip_bg = (background >= 0 && in.data[i] >= background);
      unsigned idx = static_cast<unsigned>(in.data[i]) - skip_bg;

      // deliberate use of unsigned overflow to detect negative labels as out-of-range
      if (idx < nboxes) {
        if (!hit(hits, idx)) {
          ranges[idx].lo = i;
        }
        ranges[idx].hi = i;
      }
    }
  }

  vec<ndim, Coord> lo, hi = 0;
  for (int i = 0; i < ndim; i++) {
    lo[i] = pos[i];
    hi[i] = next<Coord>(pos[i]);  // one past
  }

  const int d = dim_mapping[simplified_ndim-1];

  for (int word = 0; word < hits.size(); word++) {
    uint32_t mask = hits[word];
    int i = 32 * word;
    while (mask) {
      if ((mask & 0xffu) == 0) {  // skip 8 labels if not set
        mask >>= 8;
        i += 8;
        continue;
      }
      if (mask & 1) {  // label found? mark it
        lo[d] = ranges[i].lo;
        hi[d] = next<Coord>(ranges[i].hi);  // one past the index found in this function
        if (boxes[i].empty()) {
          // empty box - create a new one
          boxes[i] = { lo, hi };
        } else {
          // expand existing
          boxes[i].lo = min(boxes[i].lo, lo);
          boxes[i].hi = max(boxes[i].hi, hi);
        }
      }
      mask >>= 1;
      i++;  // skip one label
    }
  }
}

/**
 * @brief Calculates a bounding box for each label in `in`
 *
 * @param boxes       output span of bounding boxes
 * @param ranges      information about first and last occurrence of each label in current row
 * @param hits        information about which labels have been encountered in this row
 * @param in          current tensor slice
 * @param dim_mapping mapping from simplified to original (and bounding box) dimensions
 * @param background  label value interpreted as background (background has no bounding box)
 * @param pos         origin of current slice
 */
template <typename Coord, typename Label, int simplified_ndim, int ndim, int remaining_dims>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           span<Range> ranges,
                           span<uint32_t> hits,
                           TensorSlice<const Label, remaining_dims> in,
                           ivec<simplified_ndim> dim_mapping,
                           Label background,
                           i64vec<ndim> pos = {}) {
  int64_t n = in.size[0];
  int d = dim_mapping[simplified_ndim - remaining_dims];
  for (int64_t i = 0; i < n; i++) {
    pos[d] = i;
    GetLabelBoundingBoxes(boxes, ranges, hits, in.slice(i), dim_mapping, background, pos);
  }
}

}  // namespace detail

/**
 * @brief Calculates a bounding box for each lablel in `in`
 *
 * @param boxes     output bounding boxes; labels whose box index is outside of valid range of
 *                  indices in `boxes` are ignored; box index for a label is calculaed as:
 *                  label > background ? label-1 : label
 * @param in        input tensor containing zero-based labels
 * @param backgroun the label value interpreted as a background; it doesn't have its corresponding
 *                  bounding box
 */
template <typename Coord, typename Label, int ndim>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           const TensorView<StorageCPU, Label, ndim> &in,
                           std::remove_const_t<Label> background) {
  std::fill(boxes.begin(), boxes.end(), Box<ndim, Coord>{});
  if (in.num_elements() == 0)
    return;
  SmallVector<detail::Range, 16> ranges;
  SmallVector<uint32_t, 4> hits;
  ranges.resize(boxes.size());
  hits.resize(div_ceil(boxes.size(), 32));
  TensorShape<> simplified_shape;
  SmallVector<int, 6> dim_mapping;
  for (int d = 0; d < ndim; d++) {
    if (in.shape[d] == 1)
      continue;
    dim_mapping.push_back(d);
    simplified_shape.shape.push_back(in.shape[d]);
  }

  if (simplified_shape.empty()) {
    // add artificial dimension, so we can process a scalar as a 1D object
    simplified_shape.shape.push_back(1);
    dim_mapping.push_back(0);
  }

  VALUE_SWITCH(simplified_shape.size(), simplified_ndim, (1, 2, 3, 4, 5, 6), (
    ivec<simplified_ndim> coord_vec;
    for (int d = 0; d < simplified_ndim; d++)
      coord_vec[d] = dim_mapping[d];
    auto simplified_in = make_tensor_cpu<simplified_ndim, const Label>(
      in.data, simplified_shape.to_static<simplified_ndim>());
    GetLabelBoundingBoxes<Coord, std::remove_const_t<Label>>(
        boxes, make_span(ranges), make_span(hits),
        detail::FullTensorSlice(simplified_in), coord_vec,
        background);
  ), (   // NOLINT
    throw std::invalid_argument("Only up to 6 non-degenerate dimensions are supported")
  ));  // NOLINT
}

}  // namespace label_bbox
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_STRUCTURE_LABEL_BBOX_H_
