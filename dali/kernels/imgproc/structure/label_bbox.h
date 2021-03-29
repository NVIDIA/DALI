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
#include "dali/core/exec/engine.h"
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
 * First, 1D ranges are found for each foreground label present in `in`.
 * Then, the boxes that correspond to these labels are updated by expanding their
 * extent in the innermost dimension to match the range found and in the others to
 * contain `origin`.
 *
 * @param boxes       output span of bounding boxes
 * @param ranges      information about first and last occurrence of each label in current row
 * @param hits        information about which labels have been encountered in this row
 * @param in          current tensor slice
 * @param dim_mapping mapping from simplified to original (and bounding box) dimensions
 * @param background  label value interpreted as background (background has no bounding box)
 * @param origin      origin of current slice
 */
template <typename Coord, typename Label, int simplified_ndim, int ndim>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           span<Range> ranges,
                           span<uint32_t> hits,
                           TensorSlice<const Label, 1> in,
                           ivec<simplified_ndim> dim_mapping,
                           Label background,
                           i64vec<ndim> origin = {}) {
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
    lo[i] = origin[i];
    hi[i] = next<Coord>(origin[i]);  // one past
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
        lo[d] = ranges[i].lo + origin[d];
        hi[d] = next<Coord>(ranges[i].hi + origin[d]);  // one past the index found in this function
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
 * This function simply recurses down and adances origin as it traverses the input tensor.
 *
 * @param boxes       output span of bounding boxes
 * @param ranges      information about first and last occurrence of each label in current row
 * @param hits        information about which labels have been encountered in this row
 * @param in          current tensor slice
 * @param dim_mapping mapping from simplified to original (and bounding box) dimensions
 * @param background  label value interpreted as background (background has no bounding box)
 * @param origin      origin of current slice
 */
template <typename Coord, typename Label, int simplified_ndim, int ndim, int remaining_dims>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           span<Range> ranges,
                           span<uint32_t> hits,
                           TensorSlice<const Label, remaining_dims> in,
                           ivec<simplified_ndim> dim_mapping,
                           Label background,
                           i64vec<ndim> origin) {
  int64_t n = in.size[0];
  int d = dim_mapping[simplified_ndim - remaining_dims];
  for (int64_t i = 0; i < n; i++, origin[d]++) {
    GetLabelBoundingBoxes(boxes, ranges, hits, in.slice(i), dim_mapping, background, origin);
  }
}


/**
 * @brief Calculates a bounding box for each label in `in`.
 *
 * This function creates execution environment (ranges, hits) and invokes an overload
 * that uses this environment.
 *
 * @param boxes       output bounding boxes; labels whose box index is outside of valid range of
 *                    indices in `boxes` are ignored; box index for a label is calculated as:
 *                    background >= 0 && label > background ? label-1 : label
 * @param in          input tensor containing zero-based labels
 * @param background  the label value interpreted as a background; it doesn't have its corresponding
 *                    bounding box
 * @param origin      origin of current slice
 */
template <typename Coord, typename Label, int simplified_ndim, int ndim,
          int remaining_dims>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           TensorSlice<const Label, remaining_dims> in,
                           ivec<simplified_ndim> dim_mapping,
                           Label background,
                           i64vec<ndim> origin) {
  SmallVector<detail::Range, 16> ranges;
  SmallVector<uint32_t, 4> hits;
  ranges.resize(boxes.size());
  hits.resize(div_ceil(boxes.size(), 32));
  GetLabelBoundingBoxes(boxes, make_span(ranges), make_span(hits),
                        in, dim_mapping, background, origin);
}

/**
 * @brief Calculates a bounding box for each label in `in`
 *
 * This function checks whether the tensor slice is suitable for parallel execution and, if so,
 * tries to splits the input along the most suitable dimension and launches several subproblems.
 * Once finished, the results are combined and stored in the final output `boxes`.
 *
 * @param boxes       output bounding boxes; labels whose box index is outside of valid range of
 *                    indices in `boxes` are ignored; box index for a label is calculated as:
 *                    background >= 0 && label > background ? label-1 : label
 * @param in          input tensor containing zero-based labels
 * @param background  the label value interpreted as a background; it doesn't have its corresponding
 *                    bounding box
 * @param engine      thread-pool-like object
 */
template <typename Coord, typename Label, int simplified_ndim, int ndim,
          int remaining_dims, typename ExecutionEngine>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           TensorSlice<const Label, remaining_dims> in,
                           ivec<simplified_ndim> dim_mapping,
                           Label background,
                           ExecutionEngine &engine) {
  const int64_t min_parallel_elements = 1<<16;
  if (engine.NumThreads() == 1 || in.size[0] * in.stride[0] < min_parallel_elements) {
    GetLabelBoundingBoxes(boxes, in, dim_mapping, background, {});
    return;
  }
  SmallVector<Box<ndim, Coord>, 64> tmp_boxes;
  int max_d = -1;
  int64_t max_result = 0;
  for (int d = 0; d < simplified_ndim; d++) {
    int mul = (d == (simplified_ndim - 1) ? 1 : 2);  // it's less efficient to slice last dim
    int64_t result = in.size[d] * mul;
    if (result > max_result) {
      max_d = d;
      max_result = result;
    }
  }
  int num_chunks = std::min<int64_t>(in.size[max_d], engine.NumThreads());
  tmp_boxes.resize(boxes.size() * (num_chunks - 1));
  int64_t stride = in.stride[max_d];
  TensorSlice<const Label, remaining_dims> part = in;
  for (int i = 0; i < num_chunks; i++) {
    int64_t start = in.size[max_d] * i / num_chunks;
    int64_t end = in.size[max_d] * (i + 1) / num_chunks;
    span<Box<ndim, Coord>> part_boxes;
    if (i == 0)
      part_boxes = boxes;  // store 1st chunk directly
    else
      part_boxes = make_span(&tmp_boxes[(i - 1)*boxes.size()], boxes.size());
    part.size[max_d] = end - start;
    part.data = in.data + start * stride;
    engine.AddWork([=](int) {
      i64vec<ndim> origin = {};
      origin[dim_mapping[max_d]] = start;
      GetLabelBoundingBoxes(part_boxes, part, dim_mapping, background, origin);
    });
  }
  engine.RunAll();
  // BBox reduction
  for (int i = 1; i < num_chunks; i++) {
    auto part_boxes = make_span(&tmp_boxes[(i - 1)*boxes.size()], boxes.size());
    for (int j = 0; j < boxes.size(); j++) {
      if (part_boxes[j].empty())
        continue;
      if (boxes[j].empty()) {
        boxes[j] = part_boxes[j];
      } else {
        boxes[j].lo = min(boxes[j].lo, part_boxes[j].lo);
        boxes[j].hi = max(boxes[j].hi, part_boxes[j].hi);
      }
    }
  }
}

}  // namespace detail

/**
 * @brief Calculates a bounding box for each label in `in`
 *
 * @param boxes       output bounding boxes; labels whose box index is outside of valid range of
 *                    indices in `boxes` are ignored; box index for a label is calculated as:
 *                    background >= 0 && label > background ? label-1 : label
 * @param in          input tensor containing zero-based labels
 * @param background  the label value interpreted as a background; it doesn't have its corresponding
 *                    bounding box
 * @param engine      thread-pool-like object
 */
template <typename Coord, typename Label, int ndim, typename ExecutionEngine>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           const TensorView<StorageCPU, Label, ndim> &in,
                           std::remove_const_t<Label> background,
                           ExecutionEngine &engine) {
  std::fill(boxes.begin(), boxes.end(), Box<ndim, Coord>{});
  if (in.num_elements() == 0)
    return;
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
    detail::GetLabelBoundingBoxes<Coord, std::remove_const_t<Label>>(
        boxes, detail::FullTensorSlice(simplified_in), coord_vec,
        background, engine);
  ), (   // NOLINT
    throw std::invalid_argument("Only up to 6 non-degenerate dimensions are supported")
  ));  // NOLINT
}

template <typename Coord, typename Label, int ndim>
void GetLabelBoundingBoxes(span<Box<ndim, Coord>> boxes,
                           const TensorView<StorageCPU, Label, ndim> &in,
                           std::remove_const_t<Label> background) {
  SequentialExecutionEngine seq_engn;
  GetLabelBoundingBoxes(boxes, in, background, seq_engn);
}

}  // namespace label_bbox
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_STRUCTURE_LABEL_BBOX_H_
