// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
// limitations under the License.c++ copyswa

#ifndef DALI_PIPELINE_UTIL_BOUNDING_BOX_UTILS_H_
#define DALI_PIPELINE_UTIL_BOUNDING_BOX_UTILS_H_

#include <limits>
#include "dali/core/geom/box.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/math_util.h"

namespace dali {

template <int ndim>
void AxisFlip(Box<ndim, float>& box, int axis) {
  assert(axis >= 0 && axis < ndim);
  float lo = box.lo[axis], hi = box.hi[axis];
  box.lo[axis] = 1.0f - hi;
  box.hi[axis] = 1.0f - lo;
}

template <int ndim>
void HorizontalFlip(Box<ndim, float>& box) {
  static_assert(ndim >= 1, "Unexpected number of dimensions");
  AxisFlip(box, 0);
}

template <int ndim>
void VerticalFlip(Box<ndim, float>& box) {
  static_assert(ndim >= 2, "Unexpected number of dimensions");
  AxisFlip(box, 1);
}

template <int ndim>
void DepthwiseFlip(Box<ndim, float>& box) {
  static_assert(ndim >= 3, "Unexpected number of dimensions");
  AxisFlip(box, 2);
}

template <int ndim>
constexpr TensorLayout DefaultBBoxLayout() {
  assert(ndim == 2 || ndim == 3);
  if (ndim == 3)
    return TensorLayout{"xyzXYZ"};
  else
    return TensorLayout{"xyXY"};
}

template <int ndim>
constexpr TensorLayout DefaultBBoxAnchorAndShapeLayout() {
  assert(ndim == 2 || ndim == 3);
  if (ndim == 3)
    return TensorLayout{"xyzWHD"};
  else
    return TensorLayout{"xyWH"};
}

template <int ndim>
float AspectRatio(const Box<ndim, float>& box, int dim0, int dim1) {
  assert(dim0 >= 0 && dim0 < ndim);
  assert(dim1 >= 0 && dim1 < ndim);
  auto extent = box.extent();
  auto extent0 = extent[dim0];
  auto extent1 = extent[dim1];
  return extent0 / extent1;
}

template <int ndim>
Box<ndim, float> Uniform(float min, float max) {
  return Box<ndim, float>(vec<ndim, float>(min), vec<ndim, float>(max));
}

/**
 * @brief Permutes coordinates according to an input and output layout
 */
template <typename Coords>
void PermuteCoords(Coords& coords,
                   TensorLayout orig_layout,
                   TensorLayout new_layout) {
  if (orig_layout == new_layout)
    return;
  assert(orig_layout.is_permutation_of(new_layout));
  assert(coords.size() == orig_layout.size());
  auto perm = GetDimIndices(orig_layout, new_layout);
  auto out = coords;
  for (int d = 0; d < static_cast<int>(coords.size()); d++)
    out[d] = coords[perm[d]];
  std::swap(coords, out);
}

template <int ndim>
void CheckBBoxLayout(TensorLayout layout) {
  auto default_layout_start_end   = DefaultBBoxLayout<ndim>();
  auto default_layout_start_shape = DefaultBBoxAnchorAndShapeLayout<ndim>();
  bool is_start_and_end = layout.is_permutation_of(default_layout_start_end);
  bool is_start_and_shape = layout.is_permutation_of(default_layout_start_shape);
  DALI_ENFORCE(is_start_and_end || is_start_and_shape,
    make_string("`", layout, "` should be a permutation of `", default_layout_start_end,
                "` or `", default_layout_start_shape, "`"));
}

/**
 * @brief Read bounding box coordinates from a 1D span of floats, interpreting each coordinate
 * according to the provided layout. The function will enforce that all the boxes read are within
 * the provided `limit` bounds
 * @remarks Dimension names in the layout can be low (or start) anchors: "xyz", high (or end)
 * anchors: "XYZ" or extent "WHD". For example, a layout "xyXY" implies that the bounding box
 * coordinates are following the order x_start, y_start, x_end, y_end
 */
template <int ndim>
void ReadBoxes(span<Box<ndim, float>> boxes, span<const float> coords,
               TensorLayout layout = {},
               const Box<ndim, float>& limits = Uniform<ndim>(0.0f, 1.0f)) {
  static constexpr int box_size = ndim * 2;
  assert(coords.size() % box_size == 0);
  assert(coords.size() / box_size == boxes.size());
  assert(layout.empty() || layout.size() == box_size);
  int nboxes = boxes.size();

  if (layout.empty())
    layout = DefaultBBoxLayout<ndim>();

  auto default_layout_start_end   = DefaultBBoxLayout<ndim>();
  auto default_layout_start_shape = DefaultBBoxAnchorAndShapeLayout<ndim>();
  bool is_start_and_end = layout.is_permutation_of(default_layout_start_end);
  bool is_start_and_shape = layout.is_permutation_of(default_layout_start_shape);
  DALI_ENFORCE(is_start_and_end || is_start_and_shape,
    make_string("`", layout, "` should be a permutation of `", default_layout_start_end,
                "` or `", default_layout_start_shape, "`"));
  TensorLayout ordered_layout =
      is_start_and_shape ? default_layout_start_shape : default_layout_start_end;

  std::array<float, box_size> tmp;
  for (int i = 0; i < nboxes; i++) {
    auto &box = boxes[i];
    const float* in = coords.data() + i * box_size;
    for (int d = 0; d < box_size; d++)
      tmp[d] = in[d];
    PermuteCoords(tmp, layout, ordered_layout);
    for (int d = 0; d < ndim; d++) {
      box.lo[d] = tmp[d];
      box.hi[d] = tmp[ndim + d];
    }
    if (is_start_and_shape) {
      for (int d = 0; d < ndim; d++) {
        box.hi[d] += box.lo[d];
      }
    }
  }

  if (!limits.empty()) {
    for (int i = 0; i < nboxes; i++) {
      DALI_ENFORCE(limits.contains(boxes[i]),
        make_string("box ", boxes[i], " is out of bounds ", limits));
    }
  }
}

/**
 * @brief Read one bounding box
 * @remarks see ReadBoxes
 */
template <int ndim>
void ReadBox(Box<ndim, float>& box,
             span<const float> coords,
             TensorLayout layout = {},
             const Box<ndim, float>& limits = Uniform<ndim>(0.0f, 1.0f)) {
  ReadBoxes<ndim>({&box, 1}, coords, layout, limits);
}

/**
 * @brief Write bounding box coordinates to a 1D span of floats, outputing the coordinates in the
 * order specied by the provided layout
 * @remarks Dimension names in the layout can be low (or start) anchors: "xyz", high (or end)
 * anchors: "XYZ" or extent "WHD". For example, a layout "xyXY" implies that the bounding box
 * coordinates are following the order x_start, y_start, x_end, y_end, while a layout "xyWD" means
 * that the coordinates are following the order x_start, y_start, width, height
 */
template <int ndim>
void WriteBoxes(span<float> coords,
                span<const Box<ndim, float>> boxes,
                TensorLayout layout = {}) {
  static constexpr int box_size = ndim * 2;
  assert(coords.size() % box_size == 0);
  assert(coords.size() / box_size == boxes.size());
  assert(layout.size() == box_size);
  int nboxes = boxes.size();
  auto default_layout_start_end   = DefaultBBoxLayout<ndim>();
  auto default_layout_start_shape = DefaultBBoxAnchorAndShapeLayout<ndim>();

  if (layout.empty())
    layout = DefaultBBoxLayout<ndim>();

  bool is_start_and_end = layout.is_permutation_of(default_layout_start_end);
  bool is_start_and_shape = layout.is_permutation_of(default_layout_start_shape);
  DALI_ENFORCE(is_start_and_end || is_start_and_shape,
    make_string("`", layout, "` should be a permutation of `", default_layout_start_end,
                "` or `", default_layout_start_shape, "`"));
  TensorLayout ordered_layout =
      is_start_and_shape ? default_layout_start_shape : default_layout_start_end;

  for (int i = 0; i < nboxes; i++) {
    const auto &box = boxes[i];
    std::array<float, box_size> tmp;
    for (int d = 0; d < ndim; d++) {
      tmp[d]        = box.lo[d];
      tmp[ndim + d] = box.hi[d];
    }

    if (is_start_and_shape) {
      for (int d = 0; d < ndim; d++) {
        tmp[ndim + d] -= box.lo[d];
      }
    }
    PermuteCoords(tmp, ordered_layout, layout);

    float *out = coords.data() + box_size * i;
    for (int d = 0; d < box_size; d++) {
      out[d] = tmp[d];
    }
  }
}

/**
 * @brief Write one bounding box
 * @remarks see WriteBoxes
 */
template <int ndim>
void WriteBox(span<float> coords,
              const Box<ndim, float>& box,
              TensorLayout layout = {}) {
  WriteBoxes<ndim>(coords, {&box, 1}, layout);
}

/**
 * @brief Remaps relative bounding box coordinates to the coordinate space of a subwindow
 */
template <int ndim>
Box<ndim, float> RemapBox(const Box<ndim, float> &box, const Box<ndim, float> &crop) {
  Box<ndim, float> mapped_box = box;
  auto rel_extent = crop.extent();
  auto start = (max(crop.lo, box.lo) - crop.lo) / rel_extent;
  auto end = (min(crop.hi, box.hi) - crop.lo) / rel_extent;
  mapped_box.lo = clamp(start, vec<ndim, float>{0.0f}, vec<ndim, float>{1.0f});
  mapped_box.hi = clamp(end, vec<ndim, float>{0.0f}, vec<ndim, float>{1.0f});
  return mapped_box;
}

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BOUNDING_BOX_UTILS_H_
