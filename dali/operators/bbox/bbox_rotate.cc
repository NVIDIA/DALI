// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/bbox/bbox_rotate.h"
#include "dali/core/geom/vec.h"

#if defined(__x86_64__) || defined(_M_X64)
// x86_64 (amd64) specific SIMD code (SSE, AVX, etc.)
#include <xmmintrin.h>
#elif defined(__aarch64__)
// ARM64 (aarch64) specific SIMD code (NEON, etc.)
#include <arm_neon.h>
#endif

namespace dali {

DALI_SCHEMA(BBoxRotate)
    .DocStr(
        R"code(Transforms bounding boxes so that the boxes remain in the same place in the image after
the image is rotated. Boxes that land outside the image with ``keep_size=True`` will be removed from the
output if the fraction of the remaining box after cropping is less than ``remove_threshold``. If ``labels``
are passed as a second argument, they will also be removed synchronously.

By default, boxes are expanded when rotated to ensure the original box is fully enclosed. This can be 
controlled with ``mode`` which can either retain the original height and width with ``mode='fixed'``
or between the full expansion and original shape with ``mode='halfway'``.

.. warning::
  Boxes should be rotated first before the image as this op requires the original image shape to calculate
  the rotated boxes.

Example usage is below:

.. code-block:: python

  boxes, labels = fn.bbox_rotate(
    boxes, labels, angle=angle, input_shape=image.shape(), keep_size=keep_size
  )
  image = fn.rotate(image, angle=angle, keep_size=keep_size)

)code")
    .NumInput(1, 2)
    .InputDox(
        0, "bboxes", "2D TensorList of float",
        R"code(Coordinates of the bounding boxes that are represented as a [N,4] 2D tensor.)code")
    .InputDox(1, "labels", "1D or 2D TensorList of int",
              R"code(Class labels for the bounding boxes, [N] or [N,1]. These should be provided if 
``keep_size=True`` as boxes may be truncated, this ensures the corresponding class labels are also removed.
)code")
    .NumOutput(1)
    .AdditionalOutputsFn([](const OpSpec& spec) {
      return spec.NumRegularInput() - 1;  // +1 if labels are provided
    })
    .AddArg("angle", R"code(Rotation angle in degrees.)code", DALI_FLOAT, true)
    .AddArg("input_shape", R"code(Specifies the shape of the original image the boxes belong to.
    
The order of dimensions is determined by the layout that is provided in `shape_layout`.
)code",
            DALI_INT_VEC, true)
    .AddOptionalArg("shape_layout",
                    R"code(Determines the meaning of the dimensions provided in `input_shape`.

.. note::
  If unspecified, ``"HW"`` will be assumed, which is also compatible with ``"HWC"`` since we
  select H and W internally (index 0, 1).
)code",
                    dali::TensorLayout{"HW"}, false)
    .AddOptionalArg("bbox_layout",
                    R"code(Format of the bounding box input ``xyWH`` or ``xyXY``.)code",
                    dali::TensorLayout("xyXY"), false)
    .AddOptionalArg("bbox_normalized",
                    R"code(Input bounding boxes are in normalized [0,1] format)code", true, false)
    .AddOptionalArg(
        "keep_size",
        R"code(If true, the bounding box output coordinates will assume the image canvas size was also kept 
(see ``nvidia.dali.fn.rotate``).)code",
        false, false)
    .AddOptionalArg("mode",
                    R"code(Mode of the bounding box transformation. Possible values are:

 - ``expand``: expands the bounding box to definitely enclose the target, but may be larger than rotated target.
 - ``fixed``: retains the original size of the bounding box, but may be smaller than the rotated target. 
 - ``halfway``: halfway between the expanded size and the original size.

.. note::
  Modes other than ``expand`` are not recommended for datasets with high aspect ratio boxes and high rotation angles.
)code",
                    dali::string("expand"), false)
    .AddOptionalArg(
        "remove_threshold",
        R"code(Relative area remaining threshold for removing boxes that fall outside the image boundaries
after rotation. Should be between [0-1] where 0 means no removal even if box lies outside, 1 means removal 
if any part of the box is outside.)code",
        0.1f, false);

using Mode = BBoxRotate<CPUBackend>::Mode;

/**
 * @brief Expands the bounding boxes to all corners x1y1, x2y1, x1y2, x2y2 in flattened x_coords and
 *        y_coords.
 *
 * @note Not sure if I can use the 16-byte aligned version of SSE intrinsics, using unaligned for now.
 *
 * @tparam ltrb If true, the input boxes are in left-top-right-bottom format, otherwise in
 *         xywh format.
 * @param x_coords Output flattened x coordinates of the corners.
 * @param y_coords Output flattened y coordinates of the corners.
 * @param in_boxes Input bounding boxes.
 */
template <bool ltrb>
void ExpandToAllCorners(dali::span<float> x_coords, dali::span<float> y_coords,
                        dali::span<const vec<4>> in_boxes) {
  for (int i = 0; i < in_boxes.size(); ++i) {
    auto box = in_boxes[i];
    // Convert to x2y2
    if constexpr (!ltrb) {
      box.z += box.x;
      box.w += box.y;
    }
    const auto offset = 4 * i;
#if defined(__x86_64__) || defined(_M_X64)
    // Copy X coordinates
    const __m128 xvec = _mm_setr_ps(box.x, box.z, box.x, box.z);
    _mm_storeu_ps(&x_coords[offset], xvec);
    // Copy Y coordinates
    const __m128 yvec = _mm_setr_ps(box.y, box.y, box.w, box.w);
    _mm_storeu_ps(&y_coords[offset], yvec);
#elif defined(__aarch64__)
    // Copy X coordinates
    float32x4_t xvec = {box.x, box.z, box.x, box.z};
    vst1q_f32(&x_coords[offset], xvec);
    // Copy Y coordinates
    float32x4_t yvec = {box.y, box.y, box.w, box.w};
    vst1q_f32(&y_coords[offset], yvec);
#else
    x_coords[offset + 0] = box.x;
    x_coords[offset + 1] = box.z;
    x_coords[offset + 2] = box.x;
    x_coords[offset + 3] = box.z;

    y_coords[offset + 0] = box.y;
    y_coords[offset + 1] = box.y;
    y_coords[offset + 2] = box.w;
    y_coords[offset + 3] = box.w;
#endif
  }
}

/**
 * @brief Inplace rotates the corners of the bounding boxes by a given angle.
 *
 * @note y is pointing down in an image, the -sin is bottom left rather than top right in rotation
 * matrix
 *
 * @param x_coords Flattened x coordinates of the corners.
 * @param y_coords Flattened y coordinates of the corners.
 * @param angle Rotation angle in radians.
 */
void RotateCorners(dali::span<float> x_coords, dali::span<float> y_coords, float angle) {
  const float cos_a = std::cos(angle);
  const float sin_a = std::sin(angle);

  for (int i = 0; i < x_coords.size(); ++i) {
    const float x = x_coords[i];
    const float y = y_coords[i];
    x_coords[i] = x * cos_a + y * sin_a;
    y_coords[i] = -x * sin_a + y * cos_a;
  }
}

/**
 * @brief Converts corner coordinates to ltrb bounding boxes.
 * @param x_coords Flattened x coordinates of the corners.
 * @param y_coords Flattened y coordinates of the corners.
 * @param out_boxes Output ltrb bounding boxes.
 */
void CornersToBoxes(dali::span<vec<4>> out_boxes, dali::span<const float> x_coords,
                    dali::span<const float> y_coords) {
  for (int i = 0; i < out_boxes.size(); ++i) {
    const auto offset = 4 * i;
    auto [x1, x2] = std::minmax_element(&x_coords[offset], &x_coords[offset + 4]);
    auto [y1, y2] = std::minmax_element(&y_coords[offset], &y_coords[offset + 4]);
    out_boxes[i] = vec<4>(*x1, *y1, *x2, *y2);
  }
}

/**
 * @brief Applies expansion correction to the bounding box sizes by shrinking them by the amount the
 * rotation expanded them.
 *
 * @param boxes ltrb img-coord boxes to be corrected inplace.
 * @param old_whs the original width and height of each box in pixels.
 * @param halfway whether to only correct halfway, not the full expansion.
 */
void ApplyExpansionCorrectionToBoxSize(dali::span<vec<4>> inout_boxes,
                                       const std::vector<std::pair<float, float>>& old_whs,
                                       bool halfway) {
  for (int i = 0; i < inout_boxes.size(); ++i) {
    const float new_w = inout_boxes[i].z - inout_boxes[i].x;
    const float new_h = inout_boxes[i].w - inout_boxes[i].y;
    const auto& old_wh = old_whs[i];
    float diff_w, diff_h;

    // Check if aspect ratio has flipped e.g. 45 < angle < 135
    if ((old_wh.first < old_wh.second) ^ (new_w < new_h)) {
      diff_w = (new_w - old_wh.second) * 0.5f;
      diff_h = (new_h - old_wh.first) * 0.5f;
    } else {
      diff_w = (new_w - old_wh.first) * 0.5f;
      diff_h = (new_h - old_wh.second) * 0.5f;
    }

    if (halfway) {
      diff_w *= 0.5f;
      diff_h *= 0.5f;
    }
    inout_boxes[i].x += diff_w;
    inout_boxes[i].y += diff_h;
    inout_boxes[i].z -= diff_w;
    inout_boxes[i].w -= diff_h;
  }
}

/**
 * @brief Clips and removes boxes inplace that fall below a certain threshold of area change.
 * @param inout_boxes ltrb img-coord boxes which are modified inplace
 * @param remove_threshold threshold of area change to remove boxes
 * @param image_wh width and height of the image in pixels
 * @param in_labels optional input labels to filter synchronously with truncated boxes
 * @param out_labels option result of the filtered labels
 * @return the new number of valid boxes remaining
 */
int ClipAndRemoveBoxes(dali::span<vec<4>> inout_boxes, float remove_threshold,
                       std::pair<float, float> image_wh,
                       std::optional<ConstSampleView<CPUBackend>> in_labels,
                       std::optional<SampleView<CPUBackend>> out_labels) {
  DALI_ASSERT(in_labels.has_value() == out_labels.has_value());
  const bool handle_labels = in_labels.has_value();
  if (handle_labels) {
    DALI_ASSERT(in_labels->shape()[0] == inout_boxes.size());
    DALI_ASSERT(out_labels->shape()[0] == inout_boxes.size());
  }

  int out_idx = 0;
  for (int in_idx = 0; in_idx < inout_boxes.size(); ++in_idx) {
    auto box = inout_boxes[in_idx];
    const float old_area = (box.z - box.x) * (box.w - box.y);
    box.x = std::clamp(box.x, 0.0f, image_wh.first);
    box.y = std::clamp(box.y, 0.0f, image_wh.second);
    box.z = std::clamp(box.z, 0.0f, image_wh.first);
    box.w = std::clamp(box.w, 0.0f, image_wh.second);
    const float new_area = (box.z - box.x) * (box.w - box.y);
    const float fraction_remain = new_area / old_area;
    if (fraction_remain >= remove_threshold) {
      inout_boxes[out_idx] = box;
      if (handle_labels) {
        out_labels.value().mutable_data<int>()[out_idx] = in_labels.value().data<int>()[in_idx];
      }
      ++out_idx;
    }
  }
  return out_idx;
}

int RotateBoxesKernel(ConstSampleView<CPUBackend> in_box_tensor,
                      SampleView<CPUBackend> rotate_buffer, SampleView<CPUBackend> out_box_tensor,
                      float angle, std::optional<ConstSampleView<CPUBackend>> in_labels,
                      std::optional<SampleView<CPUBackend>> out_labels, bool ltrb, bool keep_size,
                      Mode mode, float remove_threshold, std::pair<float, float> image_wh,
                      bool bbox_norm) {
  const auto num_boxes = in_box_tensor.shape()[0];
  auto in_boxes = dali::span(reinterpret_cast<const vec<4>*>(in_box_tensor.raw_data()), num_boxes);
  auto x_coords = dali::span(rotate_buffer.mutable_data<float>(), 4 * num_boxes);
  auto y_coords = dali::span(x_coords.end(), 4 * num_boxes);
  if (ltrb) {
    ExpandToAllCorners<true>(x_coords, y_coords, in_boxes);
  } else {
    ExpandToAllCorners<false>(x_coords, y_coords, in_boxes);
  }

  // Center the coordinates on zero in image coordinates
  if (bbox_norm) {
    std::transform(x_coords.begin(), x_coords.end(), x_coords.begin(),
                   [w = image_wh.first](float elem) { return (elem - 0.5) * w; });
    std::transform(y_coords.begin(), y_coords.end(), y_coords.begin(),
                   [h = image_wh.second](float elem) { return (elem - 0.5) * h; });
  } else {
    std::transform(x_coords.begin(), x_coords.end(), x_coords.begin(),
                   [half_w = image_wh.first / 2](float elem) { return elem - half_w; });
    std::transform(y_coords.begin(), y_coords.end(), y_coords.begin(),
                   [half_h = image_wh.second / 2](float elem) { return elem - half_h; });
  }

  // Need to log old wh for expansion correction if needed
  std::optional<std::vector<std::pair<float, float>>> old_wh;
  if (mode != Mode::Expand) {
    old_wh->resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i) {  // x,y coords are in order x1x2x1x2 y1y1y2y2
      (*old_wh)[i] = {x_coords[4 * i + 1] - x_coords[4 * i], y_coords[4 * i + 2] - y_coords[4 * i]};
    }
  }

  // Apply rotation
  RotateCorners(x_coords, y_coords, angle);

  // Move coordinates back to [0,{w|h}], updating to new coordinate system
  if (!keep_size) {
    const auto old_w = image_wh.first;
    image_wh.first =
        std::abs(image_wh.first * std::cos(angle)) + std::abs(image_wh.second * std::sin(angle));
    image_wh.second =
        std::abs(image_wh.second * std::cos(angle)) + std::abs(old_w * std::sin(angle));
  }
  std::transform(x_coords.begin(), x_coords.end(), x_coords.begin(),
                 [half_w = image_wh.first / 2](float elem) { return elem + half_w; });
  std::transform(y_coords.begin(), y_coords.end(), y_coords.begin(),
                 [half_h = image_wh.second / 2](float elem) { return elem + half_h; });

  // Convert back to bounding boxes
  auto out_boxes =
      dali::span(reinterpret_cast<vec<4>*>(out_box_tensor.raw_mutable_data()), num_boxes);
  CornersToBoxes(out_boxes, x_coords, y_coords);

  // Apply correction to expansion factor if required
  if (mode != Mode::Expand) {
    ApplyExpansionCorrectionToBoxSize(out_boxes, *old_wh, mode == Mode::Halfway);
  }

  // Clip to image coordinates and remove boxes (and labels) with too large a reduction.
  const auto num_out_boxes =
      ClipAndRemoveBoxes(out_boxes, remove_threshold, image_wh, in_labels, out_labels);

  if (!ltrb || bbox_norm) {  // Convert back to xywh and/or normalized format if needed
    out_boxes = dali::span(out_boxes.data(), num_out_boxes);  // handle if some boxes culled
    std::for_each(out_boxes.begin(), out_boxes.end(), [image_wh, ltrb, bbox_norm](vec<4>& box) {
      if (!ltrb) {
        box.z -= box.x;
        box.w -= box.y;
      }
      if (bbox_norm) {
        box.x /= image_wh.first;
        box.y /= image_wh.second;
        box.z /= image_wh.first;
        box.w /= image_wh.second;
      }
    });
  }

  return num_out_boxes;
}

template <>
void BBoxRotate<CPUBackend>::RunImpl(Workspace& ws) {
  const auto& in_boxes = ws.Input<CPUBackend>(0);

  auto& tpool = ws.GetThreadPool();
  for (int sample_idx = 0; sample_idx < in_boxes.num_samples(); ++sample_idx) {
    tpool.AddWork([&, sample_idx](int) {
      std::optional<ConstSampleView<CPUBackend>> in_labels;
      std::optional<SampleView<CPUBackend>> out_labels;
      if (ws.NumInput() == 2) {
        in_labels = ws.Input<CPUBackend>(1)[sample_idx];
        out_labels = ws.Output<CPUBackend>(1)[sample_idx];
      }
      float angle;
      if (spec_.HasTensorArgument("angle")) {
        angle = ws.ArgumentInput("angle")[sample_idx].data<float>()[0];
      } else {
        angle = spec_.GetArgument<float>("angle");
      }

      std::pair<float, float> image_wh;
      if (spec_.HasTensorArgument("input_shape")) {
        auto shape_tensor = ws.ArgumentInput("input_shape")[sample_idx].data<std::int64_t>();
        image_wh.first = shape_tensor[shape_wh_index_.first];
        image_wh.second = shape_tensor[shape_wh_index_.second];
      } else {
        auto shape_tensor = this->spec_.GetRepeatedArgument<int>("input_shape");
        image_wh.first = shape_tensor[shape_wh_index_.first];
        image_wh.second = shape_tensor[shape_wh_index_.second];
      }

      angle *= M_PI / 180.0f;  // Convert to radians
      const auto num_out_boxes = RotateBoxesKernel(
          in_boxes[sample_idx], bbox_rotate_buffer_[sample_idx],
          ws.Output<CPUBackend>(0)[sample_idx], angle, in_labels, out_labels, use_ltrb_, keep_size_,
          mode_, remove_threshold_, image_wh, bbox_normalized_);
      ws.Output<CPUBackend>(0).ResizeSample(sample_idx, dali::TensorShape<2>(num_out_boxes, 4));
      if (out_labels.has_value()) {
        if (ws.GetInputShape(1)[sample_idx].size() == 2) {
          ws.Output<CPUBackend>(1).ResizeSample(sample_idx, dali::TensorShape<2>(num_out_boxes, 1));
        } else {
          ws.Output<CPUBackend>(1).ResizeSample(sample_idx, dali::TensorShape<1>(num_out_boxes));
        }
      }
    });
  }
  tpool.RunAll();
}

DALI_REGISTER_OPERATOR(BBoxRotate, BBoxRotate<CPUBackend>, CPU);

}  // namespace dali
