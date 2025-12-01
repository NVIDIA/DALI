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
#include "dali/operators/image/remap/rotate_params.h"

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
    .AddOptionalArg<float>(
        "size",
        R"code(The output canvas size optionally specified in the associated `fn.rotate`.)code",
        dali::vector<float>(), true)
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
 * @tparam ltrb If true, the input boxes are in left-top-right-bottom format, otherwise in
 *         xywh format.
 * @param x_coords Output flattened x coordinates of the corners.
 * @param y_coords Output flattened y coordinates of the corners.
 * @param in_boxes Input bounding boxes.
 */
template <bool ltrb>
void ExpandToAllCorners(dali::span<float> x_coords, dali::span<float> y_coords,
                        dali::span<const vec4> in_boxes) {
  for (int i = 0; i < in_boxes.size(); ++i) {
    auto box = in_boxes[i];
    // Convert to x2y2
    if constexpr (!ltrb) {
      box.z += box.x;
      box.w += box.y;
    }
    const auto offset = 4 * i;

    const vec4 x = {box.x, box.z, box.x, box.z};
#pragma omp simd simdlen(4)
    for (int i = 0; i < 4; ++i) {
      x_coords[offset + i] = x[i];
    }

    const vec4 y = {box.y, box.y, box.w, box.w};
#pragma omp simd simdlen(4)
    for (int i = 0; i < 4; ++i) {
      y_coords[offset + i] = y[i];
    }
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
void CornersToBoxes(dali::span<vec4> out_boxes, dali::span<const float> x_coords,
                    dali::span<const float> y_coords) {
  for (int i = 0; i < out_boxes.size(); ++i) {
    const auto offset = 4 * i;
    auto [x1, x2] = std::minmax_element(&x_coords[offset], &x_coords[offset + 4]);
    auto [y1, y2] = std::minmax_element(&y_coords[offset], &y_coords[offset + 4]);
    out_boxes[i] = vec4(*x1, *y1, *x2, *y2);
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
void ApplyExpansionCorrectionToBoxSize(dali::span<vec4> inout_boxes,
                                       const std::vector<vec2>& old_whs, bool halfway) {
  for (int i = 0; i < inout_boxes.size(); ++i) {
    const float new_w = inout_boxes[i].z - inout_boxes[i].x;
    const float new_h = inout_boxes[i].w - inout_boxes[i].y;
    const auto& old_wh = old_whs[i];
    float diff_w, diff_h;

    // Check if aspect ratio has flipped e.g. 45 < angle < 135
    if ((old_wh.x < old_wh.y) ^ (new_w < new_h)) {
      diff_w = (new_w - old_wh.y) * 0.5f;
      diff_h = (new_h - old_wh.x) * 0.5f;
    } else {
      diff_w = (new_w - old_wh.x) * 0.5f;
      diff_h = (new_h - old_wh.y) * 0.5f;
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
 * @return the indicies of the boxes to keep
 */
std::vector<int> ClipAndRemoveBoxes(dali::span<vec4> inout_boxes, float remove_threshold,
                                    vec2 image_wh) {
  std::vector<int> keep_indices{};
  keep_indices.reserve(inout_boxes.size());

  int out_idx = 0;
  for (int in_idx = 0; in_idx < inout_boxes.size(); ++in_idx) {
    auto box = inout_boxes[in_idx];
    const float old_area = (box.z - box.x) * (box.w - box.y);
    box.x = std::clamp(box.x, 0.0f, image_wh.x);
    box.y = std::clamp(box.y, 0.0f, image_wh.y);
    box.z = std::clamp(box.z, 0.0f, image_wh.x);
    box.w = std::clamp(box.w, 0.0f, image_wh.y);
    const float new_area = (box.z - box.x) * (box.w - box.y);
    const float fraction_remain = new_area / old_area;
    if (fraction_remain >= remove_threshold) {
      inout_boxes[out_idx] = box;
      keep_indices.push_back(in_idx);
      ++out_idx;
    }
  }
  return keep_indices;
}

std::vector<int> RotateBoxesKernel(ConstSampleView<CPUBackend> in_box_tensor,
                                   SampleView<CPUBackend> scratch_buffer, float angle, bool ltrb,
                                   Mode mode, float remove_threshold, vec2 in_wh, vec2 out_wh,
                                   bool bbox_norm) {
  const auto num_boxes = in_box_tensor.shape()[0];
  auto in_boxes = dali::span(reinterpret_cast<const vec4*>(in_box_tensor.raw_data()), num_boxes);
  auto x_coords = dali::span(scratch_buffer.mutable_data<float>(), 4 * num_boxes);
  auto y_coords = dali::span(x_coords.end(), 4 * num_boxes);
  if (ltrb) {
    ExpandToAllCorners<true>(x_coords, y_coords, in_boxes);
  } else {
    ExpandToAllCorners<false>(x_coords, y_coords, in_boxes);
  }

  // Center the coordinates on zero in image coordinates
  if (bbox_norm) {
    std::transform(x_coords.begin(), x_coords.end(), x_coords.begin(),
                   [w = in_wh.x](float elem) { return (elem - 0.5f) * w; });
    std::transform(y_coords.begin(), y_coords.end(), y_coords.begin(),
                   [h = in_wh.y](float elem) { return (elem - 0.5f) * h; });
  } else {
    std::transform(x_coords.begin(), x_coords.end(), x_coords.begin(),
                   [half_w = in_wh.x / 2](float elem) { return elem - half_w; });
    std::transform(y_coords.begin(), y_coords.end(), y_coords.begin(),
                   [half_h = in_wh.y / 2](float elem) { return elem - half_h; });
  }

  // Need to log old wh for expansion correction if needed
  std::vector<vec2> old_wh;
  if (mode != Mode::Expand) {
    old_wh.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i) {  // x,y coords are in order x1x2x1x2 y1y1y2y2
      old_wh[i] = {x_coords[4 * i + 1] - x_coords[4 * i], y_coords[4 * i + 2] - y_coords[4 * i]};
    }
  }

  // Apply rotation
  RotateCorners(x_coords, y_coords, angle);

  // Move coordinates back to [0,{w|h}], updating to new coordinate system
  std::transform(x_coords.begin(), x_coords.end(), x_coords.begin(),
                 [half_w = out_wh.x / 2](float elem) { return elem + half_w; });
  std::transform(y_coords.begin(), y_coords.end(), y_coords.begin(),
                 [half_h = out_wh.y / 2](float elem) { return elem + half_h; });

  // Convert back to bounding boxes, re-using scratch_buffer because the number of x corner
  // values per box (4) is the same as the box coordinates so basically the x corner coordinate
  // memory will be progressively overwritten by the final output boxes.
  auto out_boxes =
      dali::span(reinterpret_cast<vec4*>(scratch_buffer.raw_mutable_data()), num_boxes);
  CornersToBoxes(out_boxes, x_coords, y_coords);

  // Apply correction to expansion factor if required
  if (mode != Mode::Expand) {
    ApplyExpansionCorrectionToBoxSize(out_boxes, old_wh, mode == Mode::Halfway);
  }

  // Clip to image coordinates and remove boxes (and labels) with too large a reduction.
  const auto out_indices = ClipAndRemoveBoxes(out_boxes, remove_threshold, out_wh);

  if (!ltrb || bbox_norm) {  // Convert back to xywh and/or normalized format if needed
    out_boxes = dali::span(out_boxes.data(), out_indices.size());  // handle if some boxes culled

    // Recip of whwh to multiply with box for normalization
    const vec4 box_norm_vec = {1.f / out_wh.x, 1.f / out_wh.y, 1.f / out_wh.x, 1.f / out_wh.y};

    std::for_each(out_boxes.begin(), out_boxes.end(), [ltrb, bbox_norm, box_norm_vec](vec4& box) {
      if (!ltrb) {
        box.z -= box.x;
        box.w -= box.y;
      }
      if (bbox_norm) {
#pragma omp simd simdlen(4)
        for (int i = 0; i < 4; ++i) {
          box[i] *= box_norm_vec[i];
        }
      }
    });
  }

  return out_indices;
}

template <>
void BBoxRotate<CPUBackend>::RunImpl(Workspace& ws) {
  const auto& in_boxes = ws.Input<CPUBackend>(0);
  std::vector<std::vector<int>> kept_box_indices(in_boxes.num_samples());

  auto& tpool = ws.GetThreadPool();
  for (int sample_idx = 0; sample_idx < in_boxes.num_samples(); ++sample_idx) {
    tpool.AddWork([&, sample_idx](int) {
      float angle;
      if (spec_.HasTensorArgument("angle")) {
        const auto angle_tensor = ws.ArgumentInput("angle")[sample_idx];
        if (dali::volume(angle_tensor.shape()) != 1) {
          DALI_FAIL("Angle tensor input must be exactly one element per sample");
        }
        angle = angle_tensor.data<float>()[0];
      } else {
        angle = spec_.GetArgument<float>("angle");
      }
      angle = deg2rad(angle);

      vec2 image_wh;
      if (spec_.HasTensorArgument("input_shape")) {
        auto shape_tensor = ws.ArgumentInput("input_shape")[sample_idx];
        if (dali::volume(shape_tensor.shape()) < shape_max_index_ + 1) {
          DALI_FAIL("`input_shape` ndim is smaller than the `shape_layout` ndim");
        }
        const auto shape_tensor_data = shape_tensor.data<std::int64_t>();
        image_wh.x = shape_tensor_data[shape_wh_index_.first];
        image_wh.y = shape_tensor_data[shape_wh_index_.second];
      } else {
        auto shape_tensor = this->spec_.GetRepeatedArgument<int>("input_shape");
        if (static_cast<int>(shape_tensor.size()) < shape_max_index_ + 1) {
          DALI_FAIL("`input_shape` ndim is smaller than the `shape_layout` ndim");
        }
        image_wh.x = shape_tensor[shape_wh_index_.first];
        image_wh.y = shape_tensor[shape_wh_index_.second];
      }

      vec2 canvas_wh;
      if (keep_size_) {
        canvas_wh = image_wh;
        if (spec_.HasTensorArgument("size") || spec_.HasArgument("size")) {
          DALI_FAIL("fn.bbox_rotate `keep_size` is mutually exclusive with `size` argument");
        }
      } else if (spec_.HasTensorArgument("size")) {
        const auto& size_tensor = ws.ArgumentInput("size")[sample_idx];
        TYPE_SWITCH(size_tensor.type(), type2id, T,
          (int32_t, int64_t, uint32_t, float),
          (
            if (dali::volume(size_tensor.shape()) != 2) {
              DALI_FAIL("`size` tensor argument must be exactly two elements (HW)");
            }
            const auto size_data = size_tensor.data<T>();
            if constexpr (std::is_integral_v<T>) {
              // No need for flooring since already integer type
              canvas_wh.x = size_data[1];
              canvas_wh.y = size_data[0];
            } else {
              canvas_wh.x = std::floor(size_data[1]);
              canvas_wh.y = std::floor(size_data[0]);
            }
          ), // NOLINT
          (DALI_FAIL("`size` must be int32, int64, uint32 or float");)
        );  // NOLINT
      } else if (spec_.HasArgument("size")) {
        auto size_tensor = this->spec_.GetRepeatedArgument<float>("size");
        if (size_tensor.size() != 2) {
          DALI_FAIL("`size` list argument must be exactly two elements (HW)");
        }
        canvas_wh.x = std::floor(size_tensor[1]);
        canvas_wh.y = std::floor(size_tensor[0]);
      } else {
        TensorShape<2> im_shape;
        im_shape[1] = image_wh.x;
        im_shape[0] = image_wh.y;
        auto [shape, parity] = RotatedCanvasSize(im_shape, angle);
        shape += (shape % 2) ^ parity;
        canvas_wh.x = shape.x;
        canvas_wh.y = shape.y;
      }

      kept_box_indices[sample_idx] =
          RotateBoxesKernel(in_boxes[sample_idx], scratch_buffer_[sample_idx], angle, use_ltrb_,
                            mode_, remove_threshold_, image_wh, canvas_wh, bbox_normalized_);
    });
  }

  tpool.RunAll();

  // Number of out boxes is derived from the number of indices kept
  auto num_out_boxes = [&](int idx) {
    return static_cast<dali::span_extent_t>(kept_box_indices[idx].size());
  };

  auto& out_boxes = ws.Output<CPUBackend>(0);
  // Resize out_boxes to the number of kept boxes
  TensorListShape<2> out_boxes_shape(in_boxes.num_samples());
  for (int sample_idx = 0; sample_idx < in_boxes.num_samples(); ++sample_idx) {
    out_boxes_shape.set_tensor_shape(sample_idx,
                                     dali::TensorShape<2>(num_out_boxes(sample_idx), 4));
  }
  out_boxes.Resize(out_boxes_shape, DALI_FLOAT);

  // Copy box data from scratch_buffer_ to out_boxes.
  for (int sample_idx = 0; sample_idx < in_boxes.num_samples(); ++sample_idx) {
    std::memcpy(out_boxes.raw_mutable_tensor(sample_idx), scratch_buffer_[sample_idx].raw_data(),
                4 * num_out_boxes(sample_idx) * sizeof(float));
  }

  // If labels were passed, copy the kept labels using the kept_box_indices
  if (ws.NumInput() == 2) {
    const auto& in_labels = ws.Input<CPUBackend>(1);
    auto& out_labels = ws.Output<CPUBackend>(1);

    // Set the output shape of the labels
    TensorListShape<-1> out_labels_shape(in_labels.num_samples(), in_labels.shape().ndim);
    if (in_labels.shape().ndim == 2) {
      for (int sample_idx = 0; sample_idx < in_boxes.num_samples(); ++sample_idx) {
        out_labels_shape.set_tensor_shape(sample_idx,
                                          dali::TensorShape<2>(num_out_boxes(sample_idx), 1));
      }
    } else {
      for (int sample_idx = 0; sample_idx < in_boxes.num_samples(); ++sample_idx) {
        out_labels_shape.set_tensor_shape(sample_idx,
                                          dali::TensorShape<1>(num_out_boxes(sample_idx)));
      }
    }
    out_labels.Resize(out_labels_shape, DALI_INT32);

    // Copy data from the input to the output
    for (int sample_idx = 0; sample_idx < in_boxes.num_samples(); ++sample_idx) {
      const auto& out_box_indices = kept_box_indices[sample_idx];
      const auto num_in_boxes = in_labels.tensor_shape(sample_idx)[0];
      const auto num_out_boxes = static_cast<dali::span_extent_t>(out_box_indices.size());
      if (num_in_boxes == num_out_boxes) {
        // Run simple memcpy if the no boxes have been pruned
        std::memcpy(out_labels.raw_mutable_tensor(sample_idx), in_labels.raw_tensor(sample_idx),
                    num_out_boxes * sizeof(int));
      } else {
        const auto in_label_span = dali::span(in_labels.tensor<int>(sample_idx), num_in_boxes);
        const auto out_label_span =
            dali::span(out_labels.mutable_tensor<int>(sample_idx), num_out_boxes);
        for (dali::span_extent_t out_idx = 0; out_idx < num_out_boxes; ++out_idx) {
          out_label_span[out_idx] = in_label_span[out_box_indices[out_idx]];
        }
      }
    }
  }
}

DALI_REGISTER_OPERATOR(BBoxRotate, BBoxRotate<CPUBackend>, CPU);

}  // namespace dali
