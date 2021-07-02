// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <cmath>

#include "dali/operators/ssd/box_encoder.h"

namespace dali {

using BoundingBox = BoxEncoder<CPUBackend>::BoundingBox;

void BoxEncoder<CPUBackend>::CalculateIousForBox(float *ious, const BoundingBox &box) const {
  ious[0] = intersection_over_union(box, anchors_[0]);
  unsigned best_idx = 0;
  float best_iou = ious[0];

  for (unsigned anchor_idx = 1; anchor_idx < anchors_.size(); ++anchor_idx) {
    ious[anchor_idx] = intersection_over_union(box, anchors_[anchor_idx]);

    if (ious[anchor_idx] >= best_iou) {
      best_iou = ious[anchor_idx];
      best_idx = anchor_idx;
    }
  }

  // For best default box matched with current object let iou = 2, to make sure there is a match,
  // as this object will be the best (highest IoU), for this default box
  ious[best_idx] = 2.;
}

vector<float> BoxEncoder<CPUBackend>::CalculateIous(const vector<BoundingBox> &boxes) const {
  vector<float> ious(boxes.size() * anchors_.size());

  for (unsigned bbox_idx = 0; bbox_idx < boxes.size(); ++bbox_idx) {
    auto ious_row = ious.data() + bbox_idx * anchors_.size();
    CalculateIousForBox(ious_row, boxes[bbox_idx]);
  }

  return ious;
}

unsigned BoxEncoder<CPUBackend>::FindBestBoxForAnchor(
  unsigned anchor_idx, const vector<float> &ious, unsigned num_boxes) const {
  unsigned best_idx = 0;
  float best_iou = ious[anchor_idx];

  for (unsigned bbox_idx = 1; bbox_idx < num_boxes; ++bbox_idx) {
    if (ious[bbox_idx * anchors_.size() + anchor_idx] >= best_iou) {
      best_iou = ious[bbox_idx * anchors_.size() + anchor_idx];
      best_idx = bbox_idx;
    }
  }

  return best_idx;
}

vector<std::pair<unsigned, unsigned>> BoxEncoder<CPUBackend>::MatchBoxesWithAnchors(
  const vector<BoundingBox> &boxes) const {
  const auto ious = CalculateIous(boxes);
  vector<std::pair<unsigned, unsigned>> matches;

  for (unsigned anchor_idx = 0; anchor_idx < anchors_.size(); ++anchor_idx) {
    const auto best_idx = FindBestBoxForAnchor(anchor_idx, ious, boxes.size());

    // Filter matches by criteria
    if (ious[best_idx * anchors_.size() + anchor_idx] > criteria_) {
      matches.push_back({best_idx, anchor_idx});
    }
  }

  return matches;
}

template <int ndim>
void WriteBoxToOutput(float *out_box_data, const vec<ndim, float> &center,
                      const vec<ndim, float> &extent) {
  for (int d = 0; d < ndim; d++) {
    out_box_data[d] = center[d];
    out_box_data[ndim + d] = extent[d];
  }
}

void BoxEncoder<CPUBackend>::WriteAnchorsToOutput(float *out_boxes, int *out_labels) const {
  if (offset_) {
    std::memset(out_boxes, 0,
                sizeof(std::remove_pointer<decltype(out_boxes)>::type) * BoundingBox::size *
                    anchors_.size());
    std::memset(out_labels, 0,
                sizeof(std::remove_pointer<decltype(out_labels)>::type) * anchors_.size());
  } else {
    for (unsigned int idx = 0; idx < anchors_.size(); idx++) {
      float *out_box = out_boxes + idx * BoundingBox::size;
      const auto &anchor = anchors_[idx];
      WriteBoxToOutput(out_box, anchor.centroid(), anchor.extent());
      out_labels[idx] = 0;
    }
  }
}

// Calculate offset from CenterWH ref box and anchor
// based on eq (2)  in https://arxiv.org/abs/1512.02325 with extra normalization
std::pair<vec2, vec2>
GetOffsets(vec2 box_center,
           vec2 box_extent,
           vec2 anchor_center,
           vec2 anchor_extent,
           const std::vector<float>& means,
           const std::vector<float>& stds,
           float scale) {
  box_center *= scale;
  box_extent *= scale;
  anchor_center *= scale;
  anchor_extent *= scale;
  vec2 center, extent;
  center[0] = ((box_center[0] - anchor_center[0]) / anchor_extent[0] - means[0]) / stds[0];
  center[1] = ((box_center[1] - anchor_center[1]) / anchor_extent[1] - means[1]) / stds[1];
  extent[0] = (std::log(box_extent[0] / anchor_extent[0]) - means[2]) / stds[2];
  extent[1] = (std::log(box_extent[1] / anchor_extent[1]) - means[3]) / stds[3];
  return std::make_pair(center, extent);
}

void BoxEncoder<CPUBackend>::WriteMatchesToOutput(
  const vector<std::pair<unsigned, unsigned>> &matches, const vector<BoundingBox> &boxes,
  const int *labels, float *out_boxes, int *out_labels) const {
  if (offset_) {
    for (const auto &match : matches) {
      auto box = boxes[match.first];
      auto anchor = anchors_[match.second];

      vec2 center, extent;
      std::tie(center, extent) = GetOffsets(box.centroid(), box.extent(), anchor.centroid(),
                                            anchor.extent(), means_, stds_, scale_);
      WriteBoxToOutput(out_boxes + match.second * BoundingBox::size, center, extent);
      out_labels[match.second] = labels[match.first];
    }
  } else {
    for (const auto &match : matches) {
      auto box = boxes[match.first];
      WriteBoxToOutput(out_boxes + match.second * BoundingBox::size, box.centroid(),
                       box.extent());
      out_labels[match.second] = labels[match.first];
    }
  }
}

void BoxEncoder<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  const auto &bboxes_input = ws.Input<CPUBackend>(kBoxesInId);
  const auto &labels_input = ws.Input<CPUBackend>(kLabelsInId);
  const auto num_boxes = bboxes_input.dim(0);
  const auto labels = labels_input.data<int>();

  vector<BoundingBox> boxes;
  boxes.resize(num_boxes);
  ReadBoxes(make_span(boxes), make_cspan(bboxes_input.data<float>(), bboxes_input.size()), {}, {});

  // Create output
  auto &bboxes_output = ws.Output<CPUBackend>(kBoxesOutId);
  bboxes_output.set_type(bboxes_input.type());
  bboxes_output.Resize({static_cast<int>(anchors_.size()), BoundingBox::size});
  auto out_boxes = bboxes_output.mutable_data<float>();

  auto &labels_output = ws.Output<CPUBackend>(kLabelsOutId);
  labels_output.set_type(labels_input.type());
  labels_output.Resize({static_cast<int>(anchors_.size())});
  auto out_labels = labels_output.mutable_data<int>();

  WriteAnchorsToOutput(out_boxes, out_labels);
  if (num_boxes == 0)
    return;

  const auto matches = MatchBoxesWithAnchors(boxes);
  WriteMatchesToOutput(matches, boxes, labels, out_boxes, out_labels);
}

DALI_REGISTER_OPERATOR(BoxEncoder, BoxEncoder<CPUBackend>, CPU);

DALI_SCHEMA(BoxEncoder)
    .DocStr(
        R"code(Encodes the input bounding boxes and labels using a set of default boxes (anchors)
passed as an argument.

This operator follows the algorithm described in "SSD: Single Shot MultiBox Detector"
and implemented in https://github.com/mlperf/training/tree/master/single_stage_detector/ssd.
Inputs must be supplied as the following Tensors:

- ``BBoxes`` that contain bounding boxes that are represented as ``[l,t,r,b]``.
- ``Labels`` that contain the corresponding label for each bounding box.

The results are two tensors:

- ``EncodedBBoxes`` that contain M-encoded bounding boxes as ``[l,t,r,b]``, where M is number of
  anchors.
- ``EncodedLabels`` that contain the corresponding label for each encoded box.
)code")
    .NumInput(2)
    .NumOutput(2)
    .AddArg("anchors",
            R"code(Anchors to be used for encoding, as the list of floats is in the ``ltrb``
format.)code",
            DALI_FLOAT_VEC)
    .AddOptionalArg(
        "criteria",
        R"code(Threshold IoU for matching bounding boxes with anchors.

The value needs to be between 0 and 1.)code",
        0.5f, false)
    .AddOptionalArg(
        "offset",
        R"code(Returns normalized offsets ``((encoded_bboxes*scale - anchors*scale) - mean) / stds``
in EncodedBBoxes that use ``std`` and the ``mean`` and ``scale`` arguments.)code",
        false)
    .AddOptionalArg("scale",
            R"code(Rescales the box and anchor values before the offset is calculated
(for example, to return to the absolute values).)code",
            1.0f)
    .AddOptionalArg("means",
            R"code([x y w h] mean values for normalization.)code",
            std::vector<float>{0.f, 0.f, 0.f, 0.f})
    .AddOptionalArg("stds",
            R"code([x y w h] standard deviations for offset normalization.)code",
            std::vector<float>{1.f, 1.f, 1.f, 1.f});

}  // namespace dali
