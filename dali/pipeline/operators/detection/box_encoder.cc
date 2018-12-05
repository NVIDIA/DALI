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

#include "dali/pipeline/operators/detection/box_encoder.h"

namespace dali {
void BoxEncoder<CPUBackend>::CalculateIousForBox(float *ious, const BoundingBox &box) const {
  ious[0] = box.IntersectionOverUnion(anchors_[0]);
  unsigned best_idx = 0;
  float best_iou = ious[0];

  for (unsigned anchor_idx = 1; anchor_idx < anchors_.size(); ++anchor_idx) {
    ious[anchor_idx] = box.IntersectionOverUnion(anchors_[anchor_idx]);

    if (ious[anchor_idx] >= best_iou) {
      best_iou = ious[anchor_idx];
      best_idx = anchor_idx;
    }
  }

  // For best default box matched with current object let iou = 2, to make sure there is a match,
  // as this object will be the best (highest IOU), for this default box
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

vector<BoundingBox> BoxEncoder<CPUBackend>::ReadBoxesFromInput(
  const float *in_boxes, unsigned num_boxes) const {
  vector<BoundingBox> boxes;
  boxes.reserve(num_boxes);
  auto in_box_data = in_boxes;

  for (unsigned idx = 0; idx < num_boxes; ++idx) {
    boxes.push_back(BoundingBox::FromLtrb(in_box_data, BoundingBox::NoBounds()));
    in_box_data += BoundingBox::kSize;
  }

  return boxes;
}

void BoxEncoder<CPUBackend>::WriteBoxToOutput(const BoundingBox &box, float *out_box_data) const {
  const auto out_box = box.AsCenterWh();

  for (unsigned idx = 0; idx < BoundingBox::kSize; ++idx)
    out_box_data[idx] = out_box[idx];
}

void BoxEncoder<CPUBackend>::WriteAnchorsToOutput(float *out_boxes, int *out_labels) const {
  for (unsigned idx = 0; idx < anchors_.size(); ++idx) {
    WriteBoxToOutput(anchors_[idx], out_boxes + idx * BoundingBox::kSize);
    out_labels[idx] = 0;
  }
}

void BoxEncoder<CPUBackend>::WriteMatchesToOutput(
  const vector<std::pair<unsigned, unsigned>> matches, const vector<BoundingBox> &boxes,
  const int *labels, float *out_boxes, int *out_labels) const {
  for (const auto &match : matches) {
    WriteBoxToOutput(boxes[match.first], out_boxes + match.second * BoundingBox::kSize);
    out_labels[match.second] = labels[match.first];
  }
}

void BoxEncoder<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &bboxes_input = ws->Input<CPUBackend>(0);
  const auto &labels_input = ws->Input<CPUBackend>(1);

  const auto num_boxes = bboxes_input.dim(0);

  const auto labels = labels_input.data<int>();
  const auto boxes = ReadBoxesFromInput(bboxes_input.data<float>(), num_boxes);

  // Create output
  auto bboxes_output = ws->Output<CPUBackend>(0);
  bboxes_output->set_type(bboxes_input.type());
  bboxes_output->Resize({static_cast<int>(anchors_.size()), BoundingBox::kSize});
  auto out_boxes = bboxes_output->mutable_data<float>();

  auto labels_output = ws->Output<CPUBackend>(1);
  labels_output->set_type(labels_input.type());
  labels_output->Resize({static_cast<int>(anchors_.size())});
  auto out_labels = labels_output->mutable_data<int>();

  WriteAnchorsToOutput(out_boxes, out_labels);

  if (num_boxes == 0)
    return;

  const auto matches = MatchBoxesWithAnchors(boxes);
  WriteMatchesToOutput(matches, boxes, labels, out_boxes, out_labels);
}

DALI_REGISTER_OPERATOR(BoxEncoder, BoxEncoder<CPUBackend>, CPU);

DALI_SCHEMA(BoxEncoder)
    .DocStr(
        R"code("Encodes input bounding boxes and labels using set of default boxes (anchors) passed
during op construction. Follows algorithm described in https://arxiv.org/abs/1512.02325 and
implemented in https://github.com/mlperf/training/tree/master/single_stage_detector/ssd
Inputs must be supplied as two Tensors: `BBoxes` containing bounding boxes represented as
`[l,t,r,b]`, and `Labels` containing the corresponding label for each bounding box.
Results are two tensors: `EncodedBBoxes` containing M encoded bounding boxes as `[l,t,r,b]`,
where M is number of anchors and `EncodedLabels` containing the corresponding label for each
encoded box.")code")
    .NumInput(2)
    .NumOutput(2)
    .AddArg("anchors",
            R"code(Anchors to be used for encoding. List of floats in ltrb format.)code",
            DALI_FLOAT_VEC)
    .AddOptionalArg(
        "criteria",
        R"code(Threshold IOU for matching bounding boxes with anchors. Value between 0 and 1. Default is 0.5.)code",
        0.5f, false);

}  // namespace dali
