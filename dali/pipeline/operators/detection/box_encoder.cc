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
vector<float> BoxEncoder<CPUBackend>::CalculateIous(vector<BoundingBox> boxes) {
  vector<float> ious(boxes.size() * NumAnchors());

  for (auto bbox_idx = 0u; bbox_idx < boxes.size(); ++bbox_idx) {
    int best_idx = -1;
    float best_iou = -1.;

    for (int anchor_idx = 0; anchor_idx < NumAnchors(); ++anchor_idx) {
      ious[bbox_idx * anchors_.size() + anchor_idx] =
          boxes[bbox_idx].IntersectionOverUnion(anchors_[anchor_idx]);

      if (ious[bbox_idx * anchors_.size() + anchor_idx] >= best_iou) {
        best_iou = ious[bbox_idx * NumAnchors() + anchor_idx];
        best_idx = anchor_idx;
      }
    }

    // For best default box matched with current object let iou = 2, to make sure there is a match,
    // as this object will be the best (highest IOU), for this default box
    ious[bbox_idx * NumAnchors() + best_idx] = 2.;
  }

  return ious;
}

void BoxEncoder<CPUBackend>::MatchBoxesWithAnchors(
  const vector<BoundingBox> &boxes, const int *labels, float *out_boxes, int *out_labels) {
  int num_boxes = boxes.size();
  auto ious = CalculateIous(boxes);

  for (int anchor_idx = 0; anchor_idx < NumAnchors(); ++anchor_idx) {
    int best_idx = -1;
    float best_iou = -1.;

    for (int bbox_idx = 0; bbox_idx < num_boxes; ++bbox_idx) {
      if (ious[bbox_idx * NumAnchors() + anchor_idx] >= best_iou) {
        best_iou = ious[bbox_idx * NumAnchors() + anchor_idx];
        best_idx = bbox_idx;
      }
    }

    out_labels[anchor_idx] = 0;

    // Filter matches by criteria
    // We only report a match, when IOU > criteria
    if (best_iou > criteria_) {
      out_labels[anchor_idx] = labels[best_idx];
      boxes[best_idx].CopyAsCenterWh(out_boxes + anchor_idx * BoundingBox::kSize);
    }
  }
}

void BoxEncoder<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &bboxes_input = ws->Input<CPUBackend>(0);
  const auto &labels_input = ws->Input<CPUBackend>(1);

  const auto num_boxes = bboxes_input.dim(0);

  auto labels = labels_input.data<int>();
  auto boxes = BoundingBox::FromLtrbArray(
    bboxes_input.data<float>(), num_boxes, false);

  // Create output
  auto bboxes_output = ws->Output<CPUBackend>(0);
  bboxes_output->set_type(bboxes_input.type());
  bboxes_output->Resize({NumAnchors(), BoundingBox::kSize});
  auto out_boxes = bboxes_output->mutable_data<float>();

  auto labels_output = ws->Output<CPUBackend>(1);
  labels_output->set_type(labels_input.type());
  labels_output->Resize({NumAnchors()});
  auto out_labels = labels_output->mutable_data<int>();

  for (int idx = 0; idx < NumAnchors(); ++idx)
    anchors_[idx].CopyAsCenterWh(out_boxes + idx * BoundingBox::kSize);

  MatchBoxesWithAnchors(boxes, labels, out_boxes, out_labels);
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
        0.5f, DALI_FLOAT);

}  // namespace dali
