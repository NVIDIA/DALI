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
namespace detail {
using Anchor = BoxEncoder<CPUBackend>::Anchor;
using BBox = BoxEncoder<CPUBackend>::BBox;

float Iou(const Anchor a1, const Anchor a2) {
  float l = std::max(a1.x, a2.x);
  float t = std::max(a1.y, a2.y);
  float r = std::min(a1.z, a2.z);
  float b = std::min(a1.w, a2.w);

  float width = r - l;
  float height = b - t;

  if (width <= 0.f || height <= 0.f) return 0.f;

  float intersection = width * height;
  float area1 = (a1.w - a1.y) * (a1.z - a1.x);
  float area2 = (a2.w - a2.y) * (a2.z - a2.x);

  return intersection / (area1 + area2 - intersection);
}

vector<float> CalculateIous(const Anchor *anchors, const BBox *bboxes, int N, int M) {
  vector<float> ious(N * M);

  for (int bbox_idx = 0; bbox_idx < N; ++bbox_idx) {
    int best_idx = -1;
    float best_iou = -1.;

    for (int anchor_idx = 0; anchor_idx < M; ++anchor_idx) {
      ious[bbox_idx * M + anchor_idx] = detail::Iou(bboxes[bbox_idx], anchors[anchor_idx]);

      if (ious[bbox_idx * M + anchor_idx] >= best_iou) {
        best_iou = ious[bbox_idx * M + anchor_idx];
        best_idx = anchor_idx;
      }
    }

    // For best default box matched with current object let iou = 2, to make sure there is a match,
    // as this object will be the best (highest IOU), for this default box
    ious[bbox_idx * M + best_idx] = 2.;
  }
  return ious;
}

Anchor ToCenterWH(const Anchor a) {
  return {
    0.5f * (a.x + a.z),
    0.5f * (a.y + a.w),
    a.z - a.x, 
    a.w - a.y};
}
}  // namespace detail

template <>
void BoxEncoder<CPUBackend>::FindMatchingAnchors(
  const float *ious, const int64 N, const BBox *bboxes,
  const int *labels, BBox *out_boxes, int *out_labels) {
  // For every anchor we are looking, if there is a match
  for (int anchor_idx = 0; anchor_idx < M_; ++anchor_idx) {
    int best_idx = -1;
    float best_iou = -1.;

    //Looking for bbox with best IOU with current anchor
    for (int bbox_idx = 0; bbox_idx < N; ++bbox_idx) {
      if (ious[bbox_idx * M_ + anchor_idx] >= best_iou) {
        best_iou = ious[bbox_idx * M_ + anchor_idx];
        best_idx = bbox_idx;
      }
    }

    out_labels[anchor_idx] = 0;

    // Filter matches by criteria
    // We only report a match, when IOU > criteria
    if (best_iou > criteria_) {
      out_boxes[anchor_idx] = bboxes[best_idx];
      out_labels[anchor_idx] = labels[best_idx];
    }

    // Change to x, y, w, h per canonical SSD implementation
    out_boxes[anchor_idx] = detail::ToCenterWH(out_boxes[anchor_idx]);
  }
}

template <>
void BoxEncoder<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &bboxes_input = ws->Input<CPUBackend>(0);
  const auto &labels_input = ws->Input<CPUBackend>(1);

  const auto N = bboxes_input.dim(0);

  auto bboxes = reinterpret_cast<const BBox *>(bboxes_input.data<float>());
  auto labels = labels_input.data<int>();
  auto anchors = reinterpret_cast<const Anchor *>(anchors_.data<float>());

  auto ious = detail::CalculateIous(anchors, bboxes, N, M_);

  // Create output
  auto bboxes_output = ws->Output<CPUBackend>(0);
  bboxes_output->set_type(anchors_.type());
  bboxes_output->ResizeLike(anchors_);
  auto out_boxes = reinterpret_cast<BBox *>(bboxes_output->mutable_data<float>());

  auto labels_output = ws->Output<CPUBackend>(1);
  labels_output->set_type(labels_input.type());
  labels_output->Resize({M_});
  auto out_labels = labels_output->mutable_data<int>();

  // Copy default boxes (anchors) to output
  TypeInfo type = anchors_.type();
  type.Copy<CPUBackend, CPUBackend>(bboxes_output->raw_mutable_data(), anchors_.raw_data(),
                                    anchors_.size(), 0);

  FindMatchingAnchors(ious.data(), N, bboxes, labels, out_boxes, out_labels);
}

DALI_REGISTER_OPERATOR(BoxEncoder, BoxEncoder<CPUBackend>, CPU);

DALI_SCHEMA(BoxEncoder)
    .DocStr(
        R"code("Encodes input bounding boxes and labels using set of default boxes (anchors) passed 
        during op construction. Follows algorithm described in https://arxiv.org/abs/1512.02325 and 
        implemented in https://github.com/mlperf/training/tree/master/single_stage_detector/ssd.
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
