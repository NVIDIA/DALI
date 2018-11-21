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
float iou(const float4 b1, const float4 b2) {
  // (lt), (rb)
  float l = std::max(b1.x, b2.x);
  float t = std::max(b1.y, b2.y);
  float r = std::min(b1.z, b2.z);
  float b = std::min(b1.w, b2.w);

  float first = (r - l);
  first = (first < 0) ? 0 : first;
  float second = (b - t);
  second = (second < 0) ? 0 : second;

  float intersection = first * second;

  float area1 = (b1.w - b1.y) * (b1.z - b1.x);
  float area2 = (b2.w - b2.y) * (b2.z - b2.x);

  return intersection / (area1 + area2 - intersection);
}

float4 to_xywh(float4 box) {
  float4 result;
  result.x = 0.5f * (box.x + box.z);
  result.y = 0.5f * (box.y + box.w);
  result.z = box.z - box.x;
  result.w = box.w - box.y;

  return result;
}
}  // namespace detail

template <>
void SSDBoxEncoder<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &bboxes = ws->Input<CPUBackend>(0);
  const auto &labels = ws->Input<CPUBackend>(1);

  const auto N = bboxes.dim(0);
  const auto M = anchors_.dim(0);

  auto bboxes_data = reinterpret_cast<const float4 *>(bboxes.data<float>());
  auto labels_data = labels.data<int>();
  auto anchors_data = reinterpret_cast<const float4 *>(anchors_.data<float>());

  Tensor<CPUBackend> ious;
  ious.Resize({N, M});
  auto ious_data = ious.mutable_data<float>();

  for (int bbox_idx = 0; bbox_idx < N; ++bbox_idx) {
    int best_anchor_idx = -1;
    float best_dbox_iou = -1.;

    for (int anchor_idx = 0; anchor_idx < M; ++anchor_idx) {
      ious_data[bbox_idx * M + anchor_idx] =
          detail::iou(bboxes_data[bbox_idx], anchors_data[anchor_idx]);

      if (ious_data[bbox_idx * M + anchor_idx] >= best_dbox_iou) {
        best_dbox_iou = ious_data[bbox_idx * M + anchor_idx];
        best_anchor_idx = anchor_idx;
      }
    }

    // For best default box matched with current object let iou = 2, to make sure there is a match,
    // as this object will be the best (highest IOU), for this default box
    ious_data[bbox_idx * M + best_anchor_idx] = 2.;
  }

  // Create output
  auto encoded_bboxes = ws->Output<CPUBackend>(0);
  encoded_bboxes->set_type(anchors_.type());
  encoded_bboxes->ResizeLike(anchors_);

  auto encoded_labels = ws->Output<CPUBackend>(1);
  encoded_labels->set_type(labels.type());
  encoded_labels->Resize({M});
  int *encoded_labels_data = encoded_labels->mutable_data<int>();

  // Copy default boxes (anchors) to output
  TypeInfo type = anchors_.type();
  type.Copy<CPUBackend, CPUBackend>(encoded_bboxes->raw_mutable_data(), anchors_.raw_data(),
                                    anchors_.size(), 0);

  auto encoded_boxes_data = reinterpret_cast<float4 *>(encoded_bboxes->mutable_data<float>());

  // For every default box we are looking for the match
  for (int anchor_idx = 0; anchor_idx < M; ++anchor_idx) {
    int best_bbox_idx = -1;
    float best_bbox_iou = -1.;

    for (int bbox_idx = 0; bbox_idx < N; ++bbox_idx) {
      if (ious_data[bbox_idx * M + anchor_idx] >= best_bbox_iou) {
        best_bbox_iou = ious_data[bbox_idx * M + anchor_idx];
        best_bbox_idx = bbox_idx;
      }
    }

    encoded_labels_data[anchor_idx] = 0;

    // Filter matches by criteria
    // We only report a match, when IOU > criteria
    if (best_bbox_iou > criteria_) {
      encoded_boxes_data[anchor_idx] = bboxes_data[best_bbox_idx];
      encoded_labels_data[anchor_idx] = labels_data[best_bbox_idx];
    }

    // Change to x, y, w, h per canonical SSD implementation
    encoded_boxes_data[anchor_idx] = detail::to_xywh(encoded_boxes_data[anchor_idx]);
  }
}

DALI_REGISTER_OPERATOR(SSDBoxEncoder, SSDBoxEncoder<CPUBackend>, CPU);

DALI_SCHEMA(SSDBoxEncoder)
    .DocStr(
        "Matches set of bounding boxes to set of default bounding boxes (anchors) according to SSD "
        "algorithm.")
    .NumInput(2)
    .NumOutput(2)
    .AddArg("anchors", R"code(Default bounding boxes to be encoded. List of floats in ltrb format.)code", DALI_FLOAT_VEC)
    .AddOptionalArg("criteria",
                    R"code(Threshold IOU for matching bounding boxes with anchors)code", 0.5f,
                    DALI_FLOAT);

}  // namespace dali
