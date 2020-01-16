// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/crop/bbox_crop.h"

namespace dali {

DALI_SCHEMA(RandomBBoxCrop)
    .DocStr(
        R"code(Perform a prospective crop to an image while keeping bounding boxes and labels consistent. Inputs must be supplied as
two Tensors: `BBoxes` containing bounding boxes represented as `[l,t,r,b]` or `[x,y,w,h]`, and `Labels` containing the
corresponding label for each bounding box. Resulting prospective crop is provided as two Tensors: `Begin` containing the starting
coordinates for the `crop` in `(x,y)` format, and 'Size' containing the dimensions of the `crop` in `(w,h)` format.
Bounding boxes are provided as a `(m*4)` Tensor, where each bounding box is represented as `[l,t,r,b]` or `[x,y,w,h]`. Resulting
labels match the boxes that remain, after being discarded with respect to the minimum accepted intersection threshold.
Be advised, when `allow_no_crop` is `false` and `thresholds` does not contain `0` it is good to increase `num_attempts` as otherwise
it may loop for a very long time.)code")
    .NumInput(2)
    .NumOutput(4)
    .AddOptionalArg(
        "thresholds",
        R"code(Minimum overlap (Intersection over union) of the bounding boxes with respect to the prospective crop.
Selected at random for every sample from provided values. Default imposes no restrictions on Intersection over Union for boxes and crop.)code",
        std::vector<float>{0.f})
    .AddOptionalArg(
        "aspect_ratio",
        R"code(Range `[min, max]` of valid aspect ratio values for new crops. Value for `min` should be greater or equal to `0.0`.
Default values disallow changes in aspect ratio.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "scaling",
        R"code(Range `[min, max]` for crop size with respect to original image dimensions. Value for `min` should be greater or equal to `0.0`.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "ltrb",
        R"code(If true, bboxes are returned as [left, top, right, bottom], else [x, y, width, height].)code",
        true)
    .AddOptionalArg(
        "num_attempts",
        R"code(Number of attempts to retrieve a patch with the desired parameters.)code",
        1)
    .AddOptionalArg(
        "allow_no_crop",
        R"code(If true, includes no cropping as one of the random options.)code",
        true);

template <>
void RandomBBoxCrop<CPUBackend>::WriteCropToOutput(
  SampleWorkspace &ws, const Crop &crop) const {
  const auto coordinates = crop.AsXywh();

  // Copy the anchor to output
  auto &anchor_out = ws.Output<CPUBackend>(0);
  anchor_out.Resize({2});

  auto *anchor_out_data = anchor_out.mutable_data<float>();
  anchor_out_data[0] = coordinates[0];
  anchor_out_data[1] = coordinates[1];

  // Copy the offsets to output 1
  auto &offsets_out = ws.Output<CPUBackend>(1);
  offsets_out.Resize({2});

  auto *offsets_out_data = offsets_out.mutable_data<float>();
  offsets_out_data[0] = coordinates[2];
  offsets_out_data[1] = coordinates[3];
}

template <>
void RandomBBoxCrop<CPUBackend>::WriteBoxesToOutput(
    SampleWorkspace &ws, const BoundingBoxes &bounding_boxes) const {
  auto &bbox_out = ws.Output<CPUBackend>(2);
  bbox_out.Resize(
      {static_cast<int64_t>(bounding_boxes.size()), static_cast<int64_t>(BoundingBox::kSize)});

  auto *bbox_out_data = bbox_out.mutable_data<float>();
  for (size_t i = 0; i < bounding_boxes.size(); ++i) {
    auto *output = bbox_out_data + i * BoundingBox::kSize;
    auto coordinates =
        ltrb_ ? bounding_boxes[i].AsLtrb() : bounding_boxes[i].AsXywh();
    output[0] = coordinates[0];
    output[1] = coordinates[1];
    output[2] = coordinates[2];
    output[3] = coordinates[3];
  }
}

template <>
void RandomBBoxCrop<CPUBackend>::WriteLabelsToOutput(
  SampleWorkspace &ws, const std::vector<int> &labels) const {
  auto &labels_out = ws.Output<CPUBackend>(3);
  labels_out.Resize({static_cast<Index>(labels.size()), 1});

  auto *labels_out_data = labels_out.mutable_data<int>();
  for (size_t i = 0; i < labels.size(); ++i) {
    labels_out_data[i] = labels[i];
  }
}

template <>
void RandomBBoxCrop<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  const auto &boxes_tensor = ws.Input<CPUBackend>(0);

  BoundingBoxes bounding_boxes;
  bounding_boxes.reserve(static_cast<size_t>(boxes_tensor.dim(0)));

  for (int i = 0; i < boxes_tensor.dim(0); ++i) {
    const auto *box_data =
        boxes_tensor.data<float>() + (i * BoundingBox::kSize);

    auto box = ltrb_ ? BoundingBox::FromLtrb(box_data)
                     : BoundingBox::FromXywh(box_data);
    bounding_boxes.emplace_back(box);
  }

  const auto &labels_tensor = ws.Input<CPUBackend>(1);

  std::vector<int> labels;
  labels.reserve(static_cast<size_t>(labels_tensor.dim(0)));

  for (int i = 0; i < labels_tensor.dim(0); ++i) {
    const auto *label_data = labels_tensor.data<int>() + i;
    labels.emplace_back(*label_data);
  }

  ProspectiveCrop prospective_crop;
  int sample = ws.data_idx();
  while (!prospective_crop.success)
    prospective_crop  = FindProspectiveCrop(
        bounding_boxes, labels, SelectMinimumOverlap(sample), sample);

  const auto &selected_boxes = prospective_crop.boxes;
  const auto &selected_labels = prospective_crop.labels;

  DALI_ENFORCE(selected_boxes.size() == selected_labels.size(),
              "Expected boxes.size() == labels.size(). Received: " +
                  std::to_string(selected_boxes.size()) +
                  "!=" + std::to_string(selected_labels.size()));

  WriteCropToOutput(ws, prospective_crop.crop);
  WriteBoxesToOutput(ws, selected_boxes);
  WriteLabelsToOutput(ws, selected_labels);
}

DALI_REGISTER_OPERATOR(RandomBBoxCrop, RandomBBoxCrop<CPUBackend>, CPU);

}  // namespace dali
