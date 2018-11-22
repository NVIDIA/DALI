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

#include "dali/pipeline/operators/crop/bbox_crop.h"

namespace dali {

DALI_SCHEMA(RandomBBoxCrop)
    .DocStr(
        R"code(Perform a prospective crop to an image while keeping bounding boxes consistent. Inputs must be supplied as two Tensors:
        `Images` containing image data in NHWC format, and `BBoxes` containing bounding boxes represented as `[l,t,r,b]` or `[x,y,w,h]`.
        Resulting prospective crop is provided as two Tensors: `Begin` containing the starting coordinates for the `crop` in `(x,y)` format,
        and 'Size' containing the dimensions of the `crop` in `(w,h)` format. Bounding boxes are provided as a `(m*4)` Tensor,
        where each bounding box is represented as `[l,t,r,b]` or `[x,y,w,h]`.)code")
    .NumInput(1)
    .NumOutput(3)
    .AddOptionalArg(
        "thresholds",
        R"code(Minimum overlap (Intersection over union) of the bounding boxes with respect to the prospective crop.
    Selected at random for every sample from provided values. Default value is `[0.0]`, leaving the input image as-is in the new crop.)code",
        std::vector<float>{0.f})
    .AddOptionalArg(
        "aspect_ratio",
        R"code(Range `[min, max]` of valid aspect ratio values for new crops. Value for `min` should be greater or equal to `0.0`.
        Default values are `[1.0, 1.0]`, disallowing changes in aspect ratio.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "scaling",
        R"code(Range `[min, max]` for crop size with respect to original image dimensions. Value for `min` should be greater or equal to `0.0`
        Default values are `[1.0, 1.0]`.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "ltrb",
        R"code(If true, bboxes are returned as [left, top, right, bottom], else [x, y, width, height]. By default is set to `true`.)code",
        true)
    .AddOptionalArg(
        "num_attempts",
        R"code(Number of attempts to retrieve a patch with the desired parameters.)code",
        1)
    .EnforceInputLayout(DALI_NHWC);

template <>
void RandomBBoxCrop<CPUBackend>::WriteCropToOutput(SampleWorkspace *ws,
                                                   const Crop &crop) {
  const auto coordinates = crop.AsXywh();

  // Copy the anchor to output
  auto *anchor_out = ws->Output<CPUBackend>(0);
  anchor_out->Resize({2});

  auto *anchor_out_data = anchor_out->mutable_data<float>();
  anchor_out_data[0] = coordinates[0];
  anchor_out_data[1] = coordinates[1];

  // Copy the offsets to output 1
  auto *offsets_out = ws->Output<CPUBackend>(1);
  offsets_out->Resize({2});

  auto *offsets_out_data = offsets_out->mutable_data<float>();
  offsets_out_data[0] = coordinates[2];
  offsets_out_data[1] = coordinates[3];
}

template <>
void RandomBBoxCrop<CPUBackend>::WriteBoxesToOutput(
    SampleWorkspace *ws, const BoundingBoxes &bounding_boxes) {
  auto *bbox_out = ws->Output<CPUBackend>(2);
  bbox_out->Resize(
      {static_cast<Index>(bounding_boxes.size()), BoundingBox::kSize});

  auto *bbox_out_data = bbox_out->mutable_data<float>();
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
void RandomBBoxCrop<CPUBackend>::RunImpl(SampleWorkspace *ws, const int) {
  const auto &boxes_tensor = ws->Input<CPUBackend>(0);

  BoundingBoxes bounding_boxes;
  bounding_boxes.reserve(static_cast<size_t>(boxes_tensor.dim(0)));

  for (int i = 0; i < boxes_tensor.dim(0); ++i) {
    const auto *box_data =
        boxes_tensor.data<float>() + (i * BoundingBox::kSize);

    auto box = ltrb_ ? BoundingBox::FromLtrb(box_data)
                     : BoundingBox::FromXywh(box_data);
    bounding_boxes.emplace_back(box);
  }

  const auto prospective_crop =
      FindProspectiveCrop(bounding_boxes, SelectMinimumOverlap());

  WriteCropToOutput(ws, prospective_crop.first);
  WriteBoxesToOutput(ws, prospective_crop.second);
}

DALI_REGISTER_OPERATOR(RandomBBoxCrop, RandomBBoxCrop<CPUBackend>, CPU);

}  // namespace dali
