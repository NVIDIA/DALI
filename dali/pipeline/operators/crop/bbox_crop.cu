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

template <>
void RandomBBoxCrop<GPUBackend>::WriteCropToOutput(DeviceWorkspace *ws,
                       const std::vector<Crop> &crop,
                       const std::vector<unsigned int>& height,
                       const std::vector<unsigned int>& width,
                       unsigned int idx) {
    const static unsigned int kAnchorSize = 2;
    const static unsigned int kOffsetSize = 2;

    DALI_ENFORCE(height.size() == static_cast<size_t>(batch_size_));
    DALI_ENFORCE(width.size() == static_cast<size_t>(batch_size_));
    DALI_ENFORCE(crop.size() == static_cast<size_t>(batch_size_));

    std::vector<Dims> anchors_shape(batch_size_, {kAnchorSize});
    std::vector<Dims> offsets_shape(batch_size_, {kOffsetSize});

    const auto anchors = ws->Output<GPUBackend>(idx);
    const auto offsets = ws->Output<GPUBackend>(idx+1);

    anchors->Resize(anchors_shape);
    offsets->Resize(offsets_shape);

    for (int i = 0; i < batch_size_; ++i) {
        float* anchor_out_data = anchors->template mutable_data<float>() + (i*kAnchorSize);
        anchor_out_data[0] = crop[i].left * width[i];
        anchor_out_data[1] = crop[i].top * height[i];

        float* offsets_out_data = offsets->template mutable_data<float>() + (i*kOffsetSize);
        offsets_out_data[0] = (crop[i].right - crop[i].left) * width[i];
        offsets_out_data[1] = (crop[i].bottom - crop[i].top) * height[i];
    }
}

template <>
void RandomBBoxCrop<GPUBackend>::WriteBoxesToOutput(DeviceWorkspace *ws,
                        const std::vector<BoundingBoxes> &bounding_boxes,
                        unsigned int idx) {
    DALI_ENFORCE(bounding_boxes.size() == static_cast<size_t>(batch_size_));

    std::vector<Dims> boxes_shape(batch_size_);

    for (int i = 0; i < batch_size_; ++i) {
        boxes_shape[i] = {static_cast<long>(bounding_boxes[i].size()), kBboxSize};
    }

    const auto boxes = ws->Output<GPUBackend>(idx+2);
    boxes->Resize(boxes_shape);

    float* boxes_out_data = boxes->template mutable_data<float>();

    for (int i = 0; i < batch_size_; ++i) {
        for (size_t j = 0; j < bounding_boxes[i].size(); j++) {
            boxes_out_data[0] = bounding_boxes[i][j].left;
            boxes_out_data[1] = bounding_boxes[i][j].top;
            boxes_out_data[2] = ltrb_ ? bounding_boxes[i][j].right
                                      : bounding_boxes[i][j].right - bounding_boxes[i][j].left;
            boxes_out_data[3] = ltrb_ ? bounding_boxes[i][j].bottom
                                      : bounding_boxes[i][j].bottom - bounding_boxes[i][j].top;

            boxes_out_data += kBboxSize;
        }
    }
}

template <>
void RandomBBoxCrop<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  std::vector<Crop> crop;
  std::vector<unsigned int> height;
  std::vector<unsigned int> width;
  std::vector<BoundingBoxes> boxes_out;
  crop.reserve(batch_size_);
  height.reserve(batch_size_);
  width.reserve(batch_size_);
  boxes_out.reserve(batch_size_);

  auto &images = ws->Input<GPUBackend>(idx);
  auto &boxes = ws->Input<GPUBackend>(idx+1);

  for (int i = 0; i < batch_size_; ++i) {
    BoundingBoxes bounding_boxes;
    bounding_boxes.reserve(boxes.tensor_shape(i)[0]);

    for (int j = 0; i < boxes.tensor_shape(i)[0]; ++i) {
      const float *box = boxes.template tensor<float>(j) + (j * kBboxSize);
      // ltrb expected
      bounding_boxes.emplace_back(box[0], box[1], box[2], box[3]);
    }

    const ImageShape image_shape(images.tensor_shape(i));

    auto prospective_crop =
        FindProspectiveCrop(image_shape, bounding_boxes, SelectMinimumOverlap());

    crop.push_back(prospective_crop.first);
    height.push_back(image_shape.height);
    width.push_back(image_shape.width);
    boxes_out.emplace_back(prospective_crop.second);
  }

  WriteCropToOutput(ws, crop, height, width, idx);
  WriteBoxesToOutput(ws, boxes_out, idx);
}

// Register operator
DALI_REGISTER_OPERATOR(RandomBBoxCrop, RandomBBoxCrop<GPUBackend>, GPU);

}  // namespace dali
