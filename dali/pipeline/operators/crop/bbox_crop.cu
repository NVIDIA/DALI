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

//template <>
//void RandomBBoxCrop<GPUBackend>::WriteCropToOutput(
//    DeviceWorkspace *ws, const std::vector<Crop> &crop, unsigned int idx) {
//  const static unsigned int kAnchorSize = 2U;
//  const static unsigned int kOffsetSize = 2U;
//
//  DALI_ENFORCE(shape.size() == static_cast<size_t>(batch_size_));
//  DALI_ENFORCE(crop.size() == static_cast<size_t>(batch_size_));
//
//  std::vector<Dims> anchors_shape(static_cast<unsigned long>(batch_size_),
//                                  {kAnchorSize});
//  std::vector<Dims> offsets_shape(static_cast<unsigned long>(batch_size_),
//                                  {kOffsetSize});
//
//  TensorList<CPUBackend> anchors;
//  TensorList<CPUBackend> offsets;
//
//  anchors.Resize(anchors_shape);
//  offsets.Resize(offsets_shape);
//
//  for (int i = 0; i < batch_size_; ++i) {
//    auto anchor_out_data =
//        anchors.template mutable_data<float>() + (i * kAnchorSize);
//    anchor_out_data[0] = crop[i].left;
//    anchor_out_data[1] = crop[i].top;
//
//    auto offsets_out_data =
//        offsets.template mutable_data<float>() + (i * kOffsetSize);
//    offsets_out_data[0] = (crop[i].right - crop[i].left);
//    offsets_out_data[1] = (crop[i].bottom - crop[i].top);
//  }
//
//  ws->Output<GPUBackend>(idx)->Copy(anchors, ws->stream());
//  ws->Output<GPUBackend>(idx + 1)->Copy(offsets, ws->stream());
//}
//
//template <>
//void RandomBBoxCrop<GPUBackend>::WriteBoxesToOutput(
//    DeviceWorkspace *ws, const std::vector<BoundingBoxes> &bounding_boxes,
//    unsigned int idx) {
//  DALI_ENFORCE(bounding_boxes.size() == static_cast<size_t>(batch_size_));
//
//  std::vector<Dims> boxes_shape(static_cast<unsigned long>(batch_size_));
//
//  for (int i = 0; i < batch_size_; ++i) {
//    boxes_shape[i] = {static_cast<long>(bounding_boxes[i].size()), kBboxSize};
//  }
//
//  TensorList<CPUBackend> boxes;
//
//  boxes.Resize(boxes_shape);
//
//  float *boxes_out_data = boxes.template mutable_data<float>();
//
//  for (int i = 0; i < batch_size_; ++i) {
//    const auto box_count = boxes_shape[i][0];
//    for (int j = 0; j < box_count; j++) {
//      auto box = bounding_boxes[i][j];
//      boxes_out_data[0] = box.left;
//      boxes_out_data[1] = box.top;
//      boxes_out_data[2] = ltrb_ ? box.right : box.right - box.left;
//      boxes_out_data[3] = ltrb_ ? box.bottom : box.bottom - box.top;
//      boxes_out_data += kBboxSize;
//    }
//  }
//
//  ws->Output<GPUBackend>(idx + 2)->Copy(boxes, ws->stream());
//}
//
//template <>
//void RandomBBoxCrop<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
//  std::vector<Crop> crop;
//  std::vector<ImageShape> shape;
//  std::vector<BoundingBoxes> boxes_out;
//  crop.reserve(static_cast<unsigned long>(batch_size_));
//  shape.reserve(static_cast<unsigned long>(batch_size_));
//  boxes_out.reserve(static_cast<unsigned long>(batch_size_));
//
//  auto &images = ws->Input<GPUBackend>(idx);
//  const auto &boxes = ws->Input<CPUBackend>(idx + 1);
//
//  auto *box_data = boxes.data<float>();
//
//  auto box_offset = 0;
//
//  for (int i = 0; i < batch_size_; ++i) {
//    const auto box_count = boxes.tensor_shape(i)[0];
//
//    DALI_ENFORCE(boxes.tensor_shape(i)[1] == kBboxSize);
//
//    BoundingBoxes bounding_boxes;
//    bounding_boxes.reserve(static_cast<unsigned long>(box_count));
//
//    for (int j = 0; j < box_count; j++) {
//      auto box = box_data + box_offset;
//      // ltrb expected
//      bounding_boxes.emplace_back(box[0], box[1], box[2], box[3]);
//
//      box_offset += 4;
//    }
//
//    const ImageShape image_shape(images.tensor_shape(i));
//
//    auto prospective_crop = FindProspectiveCrop(image_shape, bounding_boxes,
//                                                SelectMinimumOverlap());
//
//    crop.push_back(prospective_crop.first);
//    shape.push_back(image_shape);
//    boxes_out.emplace_back(prospective_crop.second);
//  }
//
//  WriteCropToOutput(ws, crop, shape, static_cast<unsigned int>(idx));
//  WriteBoxesToOutput(ws, boxes_out, static_cast<unsigned int>(idx));
//}
//
//// Register operator
//DALI_REGISTER_OPERATOR(RandomBBoxCrop, RandomBBoxCrop<GPUBackend>, GPU);

}  // namespace dali
