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

#ifndef DALI_OPERATORS_READER_COCO_READER_OP_H_
#define DALI_OPERATORS_READER_COCO_READER_OP_H_

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <istream>
#include <memory>

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/coco_loader.h"

namespace dali {
class COCOReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOReader(const OpSpec& spec):
    DataReader<CPUBackend, ImageLabelWrapper>(spec),
    read_masks_(spec.GetArgument<bool>("masks")),
    save_img_ids_(spec.GetArgument<bool>("save_img_ids")) {
    ValidateOptions(spec);
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<CocoLoader>(
      spec,
      offsets_,
      boxes_,
      labels_,
      counts_,
      masks_meta_,
      mask_coords_,
      read_masks_,
      save_img_ids_,
      original_ids_,
      shuffle_after_epoch);
  }

  void RunImpl(SampleWorkspace &ws) override {
    const ImageLabelWrapper& image_label = GetSample(ws.data_idx());

    Index image_size = image_label.image.size();
    auto &image_output = ws.Output<CPUBackend>(0);
    int image_id = image_label.label;

    image_output.Resize({image_size});
    image_output.mutable_data<uint8_t>();
    std::memcpy(
      image_output.raw_mutable_data(),
      image_label.image.raw_data(),
      image_size);
    image_output.SetSourceInfo(image_label.image.GetSourceInfo());

    auto &boxes_output = ws.Output<CPUBackend>(1);
    boxes_output.Resize({counts_[image_id], 4});
    auto boxes_out_data = boxes_output.mutable_data<float>();
    memcpy(
      boxes_out_data,
      boxes_.data() + 4 * offsets_[image_id],
      counts_[image_id] * 4 * sizeof(float));

    auto &labels_output = ws.Output<CPUBackend>(2);
    labels_output.Resize({counts_[image_id], 1});
    auto labels_out_data = labels_output.mutable_data<int>();
    memcpy(
      labels_out_data,
      labels_.data() + offsets_[image_id],
      counts_[image_id] * sizeof(int));

    if (read_masks_) {
      auto &masks_meta_output = ws.Output<CPUBackend>(3);
      auto &masks_coords_output = ws.Output<CPUBackend>(4);

      const auto &meta = masks_meta_[image_id];
      const auto &coords = mask_coords_[image_id];

      masks_meta_output.Resize({static_cast<int>(meta.size()) / 3, 3});
      masks_coords_output.Resize({static_cast<int>(coords.size()) / 2, 2});

      auto masks_meta_data = masks_meta_output.mutable_data<int>();
      auto masks_coords_out_data = masks_coords_output.mutable_data<float>();

      std::memcpy(
        masks_meta_data,
        meta.data(),
        meta.size() * sizeof(int));
      std::memcpy(
        masks_coords_out_data,
        coords.data(),
        coords.size() * sizeof(float));
    }

    if (save_img_ids_) {
      auto &id_output = ws.Output<CPUBackend>(3 + 2 * static_cast<int>(read_masks_));
      id_output.Resize({1});
      auto id_out_data = id_output.mutable_data<int>();
      memcpy(
        id_out_data,
        original_ids_.data() + image_id,
        sizeof(int));
    }
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);

 private:
  std::vector<int> offsets_;
  std::vector<float> boxes_;
  std::vector<int> labels_;
  std::vector<int> counts_;

  std::vector<std::vector<int> > masks_meta_;
  std::vector<std::vector<float> > mask_coords_;

  bool read_masks_ = false;

  bool save_img_ids_;
  std::vector<int> original_ids_;

  void ValidateOptions(const OpSpec &spec);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_COCO_READER_OP_H_
