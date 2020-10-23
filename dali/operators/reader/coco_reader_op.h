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

extern "C" {
#include "third_party/cocoapi/common/maskApi.h"
}

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/coco_loader.h"

namespace dali {
class COCOReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOReader(const OpSpec& spec): DataReader<CPUBackend, ImageLabelWrapper>(spec) {
    ValidateOptions(spec);
  
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");

    output_polygon_masks_ = spec.HasArgument("masks") && !spec.HasArgument("polygon_masks")
                                ? spec.GetArgument<bool>("masks")
                                : spec.GetArgument<bool>("polygon_masks");
    output_polygon_masks_ = spec.GetArgument<bool>("pixelwise_masks");
    DALI_ENFORCE(output_polygon_masks_ + output_polygon_masks_,
      "``polygon_masks`` and ``pixelwise_masks`` are mutually exclusive");

    output_image_ids_ = spec.HasArgument("save_img_ids") && !spec.HasArgument("image_ids")
                                ? spec.GetArgument<bool>("save_img_ids")
                                : spec.GetArgument<bool>("image_ids");

    loader_ = InitLoader<CocoLoader>(
      spec,
      heights_,
      widths_,
      offsets_,
      boxes_,
      labels_,
      counts_,
      masks_meta_,
      mask_coords_,
      masks_rles_,
      masks_rles_idx_,
      read_masks_,
      pixelwise_masks_,
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
    std::memcpy(
      boxes_output.mutable_data<float>(),
      boxes_.data() + 4 * offsets_[image_id],
      counts_[image_id] * 4 * sizeof(float));

    auto &labels_output = ws.Output<CPUBackend>(2);
    labels_output.Resize({counts_[image_id], 1});
    std::memcpy(
      labels_output.mutable_data<int>(),
      labels_.data() + offsets_[image_id],
      counts_[image_id] * sizeof(int));

    int curr_out_idx = 3;
    if (output_polygon_masks_) {
      auto &masks_meta_output = ws.Output<CPUBackend>(curr_out_idx++);
      auto &masks_coords_output = ws.Output<CPUBackend>(curr_out_idx++);

      const auto &meta = masks_meta_[image_id];
      const auto &coords = mask_coords_[image_id];

      masks_meta_output.Resize({static_cast<int64_t>(meta.size()) / 3, 3});
      masks_coords_output.Resize({static_cast<int64_t>(coords.size()) / 2, 2});

      std::memcpy(
        masks_meta_output.mutable_data<int>(),
        meta.data(),
        meta.size() * sizeof(int));
      std::memcpy(
        masks_coords_output.mutable_data<float>(),
        coords.data(),
        coords.size() * sizeof(float));
    }

    if (output_pixelwise_masks_) {
      auto &masks_output = ws.Output<CPUBackend>(curr_out_idx++);
      masks_output.Resize({heights_[image_id], widths_[image_id], 1});
      masks_output.SetLayout("HWC");
      auto masks_out_data = masks_output.mutable_data<int>();
      PixelwiseMasks(image_id, masks_out_data);
    }

    if (output_image_ids_) {
      auto &id_output = ws.Output<CPUBackend>(curr_out_idx++);
      id_output.Resize({1});
      std::memcpy(
        id_output.mutable_data<int>(),
        original_ids_.data() + image_id,
        sizeof(int));
    }
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);

 private:
  std::vector<int> heights_;
  std::vector<int> widths_;
  std::vector<int> offsets_;
  std::vector<float> boxes_;
  std::vector<int> labels_;
  std::vector<int> counts_;

  std::vector<std::vector<int> > masks_meta_;
  std::vector<std::vector<float> > mask_coords_;
  std::vector<std::vector<int> > masks_rles_idx_;
  std::vector<std::vector<std::string> > masks_rles_;

  bool output_polygon_masks_ = false;
  bool output_pixelwise_masks_ = false;
  bool output_image_ids_ = false;

  std::vector<int> original_ids_;

  void ValidateOptions(const OpSpec &spec);
  void PixelwiseMasks(int image_id, int* masks_output);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_COCO_READER_OP_H_
